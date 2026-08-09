//! Native dual-fisheye candidate selection and extraction helpers.
//!
//! Candidate frames are paired by their sequence number before scoring.  A
//! frame is never selected independently for one lens: each base-FPS interval
//! produces one shared lens0/lens1 pair, ranked by the conservative minimum of
//! the two lens sharpness scores.  This keeps the two physical cameras on the
//! same timestamp while still repairing a blurry candidate.

use image::{imageops::FilterType, GenericImageView, ImageReader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::fisheye::{ValidRegion, DJI_VALID_RADIUS_RATIO};

/// Fallible result returned by extraction helpers.
pub type ExtractionResult<T> = Result<T, ExtractionError>;

/// Errors from candidate discovery, image scoring, and atomic output commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractionError {
    InvalidInput(String),
    Image(String),
    Io(String),
}

impl fmt::Display for ExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid extraction input: {message}"),
            Self::Image(message) => write!(formatter, "extraction image error: {message}"),
            Self::Io(message) => write!(formatter, "extraction I/O error: {message}"),
        }
    }
}

impl std::error::Error for ExtractionError {}

impl From<io::Error> for ExtractionError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<image::ImageError> for ExtractionError {
    fn from(error: image::ImageError) -> Self {
        Self::Image(error.to_string())
    }
}

/// Candidate/output directories and cadence settings.
#[derive(Debug, Clone)]
pub struct ExtractionRequest {
    pub lens0_candidates: PathBuf,
    pub lens1_candidates: PathBuf,
    pub lens0_output: PathBuf,
    pub lens1_output: PathBuf,
    /// Prefix shared by both lens outputs (for example `source003_`) so
    /// multiple captures can coexist without frame-number collisions.
    pub output_prefix: String,
    pub base_fps: f64,
    /// FPS used when the candidate images were actually decoded.  This can be
    /// the requested base FPS or the dense FPS when blur repair is enabled.
    pub candidate_fps: f64,
    /// Requested dense cadence.  It is retained in metadata; cadence adaptation
    /// itself belongs to the caller because this helper has no motion/IMU input.
    pub dense_fps: f64,
    /// When false, keep the earliest synchronized pair in each interval
    /// without decoding candidates for sharpness scoring.
    pub score_candidates: bool,
    /// When false, select and score candidates without committing copied
    /// outputs.  Selection records and metadata are still written.
    pub copy_selected_outputs: bool,
    pub skip_completed: bool,
    /// Optional output path for the JSON selection metadata.  If omitted, a
    /// `selection.json` sibling of `lens0_output` is used.
    pub metadata_path: Option<PathBuf>,
}

/// Progress emitted once per interval and once per output commit or
/// selection-only completion.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractionProgress {
    pub interval: usize,
    pub total_intervals: usize,
    pub stage: ExtractionStage,
    pub sequence: Option<u64>,
    pub fraction: f32,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionStage {
    Scanning,
    Scoring,
    Writing,
    Skipped,
    Completed,
    Cancelled,
}

/// Individual paired-candidate decision persisted in metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRecord {
    pub interval: usize,
    pub sequence: u64,
    pub lens0_source: PathBuf,
    pub lens1_source: PathBuf,
    pub lens0_score: f64,
    pub lens1_score: f64,
    /// Conservative aggregate (`min(lens0_score, lens1_score)`).
    pub pair_score: f64,
    pub selected: bool,
    pub skipped_existing: bool,
    pub output_lens0: Option<PathBuf>,
    pub output_lens1: Option<PathBuf>,
}

/// JSON payload written atomically after each run (including cancellation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionMetadata {
    pub schema_version: u32,
    pub candidate_storage: String,
    pub base_fps: f64,
    pub candidate_fps: f64,
    pub requested_dense_fps: f64,
    pub sharpness_scoring: bool,
    pub sharpness_analysis_max_dimension: Option<u32>,
    pub copy_selected_outputs: bool,
    /// Whether every interval produced a complete, committed output pair.
    /// Selection-only runs always leave this false.
    pub outputs_committed: bool,
    pub intervals: usize,
    pub cancelled: bool,
    pub selections: Vec<SelectionRecord>,
}

/// Aggregate extraction result.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractionSummary {
    pub total_intervals: usize,
    pub selected_intervals: usize,
    pub skipped_intervals: usize,
    pub cancelled: bool,
    pub metadata_path: PathBuf,
    pub selections: Vec<SelectionRecord>,
}

#[derive(Debug, Clone)]
struct CandidateFrame {
    key: String,
    sequence: u64,
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SharpnessScore {
    pub laplacian_variance: f64,
    pub tenengrad_mean: f64,
    pub combined: f64,
}

/// Longest side used by the sharpness proxy image.
///
/// Sharpness ranking only needs enough samples to distinguish edges; running
/// the derivative loops over an 8K source frame wastes CPU and keeps multiple
/// full-resolution buffers alive.  A 512-pixel bound still gives a dense
/// sample of the fisheye circle while keeping the scoring working set small.
/// With two square lenses this is about 2 * pi * (0.49 * 512)^2 valid samples,
/// roughly the same pixel budget as a 768x512 gs360crop/reference atlas.
pub const SHARPNESS_MAX_DIMENSION: u32 = 512;

/// Select and atomically copy one shared lens pair for each base-FPS interval.
pub fn extract_selected_pairs(
    request: &ExtractionRequest,
    should_cancel: impl Fn() -> bool + Sync,
    on_progress: impl Fn(ExtractionProgress),
) -> ExtractionResult<ExtractionSummary> {
    validate_request(request)?;
    on_progress(ExtractionProgress {
        interval: 0,
        total_intervals: 0,
        stage: ExtractionStage::Scanning,
        sequence: None,
        fraction: 0.0,
        message: "正在掃描雙魚眼配對候選影格".to_string(),
    });

    let lens0 = collect_candidates(&request.lens0_candidates)?;
    let lens1 = collect_candidates(&request.lens1_candidates)?;
    let intervals = build_intervals(&lens0, &lens1, request.candidate_fps, request.base_fps);
    let total_intervals = intervals.len();
    let metadata_path = request
        .metadata_path
        .clone()
        .unwrap_or_else(|| default_metadata_path(&request.lens0_output));
    let mut records = Vec::new();
    let mut selected_intervals = 0usize;
    let mut skipped_intervals = 0usize;
    let mut cancelled = false;

    for (interval_index, (interval, pairs)) in intervals.iter().enumerate() {
        if should_cancel() {
            cancelled = true;
            emit_progress(
                &on_progress,
                interval_index,
                total_intervals,
                ExtractionStage::Cancelled,
                None,
                &format!("已在第 {interval} 個區間前取消"),
            );
            break;
        }

        emit_progress(
            &on_progress,
            interval_index,
            total_intervals,
            ExtractionStage::Scoring,
            None,
            &format!("正在評分 {} 組配對候選影格", pairs.len()),
        );

        let scored = if request.score_candidates {
            // Each candidate is independent and sharpness scoring is CPU-heavy.
            // Rayon reuses its shared work-stealing pool, while the nested join
            // lets both physical lenses use otherwise-idle cores as well.
            pairs
                .par_iter()
                .map(|pair| {
                    if should_cancel() {
                        return Ok(None);
                    }
                    let (lens0_score, lens1_score) = rayon::join(
                        || calculate_sharpness(&pair.0.path),
                        || calculate_sharpness(&pair.1.path),
                    );
                    if should_cancel() {
                        return Ok(None);
                    }
                    let lens0_score = lens0_score?;
                    let lens1_score = lens1_score?;
                    // A pair is only as useful as its blurriest physical lens.
                    // Using `min` prevents one highly textured lens from hiding
                    // a soft mate.
                    let pair_score = lens0_score.combined.min(lens1_score.combined);
                    Ok(Some((pair, lens0_score, lens1_score, pair_score)))
                })
                .collect::<Vec<ExtractionResult<_>>>()
                .into_iter()
                .collect::<ExtractionResult<Vec<_>>>()?
        } else {
            let empty_score = SharpnessScore {
                laplacian_variance: 0.0,
                tenengrad_mean: 0.0,
                combined: 0.0,
            };
            pairs
                .iter()
                .map_while(|pair| {
                    (!should_cancel()).then_some((pair, empty_score, empty_score, 0.0))
                })
                .map(Some)
                .collect()
        };
        let scored = scored.into_iter().flatten().collect::<Vec<_>>();
        cancelled = scored.len() != pairs.len();
        if cancelled {
            emit_progress(
                &on_progress,
                interval_index,
                total_intervals,
                ExtractionStage::Cancelled,
                None,
                "已在評分候選影格時取消",
            );
            break;
        }
        let Some(best_index) = scored
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.3
                    .total_cmp(&right.3)
                    .then_with(|| right.0 .0.sequence.cmp(&left.0 .0.sequence))
            })
            .map(|(index, _)| index)
        else {
            continue;
        };

        for (index, (pair, lens0_score, lens1_score, pair_score)) in scored.iter().enumerate() {
            let selected = index == best_index;
            records.push(SelectionRecord {
                interval: *interval,
                sequence: pair.0.sequence,
                lens0_source: pair.0.path.clone(),
                lens1_source: pair.1.path.clone(),
                lens0_score: lens0_score.combined,
                lens1_score: lens1_score.combined,
                pair_score: *pair_score,
                selected,
                skipped_existing: false,
                output_lens0: None,
                output_lens1: None,
            });
        }

        let (pair, _, _, _) = &scored[best_index];
        if !request.copy_selected_outputs {
            if should_cancel() {
                cancelled = true;
                emit_progress(
                    &on_progress,
                    interval_index,
                    total_intervals,
                    ExtractionStage::Cancelled,
                    Some(pair.0.sequence),
                    "已在選定結果前取消",
                );
                break;
            }
            selected_intervals += 1;
            emit_progress(
                &on_progress,
                interval_index,
                total_intervals,
                ExtractionStage::Completed,
                Some(pair.0.sequence),
                "已選定配對影格（未複製輸出）",
            );
            continue;
        }
        let extension = pair
            .0
            .path
            .extension()
            .and_then(|extension| extension.to_str())
            .ok_or_else(|| {
                ExtractionError::InvalidInput("candidate has no image extension".to_owned())
            })?;
        let lens1_extension = pair
            .1
            .path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default();
        if !extension.eq_ignore_ascii_case(lens1_extension) {
            return Err(ExtractionError::InvalidInput(format!(
                "paired candidates use different formats: {extension} and {lens1_extension}"
            )));
        }
        let output_name = format!(
            "{}{:08}.{}",
            request.output_prefix,
            pair.0.sequence,
            extension.to_ascii_lowercase()
        );
        let output_lens0 = request.lens0_output.join(&output_name);
        let output_lens1 = request.lens1_output.join(&output_name);
        let can_skip = request.skip_completed
            && valid_output_pair(&output_lens0, &pair.0.path)
            && valid_output_pair(&output_lens1, &pair.1.path);
        if can_skip {
            skipped_intervals += 1;
            selected_intervals += 1;
            if let Some(record) = records
                .iter_mut()
                .rev()
                .find(|record| record.interval == *interval && record.selected)
            {
                record.skipped_existing = true;
                record.output_lens0 = Some(output_lens0.clone());
                record.output_lens1 = Some(output_lens1.clone());
            }
            emit_progress(
                &on_progress,
                interval_index,
                total_intervals,
                ExtractionStage::Skipped,
                Some(pair.0.sequence),
                "選定的配對影格已存在，已略過",
            );
            continue;
        }

        if should_cancel() {
            cancelled = true;
            emit_progress(
                &on_progress,
                interval_index,
                total_intervals,
                ExtractionStage::Cancelled,
                Some(pair.0.sequence),
                "已在寫入輸出前取消",
            );
            break;
        }
        emit_progress(
            &on_progress,
            interval_index,
            total_intervals,
            ExtractionStage::Writing,
            Some(pair.0.sequence),
            "正在複製選定的配對影格",
        );
        copy_pair_with_rollback(&pair.0.path, &pair.1.path, &output_lens0, &output_lens1)?;
        selected_intervals += 1;
        if let Some(record) = records
            .iter_mut()
            .rev()
            .find(|record| record.interval == *interval && record.selected)
        {
            record.output_lens0 = Some(output_lens0.clone());
            record.output_lens1 = Some(output_lens1.clone());
        }
        emit_progress(
            &on_progress,
            interval_index,
            total_intervals,
            ExtractionStage::Completed,
            Some(pair.0.sequence),
            "已寫入選定的配對影格",
        );
    }

    if cancelled {
        emit_progress(
            &on_progress,
            total_intervals,
            total_intervals,
            ExtractionStage::Cancelled,
            None,
            "影格擷取已取消",
        );
    }
    let metadata = SelectionMetadata {
        schema_version: 4,
        candidate_storage: "filesystem_images".to_owned(),
        base_fps: request.base_fps,
        candidate_fps: request.candidate_fps,
        requested_dense_fps: request.dense_fps,
        sharpness_scoring: request.score_candidates,
        sharpness_analysis_max_dimension: request
            .score_candidates
            .then_some(SHARPNESS_MAX_DIMENSION),
        copy_selected_outputs: request.copy_selected_outputs,
        outputs_committed: request.copy_selected_outputs
            && !cancelled
            && total_intervals > 0
            && selected_intervals == total_intervals,
        intervals: total_intervals,
        cancelled,
        selections: records.clone(),
    };
    write_selection_metadata_atomic(&metadata_path, &metadata)?;
    if request.copy_selected_outputs && !cancelled && !request.output_prefix.is_empty() {
        cleanup_stale_outputs(request, &records)?;
    }
    Ok(ExtractionSummary {
        total_intervals,
        selected_intervals,
        skipped_intervals,
        cancelled,
        metadata_path,
        selections: records,
    })
}

fn validate_request(request: &ExtractionRequest) -> ExtractionResult<()> {
    for (label, path) in [
        ("lens0 candidate", &request.lens0_candidates),
        ("lens1 candidate", &request.lens1_candidates),
    ] {
        if !path.is_dir() {
            return Err(ExtractionError::InvalidInput(format!(
                "{label} directory does not exist: {}",
                path.display()
            )));
        }
    }
    for (label, value) in [
        ("base_fps", request.base_fps),
        ("candidate_fps", request.candidate_fps),
        ("dense_fps", request.dense_fps),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(ExtractionError::InvalidInput(format!(
                "{label} must be finite and > 0"
            )));
        }
    }
    Ok(())
}

fn collect_candidates(root: &Path) -> ExtractionResult<Vec<CandidateFrame>> {
    let mut paths = Vec::new();
    collect_images(root, &mut paths)?;
    paths.sort_by_key(|path| {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_ascii_lowercase()
    });
    let mut result = Vec::with_capacity(paths.len());
    for path in paths {
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let stem = relative
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                ExtractionError::InvalidInput("candidate has no UTF-8 stem".to_string())
            })?;
        let sequence = trailing_sequence(stem).unwrap_or(0);
        // Numeric frame IDs are the synchronization contract.  Prefixes may
        // differ between lens encoders (`lens0_000123` vs `lens1_000123`), so
        // pair those by sequence rather than by the complete filename.  For a
        // non-numeric test/fixture name, retain the relative stem as fallback.
        let key = match trailing_sequence(stem) {
            Some(sequence) => format!("sequence:{sequence}"),
            None => relative
                .with_extension("")
                .to_string_lossy()
                .replace('\\', "/"),
        };
        result.push(CandidateFrame {
            key,
            sequence,
            path,
        });
    }
    Ok(result)
}

fn collect_images(root: &Path, paths: &mut Vec<PathBuf>) -> ExtractionResult<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_images(&path, paths)?;
        } else if file_type.is_file() && is_image_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("png") | Some("jpg") | Some("jpeg")
    )
}

fn build_intervals(
    lens0: &[CandidateFrame],
    lens1: &[CandidateFrame],
    candidate_fps: f64,
    base_fps: f64,
) -> Vec<(usize, Vec<(CandidateFrame, CandidateFrame)>)> {
    let mut lens0_by_key: BTreeMap<&str, Vec<&CandidateFrame>> = BTreeMap::new();
    let mut lens1_by_key: BTreeMap<&str, Vec<&CandidateFrame>> = BTreeMap::new();
    for candidate in lens0 {
        lens0_by_key
            .entry(candidate.key.as_str())
            .or_default()
            .push(candidate);
    }
    for candidate in lens1 {
        lens1_by_key
            .entry(candidate.key.as_str())
            .or_default()
            .push(candidate);
    }
    // Pair by numeric sequence (or the relative stem fallback for fixtures).
    let mut intervals: BTreeMap<usize, Vec<(CandidateFrame, CandidateFrame)>> = BTreeMap::new();
    for (key, lens0_candidates) in lens0_by_key {
        let Some(lens1_candidates) = lens1_by_key.get(key) else {
            continue;
        };
        for lens0 in lens0_candidates {
            for lens1 in lens1_candidates {
                if lens0.sequence != lens1.sequence {
                    continue;
                }
                // FFmpeg image sequences normally begin at 1.  Saturating
                // subtraction also keeps synthetic/adapter sequences that
                // begin at 0 aligned to the first interval.
                let timestamp = lens0.sequence.saturating_sub(1) as f64 / candidate_fps;
                let interval = (timestamp * base_fps).floor().max(0.0) as usize;
                intervals
                    .entry(interval)
                    .or_default()
                    .push((lens0.clone(), (*lens1).clone()));
            }
        }
    }
    intervals.into_iter().collect()
}

fn trailing_sequence(stem: &str) -> Option<u64> {
    let digits = stem
        .chars()
        .rev()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.chars().rev().collect::<String>().parse().ok()
    }
}

/// Calculate the reference sharpness score: Gaussian pre-blur, Laplacian
/// variance, and Tenengrad Sobel energy.  The combined score is monotonic for
/// the pairwise comparison used by selection, not an absolute focus measure.
pub fn calculate_sharpness(path: &Path) -> ExtractionResult<SharpnessScore> {
    let image = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let image = sharpness_proxy_image(image).to_luma8();
    let width = image.width();
    let height = image.height();
    let pixels = image.into_raw();
    if width == 0 || height == 0 {
        return Ok(zero_sharpness_score());
    }
    calculate_sharpness_from_grayscale(width, height, &pixels)
}

/// Score a packed grayscale frame held in memory.
///
/// `pixels` must contain exactly `width * height` 8-bit grayscale samples in
/// row-major order.  The packed form is convenient when the decoder already
/// produced one 512x512 frame per buffer; callers with a wider composite
/// frame can use [`calculate_sharpness_from_gray8`] and pass its row stride.
pub fn calculate_sharpness_from_grayscale(
    width: u32,
    height: u32,
    pixels: &[u8],
) -> ExtractionResult<SharpnessScore> {
    let width_usize = width as usize;
    let height_usize = height as usize;
    let expected_len = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| ExtractionError::InvalidInput("grayscale dimensions overflow".to_owned()))?;
    if pixels.len() != expected_len {
        return Err(ExtractionError::InvalidInput(format!(
            "packed grayscale buffer has {} bytes; expected {expected_len}",
            pixels.len()
        )));
    }
    calculate_sharpness_from_gray8(width, height, width_usize, pixels)
}

/// Score an in-memory 8-bit grayscale frame with an optional row stride.
///
/// `pixels` points at the first sample of the first row.  Each subsequent row
/// begins `row_stride` bytes later, so a left or right 512x512 lens can be
/// scored directly from a packed 1024x512 composite without writing a proxy
/// JPEG to disk.  The buffer must contain at least
/// `(height - 1) * row_stride + width` bytes; trailing bytes are allowed.
pub fn calculate_sharpness_from_gray8(
    width: u32,
    height: u32,
    row_stride: usize,
    pixels: &[u8],
) -> ExtractionResult<SharpnessScore> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 {
        return Err(ExtractionError::InvalidInput(
            "grayscale dimensions must be greater than zero".to_owned(),
        ));
    }
    if row_stride < width {
        return Err(ExtractionError::InvalidInput(format!(
            "grayscale row stride {row_stride} is smaller than width {width}"
        )));
    }
    let required_len = height
        .saturating_sub(1)
        .checked_mul(row_stride)
        .and_then(|last_row_offset| last_row_offset.checked_add(width))
        .ok_or_else(|| ExtractionError::InvalidInput("grayscale dimensions overflow".to_owned()))?;
    if required_len > pixels.len() {
        return Err(ExtractionError::InvalidInput(format!(
            "grayscale buffer has {} bytes; requires at least {required_len}",
            pixels.len()
        )));
    }
    if width < 5 || height < 5 {
        return Ok(zero_sharpness_score());
    }

    let blurred = gaussian_blur_3x3_with_stride(pixels, width, height, row_stride);
    // Native fisheye frames commonly contain a black/invalid region outside
    // the optical circle.  Ignore that region and a narrow inner border so
    // its hard edge cannot dominate derivative-based scores.
    let valid_region = ValidRegion::new(
        width as u32,
        height as u32,
        DJI_VALID_RADIUS_RATIO * 0.98,
        None,
    );
    let mut laplacian_sum = 0.0f64;
    let mut laplacian_sq_sum = 0.0f64;
    let mut tenengrad_sum = 0.0f64;
    let mut count = 0.0f64;
    for y in 2..(height - 2) {
        let row_offset_squared = valid_region.row_offset_squared(y as u32);
        for x in 2..(width - 2) {
            let index = y * width + x;
            if !valid_region.contains_x(x as u32, y as u32, row_offset_squared) {
                continue;
            }
            let center = blurred[index] as f64;
            let left = blurred[index - 1] as f64;
            let right = blurred[index + 1] as f64;
            let up = blurred[index - width] as f64;
            let down = blurred[index + width] as f64;
            let laplacian = left + right + up + down - 4.0 * center;
            laplacian_sum += laplacian;
            laplacian_sq_sum += laplacian * laplacian;

            let top_left = blurred[index - width - 1] as f64;
            let top_right = blurred[index - width + 1] as f64;
            let bottom_left = blurred[index + width - 1] as f64;
            let bottom_right = blurred[index + width + 1] as f64;
            let gx = top_right + 2.0 * right + bottom_right - top_left - 2.0 * left - bottom_left;
            let gy = bottom_left + 2.0 * down + bottom_right - top_left - 2.0 * up - top_right;
            tenengrad_sum += gx * gx + gy * gy;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return Ok(zero_sharpness_score());
    }
    let laplacian_mean = laplacian_sum / count;
    let laplacian_variance = (laplacian_sq_sum / count - laplacian_mean * laplacian_mean).max(0.0);
    let tenengrad_mean = tenengrad_sum / count;
    Ok(SharpnessScore {
        laplacian_variance,
        tenengrad_mean,
        combined: laplacian_variance.sqrt() + tenengrad_mean.sqrt(),
    })
}

fn zero_sharpness_score() -> SharpnessScore {
    SharpnessScore {
        laplacian_variance: 0.0,
        tenengrad_mean: 0.0,
        combined: 0.0,
    }
}

/// Downscale only high-resolution candidates before the expensive scoring
/// passes.  Typical high-resolution candidates use the image crate's fast
/// integer area-average thumbnail path.  Frames close to the analysis bound
/// use a Triangle filter to avoid thumbnail's documented close-size aliasing
/// caveat, and degenerate aspect ratios retain a five-pixel derivative axis.
/// Small images are returned unchanged so scoring never upscales them.
fn sharpness_proxy_image(image: image::DynamicImage) -> image::DynamicImage {
    let (width, height) = image.dimensions();
    let longest_side = width.max(height);
    if longest_side <= SHARPNESS_MAX_DIMENSION {
        return image;
    }
    let proxy = if longest_side < SHARPNESS_MAX_DIMENSION * 2 {
        image.resize(
            SHARPNESS_MAX_DIMENSION,
            SHARPNESS_MAX_DIMENSION,
            FilterType::Triangle,
        )
    } else {
        image.thumbnail(SHARPNESS_MAX_DIMENSION, SHARPNESS_MAX_DIMENSION)
    };
    if proxy.width() >= 5 && proxy.height() >= 5 {
        return proxy;
    }
    let (target_width, target_height) = if width >= height {
        (SHARPNESS_MAX_DIMENSION, 5)
    } else {
        (5, SHARPNESS_MAX_DIMENSION)
    };
    image.resize_exact(target_width, target_height, FilterType::Triangle)
}

fn gaussian_blur_3x3_with_stride(
    pixels: &[u8],
    width: usize,
    height: usize,
    row_stride: usize,
) -> Vec<f32> {
    let Some(output_len) = width.checked_mul(height) else {
        return Vec::new();
    };
    let mut output = vec![0.0f32; output_len];
    if width < 3 || height < 3 || row_stride < width {
        return output;
    }
    for y in 1..(height - 1) {
        let row_above = (y - 1) * row_stride;
        let row = y * row_stride;
        let row_below = (y + 1) * row_stride;
        for x in 1..(width - 1) {
            let sum = pixels[row_above + x - 1] as u32
                + 2 * pixels[row_above + x] as u32
                + pixels[row_above + x + 1] as u32
                + 2 * pixels[row + x - 1] as u32
                + 4 * pixels[row + x] as u32
                + 2 * pixels[row + x + 1] as u32
                + pixels[row_below + x - 1] as u32
                + 2 * pixels[row_below + x] as u32
                + pixels[row_below + x + 1] as u32;
            output[y * width + x] = sum as f32 / 16.0;
        }
    }
    output
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn stage_copy(source: &Path, destination: &Path) -> ExtractionResult<PathBuf> {
    let parent = destination
        .parent()
        .ok_or_else(|| ExtractionError::InvalidInput("output has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ExtractionError::InvalidInput("output has no UTF-8 filename".to_string()))?;
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staged = parent.join(format!(
        ".{file_name}.{}.{}.part",
        std::process::id(),
        counter
    ));
    let result = (|| -> ExtractionResult<()> {
        fs::copy(source, &staged)?;
        fs::File::open(&staged)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staged);
    }
    result.map(|()| staged)
}

fn backup_existing(destination: &Path) -> ExtractionResult<Option<PathBuf>> {
    if !destination.exists() {
        return Ok(None);
    }
    if !destination.is_file() {
        return Err(ExtractionError::InvalidInput(format!(
            "output destination is not a file: {}",
            destination.display()
        )));
    }
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ExtractionError::InvalidInput("output has no filename".to_string()))?;
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let backup = destination.with_file_name(format!(
        ".{file_name}.{}.{}.old",
        std::process::id(),
        counter
    ));
    let _ = fs::remove_file(&backup);
    fs::rename(destination, &backup)?;
    Ok(Some(backup))
}

fn restore_output(destination: &Path, backup: Option<&Path>) {
    let _ = fs::remove_file(destination);
    if let Some(backup) = backup {
        let _ = fs::rename(backup, destination);
    }
}

/// Stage both physical-lens images before exposing either one. If any normal
/// filesystem error occurs during the final two renames, restore the previous
/// pair so callers never continue with a newly committed single-sided frame.
fn copy_pair_with_rollback(
    lens0_source: &Path,
    lens1_source: &Path,
    lens0_destination: &Path,
    lens1_destination: &Path,
) -> ExtractionResult<()> {
    let lens0_staged = stage_copy(lens0_source, lens0_destination)?;
    let lens1_staged = match stage_copy(lens1_source, lens1_destination) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_file(lens0_staged);
            return Err(error);
        }
    };
    let lens0_backup = match backup_existing(lens0_destination) {
        Ok(backup) => backup,
        Err(error) => {
            let _ = fs::remove_file(lens0_staged);
            let _ = fs::remove_file(lens1_staged);
            return Err(error);
        }
    };
    let lens1_backup = match backup_existing(lens1_destination) {
        Ok(backup) => backup,
        Err(error) => {
            restore_output(lens0_destination, lens0_backup.as_deref());
            let _ = fs::remove_file(lens0_staged);
            let _ = fs::remove_file(lens1_staged);
            return Err(error);
        }
    };

    if let Err(error) = fs::rename(&lens0_staged, lens0_destination) {
        restore_output(lens0_destination, lens0_backup.as_deref());
        restore_output(lens1_destination, lens1_backup.as_deref());
        let _ = fs::remove_file(lens0_staged);
        let _ = fs::remove_file(lens1_staged);
        return Err(error.into());
    }
    if let Err(error) = fs::rename(&lens1_staged, lens1_destination) {
        restore_output(lens0_destination, lens0_backup.as_deref());
        restore_output(lens1_destination, lens1_backup.as_deref());
        let _ = fs::remove_file(lens1_staged);
        return Err(error.into());
    }

    if let Some(backup) = lens0_backup {
        let _ = fs::remove_file(backup);
    }
    if let Some(backup) = lens1_backup {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

fn rename_replace(temporary: &Path, destination: &Path) -> ExtractionResult<()> {
    match fs::rename(temporary, destination) {
        Ok(()) => Ok(()),
        Err(_first_error) if destination.is_file() => {
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let file_name = destination
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    ExtractionError::InvalidInput("output has no filename".to_string())
                })?;
            let backup = destination.with_file_name(format!(
                ".{file_name}.{}.{}.old",
                std::process::id(),
                counter
            ));
            let _ = fs::remove_file(&backup);
            fs::rename(destination, &backup)?;
            match fs::rename(temporary, destination) {
                Ok(()) => {
                    let _ = fs::remove_file(backup);
                    Ok(())
                }
                Err(error) => {
                    let _ = fs::rename(&backup, destination);
                    Err(ExtractionError::from(error))
                }
            }
        }
        Err(error) => Err(ExtractionError::from(error)),
    }
}

fn valid_output_pair(output: &Path, source: &Path) -> bool {
    let Ok(output_image) = ImageReader::open(output)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(ExtractionError::from)
        .and_then(|reader| reader.decode().map_err(ExtractionError::from))
    else {
        return false;
    };
    let Ok(source_image) = ImageReader::open(source)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(ExtractionError::from)
        .and_then(|reader| reader.decode().map_err(ExtractionError::from))
    else {
        return false;
    };
    output_image.dimensions() == source_image.dimensions() && files_equal(output, source)
}

fn files_equal(left: &Path, right: &Path) -> bool {
    let (Ok(left_meta), Ok(right_meta)) = (fs::metadata(left), fs::metadata(right)) else {
        return false;
    };
    if left_meta.len() != right_meta.len() {
        return false;
    }
    let (Ok(mut left_file), Ok(mut right_file)) = (fs::File::open(left), fs::File::open(right))
    else {
        return false;
    };
    let mut left_buffer = [0u8; 64 * 1024];
    let mut right_buffer = [0u8; 64 * 1024];
    loop {
        let (Ok(left_read), Ok(right_read)) = (
            left_file.read(&mut left_buffer),
            right_file.read(&mut right_buffer),
        ) else {
            return false;
        };
        if left_read != right_read || left_buffer[..left_read] != right_buffer[..right_read] {
            return false;
        }
        if left_read == 0 {
            return true;
        }
    }
}

fn cleanup_stale_outputs(
    request: &ExtractionRequest,
    records: &[SelectionRecord],
) -> ExtractionResult<()> {
    let retained = records
        .iter()
        .filter(|record| record.selected)
        .filter_map(|record| record.output_lens0.as_ref()?.file_name()?.to_str())
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    for directory in [&request.lens0_output, &request.lens1_output] {
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if path.is_file()
                && name.starts_with(&request.output_prefix)
                && is_image_path(&path)
                && !retained.contains(name)
            {
                fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

fn default_metadata_path(lens0_output: &Path) -> PathBuf {
    lens0_output
        .parent()
        .unwrap_or(lens0_output)
        .join("selection.json")
}

pub fn write_selection_metadata_atomic(
    path: &Path,
    metadata: &SelectionMetadata,
) -> ExtractionResult<()> {
    let bytes = serde_json::to_vec_pretty(metadata)
        .map_err(|error| ExtractionError::Io(error.to_string()))?;
    write_bytes_atomic(path, &bytes)
}

/// Mark a durable selection record after a later native-resolution commit.
/// The selected sequences must exactly match the first-pass metadata so a
/// caller cannot accidentally mark a different output set as complete.
pub fn mark_selection_outputs_committed(
    path: &Path,
    selected_sequences: &[u64],
    lens0_output: &Path,
    lens1_output: &Path,
    output_prefix: &str,
) -> ExtractionResult<()> {
    let mut metadata: Value = serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| ExtractionError::Io(error.to_string()))?;
    let recorded = metadata
        .get("selections")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ExtractionError::InvalidInput("selection metadata is missing records".to_owned())
        })?
        .iter()
        .filter(|record| record.get("selected").and_then(Value::as_bool) == Some(true))
        .filter_map(|record| record.get("sequence").and_then(Value::as_u64))
        .collect::<BTreeSet<_>>();
    let expected = selected_sequences.iter().copied().collect::<BTreeSet<_>>();
    if recorded != expected || recorded.len() != selected_sequences.len() {
        return Err(ExtractionError::InvalidInput(
            "committed sequences do not match selection metadata".to_owned(),
        ));
    }
    for sequence in selected_sequences {
        let output_name = format!("{output_prefix}{sequence:08}.jpg");
        for output in [
            lens0_output.join(&output_name),
            lens1_output.join(&output_name),
        ] {
            let metadata = fs::metadata(&output)?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(ExtractionError::InvalidInput(format!(
                    "committed output is missing or empty: {}",
                    output.display()
                )));
            }
        }
    }
    let object = metadata.as_object_mut().ok_or_else(|| {
        ExtractionError::InvalidInput("selection metadata must be a JSON object".to_owned())
    })?;
    object.insert("outputs_committed".to_owned(), Value::Bool(true));
    object.insert("copy_selected_outputs".to_owned(), Value::Bool(true));
    object.insert(
        "committed_sequences".to_owned(),
        Value::Array(
            selected_sequences
                .iter()
                .copied()
                .map(Value::from)
                .collect(),
        ),
    );
    let selections = object
        .get_mut("selections")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            ExtractionError::InvalidInput("selection metadata is missing records".to_owned())
        })?;
    for record in selections {
        if record.get("selected").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let sequence = record
            .get("sequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                ExtractionError::InvalidInput("selected record is missing sequence".to_owned())
            })?;
        let output_name = format!("{output_prefix}{sequence:08}.jpg");
        let record = record.as_object_mut().ok_or_else(|| {
            ExtractionError::InvalidInput("selection record must be a JSON object".to_owned())
        })?;
        record.insert(
            "output_lens0".to_owned(),
            Value::String(
                lens0_output
                    .join(&output_name)
                    .to_string_lossy()
                    .into_owned(),
            ),
        );
        record.insert(
            "output_lens1".to_owned(),
            Value::String(
                lens1_output
                    .join(output_name)
                    .to_string_lossy()
                    .into_owned(),
            ),
        );
    }
    let bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|error| ExtractionError::Io(error.to_string()))?;
    write_bytes_atomic(path, &bytes)
}

/// Atomically publish a fully committed pending selection record. Keeping the
/// pending file separate means a cancelled rerun cannot downgrade metadata
/// from the last successful extraction.
pub fn promote_selection_metadata(pending: &Path, destination: &Path) -> ExtractionResult<()> {
    let bytes = fs::read(pending)?;
    write_bytes_atomic(destination, &bytes)
}

pub fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> ExtractionResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| ExtractionError::InvalidInput("metadata has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ExtractionError::InvalidInput("metadata has no filename".to_string()))?;
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.{}.{}.part",
        std::process::id(),
        counter
    ));
    let result = (|| -> ExtractionResult<()> {
        fs::write(&temporary, bytes)?;
        fs::File::open(&temporary)?.sync_all()?;
        rename_replace(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn emit_progress(
    callback: &impl Fn(ExtractionProgress),
    interval: usize,
    total: usize,
    stage: ExtractionStage,
    sequence: Option<u64>,
    message: &str,
) {
    callback(ExtractionProgress {
        interval: interval + 1,
        total_intervals: total,
        stage,
        sequence,
        fraction: if total == 0 {
            0.0
        } else {
            ((interval + 1) as f32 / total as f32).clamp(0.0, 1.0)
        },
        message: message.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};
    use std::sync::atomic::AtomicUsize;
    use tempfile::TempDir;

    fn make_request(dir: &TempDir) -> ExtractionRequest {
        ExtractionRequest {
            lens0_candidates: dir.path().join("candidates/lens0"),
            lens1_candidates: dir.path().join("candidates/lens1"),
            lens0_output: dir.path().join("images/lens0"),
            lens1_output: dir.path().join("images/lens1"),
            output_prefix: String::new(),
            base_fps: 1.0,
            candidate_fps: 4.0,
            dense_fps: 8.0,
            score_candidates: true,
            copy_selected_outputs: true,
            skip_completed: true,
            metadata_path: Some(dir.path().join("metadata/selection.json")),
        }
    }

    fn checker(size: u32) -> GrayImage {
        checker_with_block(size, 2)
    }

    fn checker_with_block(size: u32, block: u32) -> GrayImage {
        let mut image = GrayImage::from_pixel(size, size, Luma([0]));
        for y in 0..size {
            for x in 0..size {
                if (x / block + y / block).is_multiple_of(2) {
                    image.put_pixel(x, y, Luma([255]));
                }
            }
        }
        image
    }

    fn write_pair(request: &ExtractionRequest, sequence: u64, blur: f32) -> ExtractionResult<()> {
        fs::create_dir_all(&request.lens0_candidates)?;
        fs::create_dir_all(&request.lens1_candidates)?;
        let sharp = checker(64);
        let first = if blur > 0.0 {
            image::imageops::blur(&sharp, blur)
        } else {
            sharp.clone()
        };
        let second = if blur > 0.0 {
            image::imageops::blur(&sharp, blur)
        } else {
            sharp
        };
        first.save(request.lens0_candidates.join(format!("{sequence:08}.png")))?;
        second.save(request.lens1_candidates.join(format!("{sequence:08}.png")))?;
        Ok(())
    }

    #[test]
    fn sharpness_proxy_caps_longest_side_without_upscaling() {
        let large = GrayImage::new(4_096, 2_048);
        let proxy = sharpness_proxy_image(DynamicImage::ImageLuma8(large));
        assert_eq!(proxy.dimensions(), (SHARPNESS_MAX_DIMENSION, 256));

        let small = GrayImage::new(320, 180);
        let proxy = sharpness_proxy_image(DynamicImage::ImageLuma8(small));
        assert_eq!(proxy.dimensions(), (320, 180));

        let extreme = GrayImage::new(4_096, 16);
        let proxy = sharpness_proxy_image(DynamicImage::ImageLuma8(extreme));
        assert_eq!(proxy.dimensions(), (SHARPNESS_MAX_DIMENSION, 5));
    }

    #[test]
    fn pair_commit_keeps_existing_outputs_when_second_stage_fails() {
        let dir = TempDir::new().unwrap();
        let lens0_source = dir.path().join("sources/lens0.jpg");
        let missing_lens1_source = dir.path().join("sources/missing-lens1.jpg");
        let lens0_output = dir.path().join("images/lens0/frame.jpg");
        let lens1_output = dir.path().join("images/lens1/frame.jpg");
        fs::create_dir_all(lens0_source.parent().unwrap()).unwrap();
        fs::create_dir_all(lens0_output.parent().unwrap()).unwrap();
        fs::create_dir_all(lens1_output.parent().unwrap()).unwrap();
        fs::write(&lens0_source, b"new-lens0").unwrap();
        fs::write(&lens0_output, b"old-lens0").unwrap();
        fs::write(&lens1_output, b"old-lens1").unwrap();

        assert!(copy_pair_with_rollback(
            &lens0_source,
            &missing_lens1_source,
            &lens0_output,
            &lens1_output,
        )
        .is_err());

        assert_eq!(fs::read(&lens0_output).unwrap(), b"old-lens0");
        assert_eq!(fs::read(&lens1_output).unwrap(), b"old-lens1");
        for directory in [
            lens0_output.parent().unwrap(),
            lens1_output.parent().unwrap(),
        ] {
            assert!(fs::read_dir(directory).unwrap().all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with('.')));
        }
    }

    #[test]
    fn high_resolution_proxy_keeps_sharp_image_above_blurred_image() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let sharp = checker_with_block(2_048, 16);
        let blurred = image::imageops::blur(&sharp, 6.0);
        let sharp_path = dir.path().join("sharp.png");
        let blurred_path = dir.path().join("blurred.png");
        sharp.save(&sharp_path)?;
        blurred.save(&blurred_path)?;

        let sharp_score = calculate_sharpness(&sharp_path)?;
        let blurred_score = calculate_sharpness(&blurred_path)?;
        assert!(
            sharp_score.combined > blurred_score.combined,
            "sharp score {} should exceed blurred score {}",
            sharp_score.combined,
            blurred_score.combined
        );
        Ok(())
    }

    #[test]
    fn in_memory_grayscale_score_matches_path_score() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let image = checker_with_block(SHARPNESS_MAX_DIMENSION, 4);
        let path = dir.path().join("frame.png");
        image.save(&path)?;

        let path_score = calculate_sharpness(&path)?;
        let packed_score =
            calculate_sharpness_from_grayscale(image.width(), image.height(), image.as_raw())?;
        assert_eq!(
            packed_score.laplacian_variance,
            path_score.laplacian_variance
        );
        assert_eq!(packed_score.tenengrad_mean, path_score.tenengrad_mean);
        assert_eq!(packed_score.combined, path_score.combined);
        Ok(())
    }

    #[test]
    fn strided_grayscale_score_matches_packed_and_path_scores() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let image = checker_with_block(SHARPNESS_MAX_DIMENSION, 4);
        let path = dir.path().join("frame.png");
        image.save(&path)?;
        let path_score = calculate_sharpness(&path)?;
        let packed_score =
            calculate_sharpness_from_grayscale(image.width(), image.height(), image.as_raw())?;

        let stride = (image.width() * 2) as usize;
        let mut composite = vec![0u8; stride * image.height() as usize];
        for (row, source) in image
            .as_raw()
            .chunks_exact(image.width() as usize)
            .enumerate()
        {
            let offset = row * stride;
            composite[offset..offset + source.len()].copy_from_slice(source);
            composite[offset + image.width() as usize..offset + stride].copy_from_slice(source);
        }
        let left_score =
            calculate_sharpness_from_gray8(image.width(), image.height(), stride, &composite)?;
        let right_score = calculate_sharpness_from_gray8(
            image.width(),
            image.height(),
            stride,
            &composite[image.width() as usize..],
        )?;

        for score in [left_score, right_score] {
            assert_eq!(score.laplacian_variance, packed_score.laplacian_variance);
            assert_eq!(score.tenengrad_mean, packed_score.tenengrad_mean);
            assert_eq!(score.combined, packed_score.combined);
            assert_eq!(score.combined, path_score.combined);
        }
        Ok(())
    }

    #[test]
    fn grayscale_buffer_validation_rejects_invalid_stride_and_length() {
        assert!(calculate_sharpness_from_grayscale(4, 4, &[0; 15]).is_err());
        assert!(calculate_sharpness_from_gray8(4, 4, 3, &[0; 16]).is_err());
        assert!(calculate_sharpness_from_gray8(4, 4, 4, &[0; 15]).is_err());
    }

    #[test]
    fn selects_shared_sharp_pair_for_both_lenses() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let request = make_request(&dir);
        write_pair(&request, 0, 0.0)?;
        write_pair(&request, 1, 3.0)?;
        let progress_count = AtomicUsize::new(0);
        let summary = extract_selected_pairs(
            &request,
            || false,
            |_| {
                progress_count.fetch_add(1, Ordering::Relaxed);
            },
        )?;
        assert_eq!(summary.selected_intervals, 1);
        let selected = summary
            .selections
            .iter()
            .find(|record| record.selected)
            .unwrap();
        assert_eq!(selected.sequence, 0);
        assert!(selected
            .output_lens0
            .as_ref()
            .unwrap()
            .ends_with("00000000.png"));
        assert!(selected
            .output_lens1
            .as_ref()
            .unwrap()
            .ends_with("00000000.png"));
        assert!(progress_count.load(Ordering::Relaxed) > 0);
        Ok(())
    }

    #[test]
    fn selection_only_mode_scores_without_committing_outputs() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let mut request = make_request(&dir);
        request.copy_selected_outputs = false;
        request.output_prefix = "source007_".to_owned();
        fs::create_dir_all(&request.lens0_candidates)?;
        fs::create_dir_all(&request.lens1_candidates)?;
        checker(64).save(request.lens0_candidates.join("00000000.png"))?;
        // Deliberately use a different extension for the paired lens.  A
        // selection-only run should not validate output extensions.
        checker(64).save(request.lens1_candidates.join("00000000.jpg"))?;

        fs::create_dir_all(&request.lens0_output)?;
        fs::create_dir_all(&request.lens1_output)?;
        let stale0 = request.lens0_output.join("source007_99999999.png");
        let stale1 = request.lens1_output.join("source007_99999999.png");
        fs::copy(request.lens0_candidates.join("00000000.png"), &stale0)?;
        fs::copy(request.lens1_candidates.join("00000000.jpg"), &stale1)?;

        let summary = extract_selected_pairs(&request, || false, |_| {})?;
        assert_eq!(summary.total_intervals, 1);
        assert_eq!(summary.selected_intervals, 1);
        assert_eq!(summary.skipped_intervals, 0);
        let selected = summary
            .selections
            .iter()
            .find(|record| record.selected)
            .unwrap();
        assert_eq!(selected.sequence, 0);
        assert!(selected.lens0_score > 0.0);
        assert!(selected.lens1_score > 0.0);
        assert!(selected.output_lens0.is_none());
        assert!(selected.output_lens1.is_none());
        assert!(!request.lens0_output.join("source007_00000000.png").exists());
        assert!(!request.lens1_output.join("source007_00000000.jpg").exists());
        assert!(stale0.is_file());
        assert!(stale1.is_file());

        let metadata: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(request.metadata_path.clone().unwrap())?)
                .map_err(|error| ExtractionError::Io(error.to_string()))?;
        assert_eq!(metadata["schema_version"], 4);
        assert_eq!(metadata["candidate_storage"], "filesystem_images");
        assert_eq!(metadata["copy_selected_outputs"], false);
        assert_eq!(metadata["outputs_committed"], false);
        assert_eq!(metadata["intervals"], 1);
        assert_eq!(metadata["selections"][0]["selected"], true);
        assert!(metadata["selections"][0]["pair_score"]
            .as_f64()
            .is_some_and(|score| score > 0.0));
        let metadata_path = request.metadata_path.clone().unwrap();
        assert!(mark_selection_outputs_committed(
            &metadata_path,
            &[1],
            &request.lens0_output,
            &request.lens1_output,
            &request.output_prefix,
        )
        .is_err());
        checker(64).save(request.lens0_output.join("source007_00000000.jpg"))?;
        checker(64).save(request.lens1_output.join("source007_00000000.jpg"))?;
        mark_selection_outputs_committed(
            &metadata_path,
            &[0],
            &request.lens0_output,
            &request.lens1_output,
            &request.output_prefix,
        )?;
        let committed: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(metadata_path)?)
                .map_err(|error| ExtractionError::Io(error.to_string()))?;
        assert_eq!(committed["outputs_committed"], true);
        assert_eq!(committed["copy_selected_outputs"], true);
        assert_eq!(committed["committed_sequences"], serde_json::json!([0]));
        assert!(committed["selections"][0]["output_lens0"]
            .as_str()
            .is_some_and(|path| path.ends_with("source007_00000000.jpg")));
        Ok(())
    }

    #[test]
    fn pair_score_uses_conservative_lens_minimum() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let request = make_request(&dir);
        write_pair(&request, 0, 0.0)?;
        // Make only lens1 blurry for sequence 1: the pair must score below the
        // fully sharp pair even though lens0 remains textured.
        fs::create_dir_all(&request.lens0_candidates)?;
        fs::create_dir_all(&request.lens1_candidates)?;
        checker(64).save(request.lens0_candidates.join("00000001.png"))?;
        image::imageops::blur(&checker(64), 3.0)
            .save(request.lens1_candidates.join("00000001.png"))?;
        let summary = extract_selected_pairs(&request, || false, |_| {})?;
        let selected = summary
            .selections
            .iter()
            .find(|record| record.selected)
            .unwrap();
        assert_eq!(selected.sequence, 0);
        Ok(())
    }

    #[test]
    fn parallel_scoring_preserves_the_earliest_tie_break() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let request = make_request(&dir);
        write_pair(&request, 0, 0.0)?;
        write_pair(&request, 1, 0.0)?;
        let summary = extract_selected_pairs(&request, || false, |_| {})?;
        let selected = summary
            .selections
            .iter()
            .find(|record| record.selected)
            .unwrap();
        assert_eq!(selected.sequence, 0);
        Ok(())
    }

    #[test]
    fn pairs_numeric_sequence_even_when_lens_prefixes_differ() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let request = make_request(&dir);
        fs::create_dir_all(&request.lens0_candidates)?;
        fs::create_dir_all(&request.lens1_candidates)?;
        checker(64).save(request.lens0_candidates.join("front_00000000.png"))?;
        checker(64).save(request.lens1_candidates.join("rear_00000000.png"))?;
        let summary = extract_selected_pairs(&request, || false, |_| {})?;
        assert_eq!(summary.selected_intervals, 1);
        assert_eq!(
            summary
                .selections
                .iter()
                .find(|record| record.selected)
                .unwrap()
                .sequence,
            0
        );
        Ok(())
    }

    #[test]
    fn resume_skips_valid_pair_and_rewrites_missing_mate() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let request = make_request(&dir);
        write_pair(&request, 0, 0.0)?;
        let first = extract_selected_pairs(&request, || false, |_| {})?;
        assert_eq!(first.skipped_intervals, 0);
        let second = extract_selected_pairs(&request, || false, |_| {})?;
        assert_eq!(second.skipped_intervals, 1);
        fs::remove_file(request.lens1_output.join("00000000.png"))?;
        let third = extract_selected_pairs(&request, || false, |_| {})?;
        assert_eq!(third.skipped_intervals, 0);
        assert!(request.lens1_output.join("00000000.png").is_file());
        let metadata = fs::read_to_string(&request.metadata_path.clone().unwrap())?;
        assert!(metadata.contains("requested_dense_fps"));
        assert!(metadata.contains("\"schema_version\": 4"));
        assert!(metadata.contains("\"candidate_storage\": \"filesystem_images\""));
        assert!(metadata.contains("\"sharpness_analysis_max_dimension\": 512"));
        assert!(metadata.contains("\"copy_selected_outputs\": true"));
        assert!(metadata.contains("\"outputs_committed\": true"));
        // Ensure the output remains a valid PNG after a resume copy.
        let _ = ImageReader::open(request.lens0_output.join("00000000.png"))?
            .with_guessed_format()?
            .decode()?;
        Ok(())
    }

    #[test]
    fn cancellation_writes_cancelled_metadata_without_outputs() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let request = make_request(&dir);
        write_pair(&request, 0, 0.0)?;
        let cancelled = AtomicUsize::new(0);
        let summary = extract_selected_pairs(
            &request,
            || cancelled.fetch_add(1, Ordering::Relaxed) > 1,
            |_| {},
        )?;
        assert!(summary.cancelled);
        assert!(summary.metadata_path.is_file());
        assert!(!request.lens0_output.join("00000000.png").exists());
        Ok(())
    }

    #[test]
    fn cancellation_still_works_when_scoring_is_disabled() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let mut request = make_request(&dir);
        request.score_candidates = false;
        write_pair(&request, 0, 0.0)?;
        let cancellation_checks = AtomicUsize::new(0);
        let summary = extract_selected_pairs(
            &request,
            || cancellation_checks.fetch_add(1, Ordering::Relaxed) > 0,
            |_| {},
        )?;
        assert!(summary.cancelled);
        assert!(summary.selections.is_empty());
        assert!(!request.lens0_output.join("00000000.png").exists());
        assert!(!request.lens1_output.join("00000000.png").exists());
        Ok(())
    }

    #[test]
    fn prefixed_outputs_keep_sources_isolated_and_clean_stale_files() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let mut request = make_request(&dir);
        request.output_prefix = "source007_".to_string();
        request.score_candidates = false;
        write_pair(&request, 0, 3.0)?;
        write_pair(&request, 1, 0.0)?;
        let first = extract_selected_pairs(&request, || false, |_| {})?;
        let selected = first
            .selections
            .iter()
            .find(|record| record.selected)
            .unwrap();
        assert_eq!(
            selected.sequence, 0,
            "scoring disabled keeps the earliest pair"
        );
        let stale0 = request.lens0_output.join("source007_99999999.png");
        let stale1 = request.lens1_output.join("source007_99999999.png");
        fs::copy(&request.lens0_candidates.join("00000000.png"), &stale0)?;
        fs::copy(&request.lens1_candidates.join("00000000.png"), &stale1)?;
        let _ = extract_selected_pairs(&request, || false, |_| {})?;
        assert!(!stale0.exists());
        assert!(!stale1.exists());
        assert!(request
            .lens0_output
            .join("source007_00000000.png")
            .is_file());
        assert!(request
            .lens1_output
            .join("source007_00000000.png")
            .is_file());
        Ok(())
    }

    #[test]
    fn jpeg_candidates_keep_jpeg_outputs_and_resume() -> ExtractionResult<()> {
        let dir = TempDir::new()?;
        let mut request = make_request(&dir);
        request.output_prefix = "source000_".to_owned();
        request.score_candidates = false;
        fs::create_dir_all(&request.lens0_candidates)?;
        fs::create_dir_all(&request.lens1_candidates)?;
        checker(64).save(request.lens0_candidates.join("00000001.jpg"))?;
        checker(64).save(request.lens1_candidates.join("00000001.jpg"))?;

        let first = extract_selected_pairs(&request, || false, |_| {})?;
        assert_eq!(first.selected_intervals, 1);
        assert!(request
            .lens0_output
            .join("source000_00000001.jpg")
            .is_file());
        assert!(request
            .lens1_output
            .join("source000_00000001.jpg")
            .is_file());
        let second = extract_selected_pairs(&request, || false, |_| {})?;
        assert_eq!(second.skipped_intervals, 1);
        Ok(())
    }
}
