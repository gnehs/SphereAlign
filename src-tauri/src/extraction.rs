//! Native dual-fisheye candidate selection and extraction helpers.
//!
//! Candidate frames are paired by their sequence number before scoring.  A
//! frame is never selected independently for one lens: each base-FPS interval
//! produces one shared lens0/lens1 pair, ranked by the conservative minimum of
//! the two lens sharpness scores.  This keeps the two physical cameras on the
//! same timestamp while still repairing a blurry candidate.

use image::{GenericImageView, ImageReader};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
    pub skip_completed: bool,
    /// Optional output path for the JSON selection metadata.  If omitted, a
    /// `selection.json` sibling of `lens0_output` is used.
    pub metadata_path: Option<PathBuf>,
}

/// Progress emitted once per interval and once per output commit.
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
#[derive(Debug, Clone, Serialize)]
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
#[derive(Debug, Clone, Serialize)]
pub struct SelectionMetadata {
    pub schema_version: u32,
    pub base_fps: f64,
    pub candidate_fps: f64,
    pub requested_dense_fps: f64,
    pub sharpness_scoring: bool,
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

/// Select and atomically copy one shared lens pair for each base-FPS interval.
pub fn extract_selected_pairs(
    request: &ExtractionRequest,
    should_cancel: impl Fn() -> bool,
    on_progress: impl Fn(ExtractionProgress),
) -> ExtractionResult<ExtractionSummary> {
    validate_request(request)?;
    on_progress(ExtractionProgress {
        interval: 0,
        total_intervals: 0,
        stage: ExtractionStage::Scanning,
        sequence: None,
        fraction: 0.0,
        message: "scanning paired fisheye candidates".to_string(),
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
                &format!("cancelled before interval {interval}"),
            );
            break;
        }

        emit_progress(
            &on_progress,
            interval_index,
            total_intervals,
            ExtractionStage::Scoring,
            None,
            &format!("scoring {} paired candidates", pairs.len()),
        );

        let mut scored = Vec::with_capacity(pairs.len());
        for pair in pairs {
            if should_cancel() {
                cancelled = true;
                break;
            }
            let lens0_score = if request.score_candidates {
                calculate_sharpness(&pair.0.path)?
            } else {
                SharpnessScore {
                    laplacian_variance: 0.0,
                    tenengrad_mean: 0.0,
                    combined: 0.0,
                }
            };
            if should_cancel() {
                cancelled = true;
                break;
            }
            let lens1_score = if request.score_candidates {
                calculate_sharpness(&pair.1.path)?
            } else {
                SharpnessScore {
                    laplacian_variance: 0.0,
                    tenengrad_mean: 0.0,
                    combined: 0.0,
                }
            };
            // A pair is only as useful as its blurriest physical lens.  Using
            // `min` prevents one highly textured lens from hiding a soft mate.
            let pair_score = lens0_score.combined.min(lens1_score.combined);
            scored.push((pair, lens0_score, lens1_score, pair_score));
        }
        if cancelled {
            emit_progress(
                &on_progress,
                interval_index,
                total_intervals,
                ExtractionStage::Cancelled,
                None,
                "cancelled while scoring candidates",
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
                "selected pair already exists; skipped",
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
                "cancelled before output commit",
            );
            break;
        }
        emit_progress(
            &on_progress,
            interval_index,
            total_intervals,
            ExtractionStage::Writing,
            Some(pair.0.sequence),
            "copying selected pair",
        );
        copy_atomic(&pair.0.path, &output_lens0)?;
        copy_atomic(&pair.1.path, &output_lens1)?;
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
            "selected pair committed",
        );
    }

    if cancelled {
        emit_progress(
            &on_progress,
            total_intervals,
            total_intervals,
            ExtractionStage::Cancelled,
            None,
            "extraction cancelled",
        );
    }
    let metadata = SelectionMetadata {
        schema_version: 1,
        base_fps: request.base_fps,
        candidate_fps: request.candidate_fps,
        requested_dense_fps: request.dense_fps,
        sharpness_scoring: request.score_candidates,
        intervals: total_intervals,
        cancelled,
        selections: records.clone(),
    };
    write_metadata_atomic(&metadata_path, &metadata)?;
    if !cancelled && !request.output_prefix.is_empty() {
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
    let image = ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .to_luma8();
    let width = image.width() as usize;
    let height = image.height() as usize;
    let pixels = image.into_raw();
    if width < 5 || height < 5 || pixels.len() != width.saturating_mul(height) {
        return Ok(SharpnessScore {
            laplacian_variance: 0.0,
            tenengrad_mean: 0.0,
            combined: 0.0,
        });
    }
    let blurred = gaussian_blur_3x3(&pixels, width, height);
    // Native fisheye frames commonly contain a black/invalid region outside
    // the optical circle.  Ignore that region and a narrow inner border so
    // its hard edge cannot dominate derivative-based scores.
    let valid_radius = (width.min(height) as f64 * 0.497 * 0.98).max(1.0);
    let center_x = (width.saturating_sub(1) as f64) * 0.5;
    let center_y = (height.saturating_sub(1) as f64) * 0.5;
    let radius_squared = valid_radius * valid_radius;
    let mut laplacian_sum = 0.0f64;
    let mut laplacian_sq_sum = 0.0f64;
    let mut tenengrad_sum = 0.0f64;
    let mut count = 0.0f64;
    for y in 2..(height - 2) {
        for x in 2..(width - 2) {
            let index = y * width + x;
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;
            if dx * dx + dy * dy > radius_squared {
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
        return Ok(SharpnessScore {
            laplacian_variance: 0.0,
            tenengrad_mean: 0.0,
            combined: 0.0,
        });
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

fn gaussian_blur_3x3(pixels: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; pixels.len()];
    if width < 3 || height < 3 || pixels.len() != width.saturating_mul(height) {
        return output;
    }
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let sum = pixels[(y - 1) * width + x - 1] as u32
                + 2 * pixels[(y - 1) * width + x] as u32
                + pixels[(y - 1) * width + x + 1] as u32
                + 2 * pixels[y * width + x - 1] as u32
                + 4 * pixels[y * width + x] as u32
                + 2 * pixels[y * width + x + 1] as u32
                + pixels[(y + 1) * width + x - 1] as u32
                + 2 * pixels[(y + 1) * width + x] as u32
                + pixels[(y + 1) * width + x + 1] as u32;
            output[y * width + x] = sum as f32 / 16.0;
        }
    }
    output
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn copy_atomic(source: &Path, destination: &Path) -> ExtractionResult<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| ExtractionError::InvalidInput("output has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ExtractionError::InvalidInput("output has no UTF-8 filename".to_string()))?;
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.{}.{}.part",
        std::process::id(),
        counter
    ));
    let result = (|| -> ExtractionResult<()> {
        fs::copy(source, &temporary)?;
        fs::File::open(&temporary)?.sync_all()?;
        rename_replace(&temporary, destination)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
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

fn write_metadata_atomic(path: &Path, metadata: &SelectionMetadata) -> ExtractionResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| ExtractionError::InvalidInput("metadata has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    let bytes = serde_json::to_vec_pretty(metadata)
        .map_err(|error| ExtractionError::Io(error.to_string()))?;
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
    use image::{GrayImage, Luma};
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
            skip_completed: true,
            metadata_path: Some(dir.path().join("metadata/selection.json")),
        }
    }

    fn checker(size: u32) -> GrayImage {
        let mut image = GrayImage::from_pixel(size, size, Luma([0]));
        for y in 0..size {
            for x in 0..size {
                if (x / 2 + y / 2) % 2 == 0 {
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
