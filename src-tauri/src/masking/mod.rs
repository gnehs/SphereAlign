//! Native-fisheye instance masks for the studio backend.
//!
//! The module treats every image below `images_dir` (including `lens0/`,
//! `lens1/`, and future `lens*` directories) as an independent camera frame.
//! `masks/` keeps the exact relative filename and uses white for pixels retained
//! by SfM/training and black for excluded pixels.  `masks_colmap/` keeps the same
//! relative hierarchy but appends `.png` to the complete image filename (for
//! example `lens0/frame.jpg.png`), matching COLMAP's `ImageReader.mask_path`
//! contract. Pixels outside the full fisheye circle and, when present, below
//! DJI's calibrated fixed optical-occlusion curve are black. Scene content such
//! as hands and selfie sticks is left to the separate semantic mask stage.
//!
//! Model loading is lazy and explicit: [`process_mask_batch`] discovers YOLO11
//! and (when requested) skyseg models from `MaskRequest::model_dir`, explicit
//! model paths, `GS360_MODEL_DIR`, `models/`, or `.models/`.  A backend can use
//! [`process_mask_batch_with_engine`] to inject a preloaded model/session (useful
//! for a long-running Tauri worker and for tests).

mod inference;
mod models;
mod skyseg;

pub use inference::YoloSegPipeline;
pub use models::{ModelDownloadProgress, ModelPaths};
pub use skyseg::SkysegPipeline;

use image::imageops::FilterType;
use image::{
    ColorType, DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageReader, Luma,
};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::fisheye::{
    LensOpticalOcclusions, OpticalOcclusion, ValidRegion, DJI_VALID_RADIUS_RATIO,
};

/// Four in-flight images keep decode, GPU inference, post-processing, and
/// durable writes overlapped without allowing full-resolution buffers to grow
/// with the host's CPU count.
const MASK_PIPELINE_WORKERS: usize = 4;
/// Match the fixed detection gate validated by the original gs360masker pipeline.
pub(crate) const YOLO_CONFIDENCE_THRESHOLD: f32 = 0.25;
/// Semantic exclusions do not need source-image precision. Keep all model
/// preprocessing, mask merging, and valid-region composition bounded, then
/// expand the final binary mask exactly once for COLMAP/source compatibility.
const MASK_WORKING_LONG_EDGE: u32 = 640;

fn mask_pipeline_worker_limit() -> usize {
    #[cfg(test)]
    if let Some(value) = std::env::var("GS360_TEST_PIPELINE_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        return value;
    }
    MASK_PIPELINE_WORKERS
}

/// A fallible result returned by this module.
pub type MaskResult<T> = Result<T, MaskError>;

/// Errors surfaced by model loading, image processing, and output commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskError {
    InvalidInput(String),
    Model(String),
    Inference(String),
    Image(String),
    Io(String),
    Cancelled,
}

impl MaskError {
    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    pub(crate) fn model(message: impl Into<String>) -> Self {
        Self::Model(message.into())
    }

    pub(crate) fn inference(message: impl Into<String>) -> Self {
        Self::Inference(message.into())
    }

    pub(crate) fn image(message: impl Into<String>) -> Self {
        Self::Image(message.into())
    }
}

impl fmt::Display for MaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid mask input: {message}"),
            Self::Model(message) => write!(formatter, "mask model error: {message}"),
            Self::Inference(message) => write!(formatter, "mask inference error: {message}"),
            Self::Image(message) => write!(formatter, "mask image error: {message}"),
            Self::Io(message) => write!(formatter, "mask I/O error: {message}"),
            Self::Cancelled => formatter.write_str("mask operation cancelled"),
        }
    }
}

impl std::error::Error for MaskError {}

impl From<io::Error> for MaskError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<image::ImageError> for MaskError {
    fn from(error: image::ImageError) -> Self {
        Self::Image(error.to_string())
    }
}

/// Cancellation token shared by the Tauri command and the inference worker.
#[derive(Clone, Debug)]
pub struct CancelToken(Arc<AtomicBool>);

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// User and output configuration for a native-fisheye mask run.
#[derive(Debug, Clone)]
pub struct MaskRequest {
    pub images_dir: PathBuf,
    pub masks_dir: PathBuf,
    pub colmap_masks_dir: PathBuf,
    pub classes: Vec<String>,
    pub mask_sky: bool,
    pub confidence: f32,
    /// Radius as a ratio of the shorter source dimension.
    pub valid_radius_ratio: f32,
    /// DJI optical calibration keyed by extraction filename prefix.
    pub optical_occlusions: BTreeMap<String, LensOpticalOcclusions>,
    /// Skip only when both output files decode and match the source dimensions.
    pub skip_verified: bool,
    /// Optional user-supplied model root. See [`ModelPaths::resolve`].
    pub model_dir: Option<PathBuf>,
    /// Application-owned directory used for verified first-use downloads.
    pub model_cache_dir: Option<PathBuf>,
    /// Optional explicit YOLO11 model path.
    pub yolo_model: Option<PathBuf>,
    /// Optional explicit skyseg model path.
    pub skyseg_model: Option<PathBuf>,
    /// Provider name (`CUDA`, `CoreML`, `DirectML`, `CPU`, …).
    pub execution_provider: Option<String>,
}

impl Default for MaskRequest {
    fn default() -> Self {
        Self {
            images_dir: PathBuf::new(),
            masks_dir: PathBuf::new(),
            colmap_masks_dir: PathBuf::new(),
            classes: Vec::new(),
            mask_sky: false,
            confidence: YOLO_CONFIDENCE_THRESHOLD,
            valid_radius_ratio: DJI_VALID_RADIUS_RATIO as f32,
            optical_occlusions: BTreeMap::new(),
            skip_verified: true,
            model_dir: None,
            model_cache_dir: None,
            yolo_model: None,
            skyseg_model: None,
            execution_provider: None,
        }
    }
}

/// Per-file progress emitted to the studio backend.
#[derive(Debug, Clone, Serialize)]
pub struct MaskProgress {
    pub index: usize,
    pub total: usize,
    /// Number of files whose terminal outcome has been recorded.  This is
    /// independent from `index`, because inference runs concurrently.
    pub completed: usize,
    pub input: PathBuf,
    pub mask_path: PathBuf,
    pub colmap_mask_path: PathBuf,
    pub stage: MaskStage,
    pub fraction: f32,
    pub message: String,
}

/// Stable progress stage names for frontend events.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskStage {
    Discovering,
    LoadingModel,
    Inference,
    Writing,
    Skipped,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaskFailure {
    pub input: PathBuf,
    pub error: String,
}

/// Aggregate result.  `cancelled` is true when the token was set before all
/// files completed; already committed files remain counted as succeeded/skipped.
#[derive(Debug, Clone, Serialize)]
pub struct MaskSummary {
    pub total: usize,
    pub succeeded: usize,
    pub skipped: usize,
    pub failed: usize,
    pub cancelled: bool,
    pub failures: Vec<MaskFailure>,
}

/// A binary exclusion mask returned by a segmentation backend (255 = remove).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentationMask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl SegmentationMask {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> MaskResult<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| MaskError::invalid_input("mask dimensions overflow"))?;
        if data.len() != expected {
            return Err(MaskError::invalid_input(format!(
                "mask data length {} does not match {}x{}",
                data.len(),
                width,
                height
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

/// Inference abstraction.  Implement this for a preloaded provider/session and
/// call [`process_mask_batch_with_engine`] to avoid loading a model per command.
pub trait MaskEngine: Send + Sync {
    fn generate_exclusion_mask(
        &self,
        image: &DynamicImage,
        classes: &[String],
        confidence: f32,
        mask_sky: bool,
        cancel: &CancelToken,
    ) -> MaskResult<SegmentationMask>;
}

/// Native YOLO11 + skyseg backend.
pub struct NativeMaskEngine {
    pub yolo: Option<YoloSegPipeline>,
    pub skyseg: Option<SkysegPipeline>,
    pub execution_provider: String,
}

impl NativeMaskEngine {
    pub fn load(
        request: &MaskRequest,
        cancel: &CancelToken,
        on_download: &dyn Fn(ModelDownloadProgress),
    ) -> MaskResult<Self> {
        let paths = ModelPaths::resolve(
            request.model_dir.as_deref(),
            request.model_cache_dir.as_deref(),
            request.yolo_model.as_deref(),
            request.skyseg_model.as_deref(),
            !request.classes.is_empty(),
            request.mask_sky,
            cancel,
            on_download,
        )?;
        let yolo = if request.classes.is_empty() {
            None
        } else {
            Some(YoloSegPipeline::load(
                &paths,
                request.execution_provider.as_deref(),
            )?)
        };
        let skyseg = if request.mask_sky {
            Some(match yolo.as_ref() {
                Some(yolo) => SkysegPipeline::load(&paths, &yolo.execution_provider)?,
                None => {
                    SkysegPipeline::load_available(&paths, request.execution_provider.as_deref())?
                }
            })
        } else {
            None
        };
        let execution_provider = yolo
            .as_ref()
            .map(|pipeline| pipeline.execution_provider.clone())
            .or_else(|| {
                skyseg
                    .as_ref()
                    .map(|pipeline| pipeline.execution_provider.clone())
            })
            .ok_or_else(|| MaskError::invalid_input("at least one mask model must be enabled"))?;
        Ok(Self {
            yolo,
            skyseg,
            execution_provider,
        })
    }
}

impl MaskEngine for NativeMaskEngine {
    fn generate_exclusion_mask(
        &self,
        image: &DynamicImage,
        classes: &[String],
        confidence: f32,
        mask_sky: bool,
        cancel: &CancelToken,
    ) -> MaskResult<SegmentationMask> {
        let mut result = if classes.is_empty() {
            let (width, height) = image.dimensions();
            SegmentationMask::new(width, height, vec![0; (width * height) as usize])?
        } else {
            self.yolo
                .as_ref()
                .ok_or_else(|| {
                    MaskError::model("YOLO11 backend is not loaded but classes are enabled")
                })?
                .generate_exclusion_mask(image, classes, confidence, cancel)?
        };
        if mask_sky {
            let skyseg = self.skyseg.as_ref().ok_or_else(|| {
                MaskError::model("skyseg backend is not loaded but mask_sky is true")
            })?;
            let sky = skyseg.generate_exclusion_mask(image, cancel)?;
            for (destination, source) in result.data.iter_mut().zip(sky.data) {
                if source != 0 {
                    *destination = 255;
                }
            }
        }
        Ok(result)
    }
}

/// Process all source images using a freshly discovered native model backend.
///
/// The callback is intentionally `Fn` so it can be a Tauri `Channel::send`
/// closure without requiring mutable state at the command boundary.
pub fn process_mask_batch(
    request: &MaskRequest,
    cancel: &CancelToken,
    on_progress: impl Fn(MaskProgress) + Sync,
) -> MaskResult<MaskSummary> {
    validate_request(request)?;
    if request.skip_verified {
        if let Some(summary) = skip_fully_verified_batch(request, cancel, &on_progress)? {
            return Ok(summary);
        }
    }
    if request.classes.is_empty() && !request.mask_sky {
        return process_with_engine(request, cancel, &NoExclusionsEngine, on_progress);
    }
    on_progress(MaskProgress {
        index: 0,
        total: 0,
        completed: 0,
        input: request.images_dir.clone(),
        mask_path: request.masks_dir.clone(),
        colmap_mask_path: request.colmap_masks_dir.clone(),
        stage: MaskStage::Discovering,
        fraction: 0.0,
        message: "正在掃描原生雙魚眼影像".to_string(),
    });
    on_progress(MaskProgress {
        index: 0,
        total: 0,
        completed: 0,
        input: request.images_dir.clone(),
        mask_path: request.masks_dir.clone(),
        colmap_mask_path: request.colmap_masks_dir.clone(),
        stage: MaskStage::LoadingModel,
        fraction: 0.0,
        message: "正在載入 YOLO11／SkySeg 模型".to_string(),
    });
    let engine = NativeMaskEngine::load(request, cancel, &|event| {
        let percent = event
            .downloaded
            .saturating_mul(100)
            .checked_div(event.total)
            .unwrap_or(0);
        on_progress(MaskProgress {
            index: 0,
            total: 0,
            completed: 0,
            input: request.images_dir.clone(),
            mask_path: request.masks_dir.clone(),
            colmap_mask_path: request.colmap_masks_dir.clone(),
            stage: MaskStage::LoadingModel,
            fraction: 0.0,
            message: format!("首次使用，正在下載 {} 模型（{}%）", event.label, percent),
        });
    })?;
    on_progress(MaskProgress {
        index: 0,
        total: 0,
        completed: 0,
        input: request.images_dir.clone(),
        mask_path: request.masks_dir.clone(),
        colmap_mask_path: request.colmap_masks_dir.clone(),
        stage: MaskStage::LoadingModel,
        fraction: 0.0,
        message: format!(
            "模型已載入 {}，CPU 推論回退已停用",
            engine.execution_provider
        ),
    });
    process_with_engine(request, cancel, &engine, on_progress)
}

/// Return a completed summary before model discovery when every output pair is
/// already usable. No progress is emitted until the full batch is verified, so
/// a partial batch still follows the normal inference path.
fn skip_fully_verified_batch(
    request: &MaskRequest,
    cancel: &CancelToken,
    on_progress: &impl Fn(MaskProgress),
) -> MaskResult<Option<MaskSummary>> {
    let files = collect_images(&request.images_dir)?;
    let total = files.len();
    let summary = |skipped, cancelled| MaskSummary {
        total,
        succeeded: skipped,
        skipped,
        failed: 0,
        cancelled,
        failures: Vec::new(),
    };
    if cancel.is_cancelled() {
        return Ok(Some(summary(0, true)));
    }

    let mut outputs = Vec::with_capacity(total);
    for input in &files {
        if cancel.is_cancelled() {
            return Ok(Some(summary(0, true)));
        }
        let Ok((mask_path, colmap_path)) = output_paths(request, input) else {
            return Ok(None);
        };
        if !mask_path.is_file() || !colmap_path.is_file() {
            return Ok(None);
        }
        outputs.push((input, mask_path, colmap_path));
    }

    for (input, mask_path, colmap_path) in &outputs {
        if cancel.is_cancelled() {
            return Ok(Some(summary(0, true)));
        }
        let image = match ImageReader::open(input).and_then(|reader| reader.with_guessed_format()) {
            Ok(reader) => match reader.decode() {
                Ok(image) => image,
                Err(_) => return Ok(None),
            },
            Err(_) => return Ok(None),
        };
        let (width, height) = image.dimensions();
        if !is_valid_mask_file(&mask_path, width, height)
            || !is_valid_mask_file(&colmap_path, width, height)
        {
            return Ok(None);
        }
    }

    for (index, (input, mask_path, colmap_path)) in outputs.iter().enumerate() {
        if cancel.is_cancelled() {
            return Ok(Some(summary(index, true)));
        }
        emit_progress(
            on_progress,
            index,
            total,
            index + 1,
            input,
            mask_path,
            colmap_path,
            MaskStage::Skipped,
            (index + 1) as f32 / total as f32,
            "已確認遮罩存在，已略過",
        );
    }
    Ok(Some(summary(total, false)))
}

struct NoExclusionsEngine;

impl MaskEngine for NoExclusionsEngine {
    fn generate_exclusion_mask(
        &self,
        image: &DynamicImage,
        _classes: &[String],
        _confidence: f32,
        _mask_sky: bool,
        cancel: &CancelToken,
    ) -> MaskResult<SegmentationMask> {
        if cancel.is_cancelled() {
            return Err(MaskError::Cancelled);
        }
        let (width, height) = image.dimensions();
        SegmentationMask::new(width, height, vec![0; width as usize * height as usize])
    }
}

/// Process with an injected backend.  This is the preferred entry point for a
/// long-lived studio worker and makes filesystem/cancellation behavior testable
/// without downloading large ONNX models.
#[cfg_attr(not(test), allow(dead_code))]
pub fn process_mask_batch_with_engine<E>(
    request: &MaskRequest,
    cancel: &CancelToken,
    engine: &E,
    on_progress: impl Fn(MaskProgress) + Sync,
) -> MaskResult<MaskSummary>
where
    E: MaskEngine + ?Sized,
{
    validate_request(request)?;
    process_with_engine(request, cancel, engine, on_progress)
}

fn process_with_engine<E>(
    request: &MaskRequest,
    cancel: &CancelToken,
    engine: &E,
    on_progress: impl Fn(MaskProgress) + Sync,
) -> MaskResult<MaskSummary>
where
    E: MaskEngine + ?Sized,
{
    let files = collect_images(&request.images_dir)?;
    let total = files.len();
    if total == 0 {
        return Ok(MaskSummary {
            total,
            succeeded: 0,
            skipped: 0,
            failed: 0,
            cancelled: false,
            failures: Vec::new(),
        });
    }

    let completed = AtomicUsize::new(0);
    let progress_lock = Mutex::new(());
    let workers = total.min(mask_pipeline_worker_limit());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("mask-pipeline-{index}"))
        .build()
        .map_err(|error| MaskError::inference(format!("create mask pipeline: {error}")))?;
    let outcomes = pool.install(|| {
        files
            .par_iter()
            .enumerate()
            .map(|(index, input)| {
                process_one_image(
                    request,
                    cancel,
                    engine,
                    &on_progress,
                    &completed,
                    &progress_lock,
                    total,
                    index,
                    input,
                )
            })
            .collect::<Vec<_>>()
    });

    let mut summary = MaskSummary {
        total,
        succeeded: 0,
        skipped: 0,
        failed: 0,
        cancelled: false,
        failures: Vec::new(),
    };
    for (input, outcome) in files.iter().zip(outcomes) {
        match outcome {
            FileOutcome::Succeeded => summary.succeeded += 1,
            FileOutcome::Skipped => {
                summary.succeeded += 1;
                summary.skipped += 1;
            }
            FileOutcome::Failed(error) => {
                summary.failed += 1;
                summary.failures.push(MaskFailure {
                    input: input.clone(),
                    error,
                });
            }
            FileOutcome::Cancelled => summary.cancelled = true,
        }
    }

    Ok(summary)
}

enum FileOutcome {
    Succeeded,
    Skipped,
    Failed(String),
    Cancelled,
}

#[allow(clippy::too_many_arguments)]
fn process_one_image<E>(
    request: &MaskRequest,
    cancel: &CancelToken,
    engine: &E,
    on_progress: &(impl Fn(MaskProgress) + Sync),
    completed: &AtomicUsize,
    progress_lock: &Mutex<()>,
    total: usize,
    index: usize,
    input: &Path,
) -> FileOutcome
where
    E: MaskEngine + ?Sized,
{
    let (mask_path, colmap_path) = match output_paths(request, input) {
        Ok(paths) => paths,
        Err(error) => {
            let completed_hint = completed.fetch_add(1, Ordering::AcqRel) + 1;
            let _guard = progress_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let completed_count = completed
                .load(Ordering::Acquire)
                .max(completed_hint)
                .min(total);
            emit_progress(
                on_progress,
                index,
                total,
                completed_count,
                input,
                &request.masks_dir,
                &request.colmap_masks_dir,
                MaskStage::Failed,
                completed_count as f32 / total as f32,
                &error.to_string(),
            );
            return FileOutcome::Failed(error.to_string());
        }
    };
    let report = |stage: MaskStage, fraction: f32, completed_hint: usize, message: &str| {
        let _guard = progress_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let completed_count = completed
            .load(Ordering::Acquire)
            .max(completed_hint)
            .min(total);
        let completed_fraction = completed_count as f32 / total as f32;
        emit_progress(
            on_progress,
            index,
            total,
            completed_count,
            input,
            &mask_path,
            &colmap_path,
            stage,
            fraction.max(completed_fraction),
            message,
        );
    };
    if cancel.is_cancelled() {
        report(
            MaskStage::Cancelled,
            completed.load(Ordering::Acquire) as f32 / total as f32,
            completed.load(Ordering::Acquire),
            "遮罩處理已取消",
        );
        return FileOutcome::Cancelled;
    }

    // A partially resumed batch reaches this per-file path even when some
    // outputs are already complete. Read only the source header before deciding
    // to skip so completed high-resolution frames are not decoded in full.
    if request.skip_verified && mask_path.is_file() && colmap_path.is_file() {
        let source_dimensions = ImageReader::open(input)
            .and_then(|reader| reader.with_guessed_format())
            .ok()
            .and_then(|reader| reader.into_dimensions().ok());
        if let Some((width, height)) = source_dimensions {
            if is_valid_mask_file(&mask_path, width, height)
                && is_valid_mask_file(&colmap_path, width, height)
            {
                let completed_count = completed.fetch_add(1, Ordering::AcqRel) + 1;
                report(
                    MaskStage::Skipped,
                    completed_count as f32 / total as f32,
                    completed_count,
                    "已確認遮罩存在，已略過",
                );
                return FileOutcome::Skipped;
            }
        }
    }

    let image = match ImageReader::open(input)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(MaskError::from)
        .and_then(|reader| reader.decode().map_err(MaskError::from))
    {
        Ok(image) => image,
        Err(error) => {
            let message = error.to_string();
            let completed_count = completed.fetch_add(1, Ordering::AcqRel) + 1;
            report(
                MaskStage::Failed,
                completed_count as f32 / total as f32,
                completed_count,
                &message,
            );
            return FileOutcome::Failed(message);
        }
    };

    let (width, height) = image.dimensions();
    let image = resize_for_mask_working_resolution(image);
    let (mask_width, mask_height) = image.dimensions();

    report(
        MaskStage::Inference,
        completed.load(Ordering::Acquire) as f32 / total as f32,
        completed.load(Ordering::Acquire),
        "正在執行 YOLO11／SkySeg 推論",
    );
    let exclusions = match engine.generate_exclusion_mask(
        &image,
        &request.classes,
        request.confidence,
        request.mask_sky,
        cancel,
    ) {
        Ok(mask) => mask,
        Err(MaskError::Cancelled) => {
            report(
                MaskStage::Cancelled,
                completed.load(Ordering::Acquire) as f32 / total as f32,
                completed.load(Ordering::Acquire),
                "遮罩處理已取消",
            );
            return FileOutcome::Cancelled;
        }
        Err(error) => {
            let message = error.to_string();
            let completed_count = completed.fetch_add(1, Ordering::AcqRel) + 1;
            report(
                MaskStage::Failed,
                completed_count as f32 / total as f32,
                completed_count,
                &message,
            );
            return FileOutcome::Failed(message);
        }
    };

    if exclusions.width != mask_width || exclusions.height != mask_height {
        let message = format!(
            "backend returned {}x{} for mask working size {}x{}",
            exclusions.width, exclusions.height, mask_width, mask_height
        );
        let completed_count = completed.fetch_add(1, Ordering::AcqRel) + 1;
        report(
            MaskStage::Failed,
            completed_count as f32 / total as f32,
            completed_count,
            &message,
        );
        return FileOutcome::Failed(message);
    }

    if cancel.is_cancelled() {
        report(
            MaskStage::Cancelled,
            completed.load(Ordering::Acquire) as f32 / total as f32,
            completed.load(Ordering::Acquire),
            "已在寫入遮罩前取消",
        );
        return FileOutcome::Cancelled;
    }

    let optical_occlusion = optical_occlusion_for_input(request, input);
    let keep = build_keep_mask(
        mask_width,
        mask_height,
        request.valid_radius_ratio,
        optical_occlusion,
        &exclusions.data,
    );
    let keep = match resize_binary_mask(keep, mask_width, mask_height, width, height) {
        Ok(mask) => mask,
        Err(error) => {
            let message = error.to_string();
            let completed_count = completed.fetch_add(1, Ordering::AcqRel) + 1;
            report(
                MaskStage::Failed,
                completed_count as f32 / total as f32,
                completed_count,
                &message,
            );
            return FileOutcome::Failed(message);
        }
    };
    report(
        MaskStage::Writing,
        completed.load(Ordering::Acquire) as f32 / total as f32,
        completed.load(Ordering::Acquire),
        "正在寫入遮罩檔案",
    );
    if let Err(error) = write_mask_atomic(&mask_path, width, height, &keep, false)
        .and_then(|_| write_mask_atomic(&colmap_path, width, height, &keep, true))
    {
        let message = error.to_string();
        let completed_count = completed.fetch_add(1, Ordering::AcqRel) + 1;
        report(
            MaskStage::Failed,
            completed_count as f32 / total as f32,
            completed_count,
            &message,
        );
        return FileOutcome::Failed(message);
    }

    let completed_count = completed.fetch_add(1, Ordering::AcqRel) + 1;
    report(
        MaskStage::Completed,
        completed_count as f32 / total as f32,
        completed_count,
        "遮罩處理完成",
    );
    FileOutcome::Succeeded
}

fn validate_request(request: &MaskRequest) -> MaskResult<()> {
    if !request.images_dir.exists() {
        return Err(MaskError::invalid_input(format!(
            "images path does not exist: {}",
            request.images_dir.display()
        )));
    }
    if request.masks_dir.as_os_str().is_empty() || request.colmap_masks_dir.as_os_str().is_empty() {
        return Err(MaskError::invalid_input(
            "mask output directories are required",
        ));
    }
    if !request.confidence.is_finite() || !(0.0..=1.0).contains(&request.confidence) {
        return Err(MaskError::invalid_input(
            "confidence must be finite and in [0, 1]",
        ));
    }
    if !request.valid_radius_ratio.is_finite() || !(0.0..=1.0).contains(&request.valid_radius_ratio)
    {
        return Err(MaskError::invalid_input(
            "valid_radius_ratio must be finite and in [0, 1]",
        ));
    }
    Ok(())
}

fn collect_images(root: &Path) -> MaskResult<Vec<PathBuf>> {
    if root.is_file() {
        return if is_image_path(root) {
            Ok(vec![root.to_path_buf()])
        } else {
            Err(MaskError::invalid_input(format!(
                "input is not a supported image: {}",
                root.display()
            )))
        };
    }
    if !root.is_dir() {
        return Err(MaskError::invalid_input(format!(
            "images path is not a file or directory: {}",
            root.display()
        )));
    }
    let mut files = Vec::new();
    collect_images_recursive(root, &mut files)?;
    files.sort_by_key(|path| {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_ascii_lowercase()
    });
    Ok(files)
}

fn collect_images_recursive(root: &Path, files: &mut Vec<PathBuf>) -> MaskResult<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_images_recursive(&path, files)?;
        } else if file_type.is_file() && is_image_path(&path) {
            files.push(path);
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
        Some("jpg")
            | Some("jpeg")
            | Some("png")
            | Some("webp")
            | Some("bmp")
            | Some("tif")
            | Some("tiff")
    )
}

fn output_paths(request: &MaskRequest, input: &Path) -> MaskResult<(PathBuf, PathBuf)> {
    let relative = if request.images_dir.is_file() {
        PathBuf::from(
            input
                .file_name()
                .ok_or_else(|| MaskError::invalid_input("input image has no filename"))?,
        )
    } else {
        input
            .strip_prefix(&request.images_dir)
            .map_err(|_| {
                MaskError::invalid_input(format!(
                    "image is outside input root: {}",
                    input.display()
                ))
            })?
            .to_path_buf()
    };
    let mask_path = request.masks_dir.join(&relative);
    let mut colmap_path = request.colmap_masks_dir.join(&relative);
    let file_name = relative
        .file_name()
        .ok_or_else(|| MaskError::invalid_input("input image has no filename"))?;
    colmap_path.set_file_name(format!("{}.png", file_name.to_string_lossy()));
    Ok((mask_path, colmap_path))
}

fn optical_occlusion_for_input<'a>(
    request: &'a MaskRequest,
    input: &Path,
) -> Option<&'a OpticalOcclusion> {
    let relative = input.strip_prefix(&request.images_dir).ok()?;
    let lens = relative.parent()?.file_name()?.to_str()?;
    let file_name = relative.file_name()?.to_str()?;
    let calibrations = request
        .optical_occlusions
        .iter()
        .find_map(|(prefix, calibrations)| file_name.starts_with(prefix).then_some(calibrations))?;
    match lens {
        "lens0" => Some(&calibrations.lens0),
        "lens1" => Some(&calibrations.lens1),
        _ => None,
    }
}

fn build_keep_mask(
    width: u32,
    height: u32,
    radius_ratio: f32,
    optical_occlusion: Option<&OpticalOcclusion>,
    exclusions: &[u8],
) -> Vec<u8> {
    let mut keep = vec![0u8; (width as usize).saturating_mul(height as usize)];
    let valid_region = ValidRegion::new(width, height, f64::from(radius_ratio), optical_occlusion);
    for y in 0..height {
        let row_offset_squared = valid_region.row_offset_squared(y);
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            if valid_region.contains_x(x, y, row_offset_squared) && exclusions[index] == 0 {
                keep[index] = 255;
            }
        }
    }
    keep
}

fn resize_for_mask_working_resolution(image: DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    let (working_width, working_height) = mask_working_dimensions(width, height);
    if (working_width, working_height) == (width, height) {
        image
    } else {
        image.resize_exact(working_width, working_height, FilterType::Triangle)
    }
}

fn mask_working_dimensions(width: u32, height: u32) -> (u32, u32) {
    let longest_edge = width.max(height);
    if longest_edge <= MASK_WORKING_LONG_EDGE || longest_edge == 0 {
        return (width, height);
    }
    let scale = f64::from(MASK_WORKING_LONG_EDGE) / f64::from(longest_edge);
    (
        (f64::from(width) * scale).round().max(1.0) as u32,
        (f64::from(height) * scale).round().max(1.0) as u32,
    )
}

fn resize_binary_mask(
    data: Vec<u8>,
    width: u32,
    height: u32,
    output_width: u32,
    output_height: u32,
) -> MaskResult<Vec<u8>> {
    if (width, height) == (output_width, output_height) {
        return Ok(data);
    }
    let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(width, height, data)
        .ok_or_else(|| MaskError::image("failed to build working-resolution mask"))?;
    Ok(
        image::imageops::resize(&image, output_width, output_height, FilterType::Nearest)
            .into_raw(),
    )
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_mask_atomic(
    path: &Path,
    width: u32,
    height: u32,
    data: &[u8],
    force_png: bool,
) -> MaskResult<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| MaskError::invalid_input("mask dimensions overflow"))?;
    if data.len() != expected {
        return Err(MaskError::invalid_input(
            "mask data does not match dimensions",
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| MaskError::invalid_input("mask output has no filename"))?;
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = path.with_file_name(format!(
        ".{file_name}.{}.{}.part",
        std::process::id(),
        counter
    ));
    let format = if force_png {
        ImageFormat::Png
    } else {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref()
        {
            Some("jpg") | Some("jpeg") => ImageFormat::Jpeg,
            Some("webp") => ImageFormat::WebP,
            Some("bmp") => ImageFormat::Bmp,
            Some("tif") | Some("tiff") => ImageFormat::Tiff,
            _ => ImageFormat::Png,
        }
    };
    // Encode directly from the shared mask slice. Constructing an owned
    // ImageBuffer here used to clone the full-resolution mask once for each of
    // the two output files, adding two avoidable allocations and memory copies
    // per source image.
    let write_result =
        image::save_buffer_with_format(&temporary, data, width, height, ColorType::L8, format);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(MaskError::from(error));
    }
    // Keep rename on the same directory/filesystem.  A complete temporary file
    // is never visible at the final path, and a stale `.part` can be removed on
    // the next run without invalidating an existing mask.
    // Windows FlushFileBuffers requires a handle opened with GENERIC_WRITE.
    // File::open is read-only and returns os error 5 when sync_all reaches it.
    let sync_result = fs::OpenOptions::new()
        .write(true)
        .open(&temporary)
        .and_then(|file| file.sync_all());
    if let Err(error) = sync_result {
        let _ = fs::remove_file(&temporary);
        return Err(MaskError::from(error));
    }
    rename_replace(&temporary, path)?;
    Ok(())
}

/// `std::fs::rename` atomically replaces files on Unix and on modern Windows,
/// but older Windows filesystems reject a destination that already exists.  The
/// fallback moves the old destination aside only after the first atomic attempt
/// fails, then restores it if the replacement cannot be committed.
fn rename_replace(temporary: &Path, destination: &Path) -> MaskResult<()> {
    match fs::rename(temporary, destination) {
        Ok(()) => Ok(()),
        Err(_first_error) if destination.is_file() => {
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let file_name = destination
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| MaskError::invalid_input("mask output has no filename"))?;
            let backup = destination.with_file_name(format!(
                ".{file_name}.{}.{}.old",
                std::process::id(),
                counter
            ));
            fs::rename(destination, &backup).map_err(MaskError::from)?;
            match fs::rename(temporary, destination) {
                Ok(()) => {
                    let _ = fs::remove_file(backup);
                    Ok(())
                }
                Err(second_error) => {
                    let _ = fs::rename(&backup, destination);
                    let _ = fs::remove_file(temporary);
                    Err(MaskError::from(second_error))
                }
            }
        }
        Err(error) => {
            let _ = fs::remove_file(temporary);
            Err(MaskError::from(error))
        }
    }
}

fn is_valid_mask_file(path: &Path, width: u32, height: u32) -> bool {
    let Ok(reader) = ImageReader::open(path) else {
        return false;
    };
    let Ok(reader) = reader.with_guessed_format() else {
        return false;
    };
    let Ok(image) = reader.decode() else {
        return false;
    };
    image.dimensions() == (width, height)
}

fn emit_progress(
    callback: &impl Fn(MaskProgress),
    index: usize,
    total: usize,
    completed: usize,
    input: &Path,
    mask_path: &Path,
    colmap_path: &Path,
    stage: MaskStage,
    fraction: f32,
    message: &str,
) {
    callback(MaskProgress {
        index: index + 1,
        total,
        completed: completed.min(total),
        input: input.to_path_buf(),
        mask_path: mask_path.to_path_buf(),
        colmap_mask_path: colmap_path.to_path_buf(),
        stage,
        fraction: fraction.clamp(0.0, 1.0),
        message: message.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use tempfile::TempDir;

    struct FakeEngine {
        calls: AtomicUsize,
        exclusion: u8,
    }

    struct ConcurrentEngine {
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
    }

    struct RecordingDimensionsEngine {
        dimensions: Mutex<Option<(u32, u32)>>,
    }

    impl MaskEngine for ConcurrentEngine {
        fn generate_exclusion_mask(
            &self,
            image: &DynamicImage,
            _classes: &[String],
            _confidence: f32,
            _mask_sky: bool,
            _cancel: &CancelToken,
        ) -> MaskResult<SegmentationMask> {
            let in_flight = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(in_flight, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(30));
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            let (width, height) = image.dimensions();
            SegmentationMask::new(width, height, vec![0; width as usize * height as usize])
        }
    }

    impl MaskEngine for FakeEngine {
        fn generate_exclusion_mask(
            &self,
            image: &DynamicImage,
            _classes: &[String],
            _confidence: f32,
            _mask_sky: bool,
            _cancel: &CancelToken,
        ) -> MaskResult<SegmentationMask> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let (width, height) = image.dimensions();
            SegmentationMask::new(
                width,
                height,
                vec![self.exclusion; (width as usize) * (height as usize)],
            )
        }
    }

    impl MaskEngine for RecordingDimensionsEngine {
        fn generate_exclusion_mask(
            &self,
            image: &DynamicImage,
            _classes: &[String],
            _confidence: f32,
            _mask_sky: bool,
            _cancel: &CancelToken,
        ) -> MaskResult<SegmentationMask> {
            let dimensions = image.dimensions();
            *self.dimensions.lock().unwrap() = Some(dimensions);
            SegmentationMask::new(
                dimensions.0,
                dimensions.1,
                vec![0; dimensions.0 as usize * dimensions.1 as usize],
            )
        }
    }

    fn request(dir: &TempDir) -> MaskRequest {
        MaskRequest {
            images_dir: dir.path().join("images"),
            masks_dir: dir.path().join("masks"),
            colmap_masks_dir: dir.path().join("masks_colmap"),
            ..MaskRequest::default()
        }
    }

    #[test]
    fn writes_same_name_and_colmap_suffix_with_circle_black() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let mut request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        let source = request.images_dir.join("lens0/frame.png");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(5, 5, Rgb([100, 100, 100])).save(&source)?;
        request.valid_radius_ratio = 0.4;
        let engine = FakeEngine {
            calls: AtomicUsize::new(0),
            exclusion: 0,
        };
        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(summary.succeeded, 1);
        let mask = ImageReader::open(request.masks_dir.join("lens0/frame.png"))?
            .decode()?
            .to_luma8();
        let colmap = request.colmap_masks_dir.join("lens0/frame.png.png");
        assert!(colmap.is_file());
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
        assert_eq!(mask.get_pixel(2, 2)[0], 255);
        assert!(!request.masks_dir.join("lens0/.frame.png.png.part").exists());
        Ok(())
    }

    #[test]
    fn keeps_scene_pixels_above_the_dji_curve_and_excludes_only_pixels_below_it() {
        let exclusions = vec![0; 100 * 100];
        let optical_occlusion = OpticalOcclusion::from_source_pixels(
            100.0,
            100.0,
            50.0,
            50.0,
            &[20.0, 50.0, 80.0],
            &[70.0, 90.0, 70.0],
        )
        .unwrap();
        let keep = build_keep_mask(
            100,
            100,
            DJI_VALID_RADIUS_RATIO as f32,
            Some(&optical_occlusion),
            &exclusions,
        );
        let pixel = |x: usize, y: usize| keep[y * 100 + x];

        assert_eq!(pixel(49, 1), 255);
        assert_eq!(pixel(99, 49), 255);
        assert_eq!(pixel(49, 88), 255);
        assert_eq!(pixel(49, 90), 0);
    }

    #[test]
    fn runs_masks_at_bounded_resolution_and_writes_source_dimensions() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let request = request(&dir);
        fs::create_dir_all(&request.images_dir)?;
        ImageBuffer::<Rgb<u8>, _>::from_pixel(2048, 1024, Rgb([100, 100, 100]))
            .save(request.images_dir.join("frame.png"))?;
        let engine = RecordingDimensionsEngine {
            dimensions: Mutex::new(None),
        };

        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        let output = ImageReader::open(request.masks_dir.join("frame.png"))?
            .decode()?
            .to_luma8();

        assert_eq!(summary.succeeded, 1);
        assert_eq!(*engine.dimensions.lock().unwrap(), Some((640, 320)));
        assert_eq!(output.dimensions(), (2048, 1024));
        Ok(())
    }

    #[test]
    fn nearest_neighbor_resize_preserves_binary_mask_values() -> MaskResult<()> {
        let resized = resize_binary_mask(vec![0, 255], 2, 1, 4, 2)?;

        assert_eq!(resized, vec![0, 0, 255, 255, 0, 0, 255, 255]);
        Ok(())
    }

    #[test]
    fn selects_dji_calibration_by_source_prefix_and_lens_folder() {
        let dir = TempDir::new().unwrap();
        let mut request = request(&dir);
        let occlusion = OpticalOcclusion::from_source_pixels(
            100.0,
            100.0,
            50.0,
            50.0,
            &[20.0, 50.0, 80.0],
            &[70.0, 90.0, 70.0],
        )
        .unwrap();
        request.optical_occlusions.insert(
            "source000_".to_owned(),
            LensOpticalOcclusions {
                lens0: occlusion.clone(),
                lens1: occlusion,
            },
        );

        assert!(optical_occlusion_for_input(
            &request,
            &request.images_dir.join("lens0/source000_000001.jpg")
        )
        .is_some());
        assert!(optical_occlusion_for_input(
            &request,
            &request.images_dir.join("lens1/source000_000001.jpg")
        )
        .is_some());
        assert!(optical_occlusion_for_input(
            &request,
            &request.images_dir.join("lens0/source001_000001.jpg")
        )
        .is_none());
    }

    #[test]
    fn skips_only_when_both_masks_are_valid() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        let source = request.images_dir.join("lens0/frame.png");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(2, 2, Rgb([100, 100, 100])).save(&source)?;
        let engine = FakeEngine {
            calls: AtomicUsize::new(0),
            exclusion: 0,
        };
        process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        let first_calls = engine.calls.load(Ordering::Relaxed);
        let mask_path = request.masks_dir.join("lens0/frame.png");
        let colmap_mask_path = request.colmap_masks_dir.join("lens0/frame.png.png");
        let mask_before = fs::read(&mask_path)?;
        let colmap_mask_before = fs::read(&colmap_mask_path)?;
        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(summary.skipped, 1);
        assert_eq!(engine.calls.load(Ordering::Relaxed), first_calls);
        assert_eq!(fs::read(&mask_path)?, mask_before);
        assert_eq!(fs::read(&colmap_mask_path)?, colmap_mask_before);
        fs::write(&colmap_mask_path, b"broken")?;
        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(summary.succeeded, 1);
        assert_eq!(summary.skipped, 0);
        assert_eq!(engine.calls.load(Ordering::Relaxed), first_calls + 1);
        Ok(())
    }

    #[test]
    fn fully_verified_batch_skips_before_loading_models() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let mut request = request(&dir);
        request.classes = vec!["person".to_string()];
        request.yolo_model = Some(dir.path().join("missing-yolo.onnx"));
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        fs::create_dir_all(request.masks_dir.join("lens0"))?;
        fs::create_dir_all(request.colmap_masks_dir.join("lens0"))?;

        ImageBuffer::<Rgb<u8>, _>::from_pixel(2, 2, Rgb([100, 100, 100]))
            .save(request.images_dir.join("lens0/frame.png"))?;
        ImageBuffer::<Luma<u8>, _>::from_pixel(2, 2, Luma([255]))
            .save(request.masks_dir.join("lens0/frame.png"))?;
        ImageBuffer::<Luma<u8>, _>::from_pixel(2, 2, Luma([255]))
            .save(request.colmap_masks_dir.join("lens0/frame.png.png"))?;

        // The explicit model path is missing. Reaching model discovery would
        // fail, so success proves the verified outputs are handled first.
        let summary = process_mask_batch(&request, &CancelToken::new(), |_| {})?;
        assert_eq!(summary.total, 1);
        assert_eq!(summary.succeeded, 1);
        assert_eq!(summary.skipped, 1);
        Ok(())
    }

    #[test]
    fn overlaps_bounded_mask_work_and_keeps_progress_monotonic() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let request = request(&dir);
        fs::create_dir_all(&request.images_dir)?;
        for index in 0..6 {
            ImageBuffer::<Rgb<u8>, _>::from_pixel(8, 8, Rgb([100, 100, 100]))
                .save(request.images_dir.join(format!("frame-{index}.png")))?;
        }
        let engine = ConcurrentEngine {
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        };
        let fractions = Mutex::new(Vec::new());
        let completed_counts = Mutex::new(Vec::new());
        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |progress| {
                fractions.lock().unwrap().push(progress.fraction);
                completed_counts
                    .lock()
                    .unwrap()
                    .push((progress.completed, progress.total));
            })?;

        assert_eq!(summary.succeeded, 6);
        assert_eq!(summary.failed, 0);
        assert!(engine.max_in_flight.load(Ordering::SeqCst) >= 2);
        assert!(engine.max_in_flight.load(Ordering::SeqCst) <= MASK_PIPELINE_WORKERS);
        let fractions = fractions.into_inner().unwrap();
        assert!(fractions.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(fractions.last().copied(), Some(1.0));
        let completed_counts = completed_counts.into_inner().unwrap();
        assert!(completed_counts
            .windows(2)
            .all(|pair| pair[0].0 <= pair[1].0));
        assert!(completed_counts
            .iter()
            .all(|(completed, total)| *completed <= *total && *total == 6));
        assert_eq!(completed_counts.last().copied(), Some((6, 6)));
        Ok(())
    }

    #[test]
    fn cancellation_does_not_commit_partial_output() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let request = request(&dir);
        fs::create_dir_all(&request.images_dir)?;
        ImageBuffer::<Rgb<u8>, _>::from_pixel(2, 2, Rgb([100, 100, 100]))
            .save(request.images_dir.join("frame.png"))?;
        let token = CancelToken::new();
        token.cancel();
        let engine = FakeEngine {
            calls: AtomicUsize::new(0),
            exclusion: 0,
        };
        let summary = process_mask_batch_with_engine(&request, &token, &engine, |_| {})?;
        assert!(summary.cancelled);
        assert!(!request.masks_dir.exists());
        Ok(())
    }

    #[test]
    #[ignore = "downloads about 216 MB and requires a supported physical GPU"]
    fn downloads_and_loads_production_models() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let mut request = request(&dir);
        request.classes = vec!["person".to_string()];
        request.mask_sky = true;
        request.model_cache_dir = Some(dir.path().join("models"));

        let engine = NativeMaskEngine::load(&request, &CancelToken::new(), &|_| {})?;
        assert!(engine.skyseg.is_some());
        Ok(())
    }

    #[test]
    #[ignore = "requires production model paths, a test image, and a physical GPU"]
    fn processes_production_models_through_the_concurrent_pipeline() -> MaskResult<()> {
        let yolo = std::env::var_os("GS360_TEST_YOLO_MODEL")
            .map(PathBuf::from)
            .expect("GS360_TEST_YOLO_MODEL is required");
        let skyseg = std::env::var_os("GS360_TEST_SKYSEG_MODEL")
            .map(PathBuf::from)
            .expect("GS360_TEST_SKYSEG_MODEL is required");
        let image = std::env::var_os("GS360_TEST_PERSON_IMAGE")
            .map(PathBuf::from)
            .expect("GS360_TEST_PERSON_IMAGE is required");
        let provider =
            std::env::var("GS360_TEST_GPU_PROVIDER").unwrap_or_else(|_| "CoreML".to_string());
        let frame_count = std::env::var("GS360_TEST_FRAME_COUNT")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|value| *value > 0)
            .unwrap_or(6);
        let dir = TempDir::new()?;
        let mut request = request(&dir);
        fs::create_dir_all(&request.images_dir)?;
        for index in 0..frame_count {
            fs::copy(
                &image,
                request.images_dir.join(format!("frame-{index}.jpg")),
            )?;
        }
        request.classes = vec!["person".to_string()];
        request.mask_sky = true;
        request.skip_verified = false;
        request.yolo_model = Some(yolo);
        request.skyseg_model = Some(skyseg);
        request.execution_provider = Some(provider);
        let cancel = CancelToken::new();
        let engine = NativeMaskEngine::load(&request, &cancel, &|_| {})?;

        let summary = process_mask_batch_with_engine(&request, &cancel, &engine, |_| {})?;

        assert_eq!(summary.succeeded, frame_count);
        assert_eq!(summary.failed, 0);
        assert!(!summary.cancelled);
        if frame_count > 1 {
            let first = ImageReader::open(request.masks_dir.join("frame-0.jpg"))?
                .decode()?
                .to_luma8();
            let second = ImageReader::open(request.masks_dir.join("frame-1.jpg"))?
                .decode()?
                .to_luma8();
            assert_eq!(first, second);
        }
        Ok(())
    }
}
