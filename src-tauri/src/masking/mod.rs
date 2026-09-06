//! Native-fisheye instance masks for the studio backend.
//!
//! The module treats every image below `images_dir` (including `lens0/`,
//! `lens1/`, and future `lens*` directories) as an independent camera frame.
//! `masks/` mirrors the source hierarchy, replaces the image extension with
//! `.png`, and uses white for pixels retained by SfM/training and black for
//! excluded pixels (for example `images/lens0/frame.jpg` maps to
//! `masks/lens0/frame.png`). Pixels outside the full fisheye circle and, when
//! present, below DJI's calibrated fixed optical-occlusion curve are black. Scene content such
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
mod tiles;
mod provenance;

pub use inference::YoloSegPipeline;
pub(crate) use models::resolve_aliked_models;
pub use models::{ModelDownloadProgress, ModelPaths};
pub use skyseg::SkysegPipeline;
pub use tiles::{read_calibration, FisheyeCamera};
pub use provenance::review_summary;

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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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
    pub classes: Vec<String>,
    pub mask_sky: bool,
    pub confidence: f32,
    /// One upright inference or a union of four quarter-turn YOLO inferences.
    pub rotations: u32,
    /// Semantic dilation in inference pixels (native proxy or perspective tile).
    pub dilation: u32,
    /// Calibrated perspective mode, keyed by lens/source. Empty uses native inference.
    pub calibrated_cameras: BTreeMap<String, FisheyeCamera>,
    /// Optional reviewed, binary keep masks. Black pixels are always preserved.
    pub additional_masks_dir: Option<PathBuf>,
    /// Radius as a ratio of the shorter source dimension.
    pub valid_radius_ratio: f32,
    /// DJI optical calibration keyed by extraction filename prefix.
    pub optical_occlusions: BTreeMap<String, LensOpticalOcclusions>,
    /// Skip only when the output decodes and matches the source dimensions.
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
            classes: Vec::new(),
            mask_sky: false,
            confidence: YOLO_CONFIDENCE_THRESHOLD,
            rotations: 1,
            dilation: 0,
            calibrated_cameras: BTreeMap::new(),
            additional_masks_dir: None,
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
    pub stage: MaskStage,
    pub fraction: f32,
    pub message: String,
}

/// Stable progress stage names for frontend events.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
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
    // Resolve the actual weights before deciding whether a previous result can
    // be reused. This does not create an ONNX session. A model/configuration
    // change must never be hidden by a dimension-only mask cache hit.
    let mut resolved = request.clone();
    if !request.classes.is_empty() || request.mask_sky {
        let paths = ModelPaths::resolve(request.model_dir.as_deref(), request.model_cache_dir.as_deref(),
            request.yolo_model.as_deref(), request.skyseg_model.as_deref(), !request.classes.is_empty(),
            request.mask_sky, cancel, &|event| on_progress(MaskProgress {
                index: 0, total: 0, completed: 0, input: request.images_dir.clone(),
                mask_path: request.masks_dir.clone(), stage: MaskStage::LoadingModel,
                fraction: 0.01, message: format!("正在確認 {} 模型（{} / {} bytes）", event.label, event.downloaded, event.total),
            }))?;
        resolved.yolo_model = paths.yolo;
        resolved.skyseg_model = paths.skyseg;
    }
    let request = &resolved;
    if request.skip_verified {
        if let Some(summary) = skip_fully_verified_batch(request, cancel, &on_progress)? {
            return Ok(summary);
        }
    }
    if request.classes.is_empty() && !request.mask_sky {
        return process_with_engine(request, cancel, &NoExclusionsEngine, |mut event| {
            event.fraction = 0.05 + 0.95 * event.fraction;
            on_progress(event);
        });
    }
    on_progress(MaskProgress {
        index: 0,
        total: 0,
        completed: 0,
        input: request.images_dir.clone(),
        mask_path: request.masks_dir.clone(),
        stage: MaskStage::Discovering,
        fraction: 0.05,
        message: "正在掃描原生雙魚眼影像".to_string(),
    });
    on_progress(MaskProgress {
        index: 0,
        total: 0,
        completed: 0,
        input: request.images_dir.clone(),
        mask_path: request.masks_dir.clone(),
        stage: MaskStage::LoadingModel,
        fraction: 0.05,
        message: "正在載入 YOLO11／SkySeg 模型".to_string(),
    });
    let loading_basis_points = AtomicU32::new(500);
    let loading_done = AtomicBool::new(false);
    let engine = thread::scope(|scope| {
        let heartbeat = scope.spawn(|| {
            let started = Instant::now();
            while !loading_done.load(Ordering::Acquire) {
                thread::park_timeout(Duration::from_secs(1));
                if loading_done.load(Ordering::Acquire) {
                    break;
                }
                on_progress(MaskProgress {
                    index: 0,
                    total: 0,
                    completed: 0,
                    input: request.images_dir.clone(),
                    mask_path: request.masks_dir.clone(),
                    stage: MaskStage::LoadingModel,
                    fraction: loading_basis_points.load(Ordering::Acquire) as f32 / 10_000.0,
                    message: format!(
                        "正在載入 YOLO11／SkySeg 模型（已等待 {} 秒）",
                        started.elapsed().as_secs(),
                    ),
                });
            }
        });
        let result = NativeMaskEngine::load(request, cancel, &|event| {
            let percent = event
                .downloaded
                .saturating_mul(100)
                .checked_div(event.total)
                .unwrap_or(0);
            let basis_points = 500_u32.saturating_add((percent as u32).saturating_mul(5));
            loading_basis_points.fetch_max(basis_points.min(1_000), Ordering::AcqRel);
            on_progress(MaskProgress {
                index: 0,
                total: 0,
                completed: 0,
                input: request.images_dir.clone(),
                mask_path: request.masks_dir.clone(),
                stage: MaskStage::LoadingModel,
                fraction: loading_basis_points.load(Ordering::Acquire) as f32 / 10_000.0,
                message: format!("首次使用，正在下載 {} 模型（{}%）", event.label, percent),
            });
        });
        loading_done.store(true, Ordering::Release);
        heartbeat.thread().unpark();
        let _ = heartbeat.join();
        result
    })?;
    on_progress(MaskProgress {
        index: 0,
        total: 0,
        completed: 0,
        input: request.images_dir.clone(),
        mask_path: request.masks_dir.clone(),
        stage: MaskStage::LoadingModel,
        fraction: 0.10,
        message: format!(
            "模型已載入 {}，CPU 推論回退已停用",
            engine.execution_provider
        ),
    });
    process_with_engine(request, cancel, &engine, |mut event| {
        event.fraction = 0.10 + 0.90 * event.fraction;
        on_progress(event);
    })
}

/// Return a completed summary before model discovery when every output is
/// already usable. No progress is emitted until the full batch is verified, so
/// a partial batch still follows the normal inference path.
fn skip_fully_verified_batch(
    request: &MaskRequest,
    cancel: &CancelToken,
    on_progress: &impl Fn(MaskProgress),
) -> MaskResult<Option<MaskSummary>> {
    let signature = provenance::request_signature(request)?;
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
        let Ok(mask_path) = output_path(request, input) else {
            return Ok(None);
        };
        if !mask_path.is_file() {
            return Ok(None);
        }
        outputs.push((input, mask_path));
    }

    for (index, (input, mask_path)) in outputs.iter().enumerate() {
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
        if !is_valid_mask_file(mask_path, width, height)
            || !provenance::can_reuse(request, input, mask_path, &signature)
        {
            return Ok(None);
        }
        on_progress(MaskProgress {
            index: index + 1,
            total,
            completed: index + 1,
            input: (*input).clone(),
            mask_path: mask_path.clone(),
            stage: MaskStage::Discovering,
            fraction: 0.05 * (index + 1) as f32 / total.max(1) as f32,
            message: format!("正在驗證既有遮罩（{} / {}）", index + 1, total),
        });
    }

    for (index, (input, mask_path)) in outputs.iter().enumerate() {
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
            MaskStage::Skipped,
            0.05 + 0.95 * (index + 1) as f32 / total as f32,
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
    let signature = provenance::request_signature(request)?;
    let total = files.len();
    let projection_cache = tiles::ProjectionCache::default();
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
                    &signature,
                    &projection_cache,
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
    signature: &str,
    projection_cache: &tiles::ProjectionCache,
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
    let mask_path = match output_path(request, input) {
        Ok(path) => path,
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
    if request.skip_verified && mask_path.is_file() {
        let source_dimensions = ImageReader::open(input)
            .and_then(|reader| reader.with_guessed_format())
            .ok()
            .and_then(|reader| reader.into_dimensions().ok());
        if let Some((width, height)) = source_dimensions {
            if is_valid_mask_file(&mask_path, width, height)
                && provenance::can_reuse(request, input, &mask_path, signature)
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

    let starting_identity = match provenance::identity(request, input, signature) {
        Ok(identity) => identity,
        Err(error) => {
            let count = completed.fetch_add(1, Ordering::AcqRel) + 1;
            report(MaskStage::Failed, count as f32 / total as f32, count, &error.to_string());
            return FileOutcome::Failed(error.to_string());
        }
    };
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
    let relative = input.strip_prefix(&request.images_dir).unwrap_or(input).to_string_lossy().replace('\\', "/");
    let camera = tiles::camera_group(&relative).and_then(|key| request.calibrated_cameras.get(&key));
    if !request.calibrated_cameras.is_empty() && camera.is_none() {
        let message = format!("No calibrated camera for {relative}; calibration must match this source and lens");
        let count = completed.fetch_add(1, Ordering::AcqRel) + 1;
        report(MaskStage::Failed, count as f32 / total as f32, count, &message);
        return FileOutcome::Failed(message);
    }
    let image = if camera.is_some() { image } else { resize_for_mask_working_resolution(image) };
    let (mask_width, mask_height) = image.dimensions();

    report(
        MaskStage::Inference,
        completed.load(Ordering::Acquire) as f32 / total as f32,
        completed.load(Ordering::Acquire),
        "正在執行 YOLO11／SkySeg 推論",
    );
    let exclusions = match camera.map_or_else(
        || infer_rotations(engine, &image, request, request.mask_sky, cancel).map(|mut mask| {
            dilate_exclusions(&mut mask, request.dilation); mask
        }),
        |camera| tiles::infer(&image, camera, request, engine, cancel, projection_cache),
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
    // Only semantic masks are resized. Compose the calibrated optical boundary
    // at native resolution so an empty semantic result preserves it exactly.
    let exclusions = match resize_binary_mask(exclusions.data, mask_width, mask_height, width, height) {
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
    let mut keep = build_keep_mask(width, height, request.valid_radius_ratio, optical_occlusion, &exclusions);
    if let Err(error) = provenance::merge_additional_mask(request, input, width, height, &mut keep) {
        let count = completed.fetch_add(1, Ordering::AcqRel) + 1;
        report(MaskStage::Failed, count as f32 / total as f32, count, &error.to_string());
        return FileOutcome::Failed(error.to_string());
    }
    report(
        MaskStage::Writing,
        completed.load(Ordering::Acquire) as f32 / total as f32,
        completed.load(Ordering::Acquire),
        "正在寫入遮罩檔案",
    );
    if let Err(error) = write_mask_atomic(&mask_path, width, height, &keep) {
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
    if let Err(error) = provenance::commit(request, input, &mask_path, signature, &starting_identity, &keep) {
        let count = completed.fetch_add(1, Ordering::AcqRel) + 1;
        report(MaskStage::Failed, count as f32 / total as f32, count, &error.to_string());
        return FileOutcome::Failed(error.to_string());
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
    if request.masks_dir.as_os_str().is_empty() {
        return Err(MaskError::invalid_input(
            "mask output directory is required",
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
    if !matches!(request.rotations, 1 | 4) || request.dilation > 16 {
        return Err(MaskError::invalid_input("rotations must be 1 or 4; dilation must be 0..16 inference pixels"));
    }
    for camera in request.calibrated_cameras.values() { camera.validate()?; }
    if !request.calibrated_cameras.is_empty() {
        for input in collect_images(&request.images_dir)? {
            let relative = input.strip_prefix(&request.images_dir).unwrap_or(&input).to_string_lossy().replace('\\', "/");
            let camera = tiles::camera_group(&relative).and_then(|key| request.calibrated_cameras.get(&key))
                .ok_or_else(|| MaskError::invalid_input(format!("no calibrated camera bound to {relative}")))?;
            if image::image_dimensions(&input)? != (camera.width, camera.height) {
                return Err(MaskError::invalid_input(format!("native dimensions differ from calibration: {relative}")));
            }
        }
    }
    if let Some(additional) = &request.additional_masks_dir {
        if !additional.is_dir() || additional.canonicalize().ok() == request.masks_dir.canonicalize().ok() {
            return Err(MaskError::invalid_input("additional masks must be an existing directory separate from the output"));
        }
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
    let mut output_sources = BTreeMap::new();
    for path in &files {
        let mut relative = path.strip_prefix(root).unwrap_or(path).to_path_buf();
        relative.set_extension("png");
        if let Some(previous) = output_sources.insert(relative.clone(), path) {
            return Err(MaskError::invalid_input(format!(
                "images {} and {} map to the same mask {}",
                previous.display(),
                path.display(),
                relative.display()
            )));
        }
    }
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

/// Terminal integrity check. Decoding is necessary: a valid PNG header can
/// conceal a truncated payload or a non-binary keep mask.
pub fn verify_mask_output_coverage(request: &MaskRequest) -> MaskResult<usize> {
    validate_request(request)?;
    let files = collect_images(&request.images_dir)?;
    for input in &files {
        let source_dimensions = ImageReader::open(input)
            .and_then(|reader| reader.with_guessed_format())
            .map_err(MaskError::from)?
            .into_dimensions()
            .map_err(MaskError::from)?;
        let mask_path = output_path(request, input)?;
        let mask_reader = ImageReader::open(&mask_path)
            .and_then(|reader| reader.with_guessed_format())
            .map_err(|error| {
                MaskError::image(format!(
                    "missing or unreadable mask {}: {error}",
                    mask_path.display()
                ))
            })?;
        if mask_reader.format() != Some(ImageFormat::Png) {
            return Err(MaskError::image(format!(
                "mask is not PNG: {}",
                mask_path.display()
            )));
        }
        let mask_dimensions = mask_reader.into_dimensions().map_err(MaskError::from)?;
        if mask_dimensions != source_dimensions {
            return Err(MaskError::image(format!(
                "mask dimensions {}x{} do not match source {}x{}: {}",
                mask_dimensions.0,
                mask_dimensions.1,
                source_dimensions.0,
                source_dimensions.1,
                mask_path.display()
            )));
        }
        if !is_valid_mask_file(&mask_path, source_dimensions.0, source_dimensions.1) {
            return Err(MaskError::image(format!("invalid binary PNG mask: {}", mask_path.display())));
        }
    }
    Ok(files.len())
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

fn output_path(request: &MaskRequest, input: &Path) -> MaskResult<PathBuf> {
    let mut relative = if request.images_dir.is_file() {
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
    relative.set_extension("png");
    Ok(request.masks_dir.join(relative))
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

fn write_mask_atomic(path: &Path, width: u32, height: u32, data: &[u8]) -> MaskResult<()> {
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
    // Encode directly from the shared mask slice as lossless 8-bit grayscale PNG.
    let write_result = image::save_buffer_with_format(
        &temporary,
        data,
        width,
        height,
        ColorType::L8,
        ImageFormat::Png,
    );
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
    if reader.format() != Some(ImageFormat::Png) {
        return false;
    }
    let Ok(image) = reader.decode() else {
        return false;
    };
    image.dimensions() == (width, height) && image.color() == ColorType::L8
        && image.as_bytes().iter().all(|value| matches!(value, 0 | 255))
}

fn validate_segmentation(mask: &SegmentationMask, dimensions: (u32, u32)) -> MaskResult<()> {
    if (mask.width, mask.height) != dimensions || mask.data.len() != dimensions.0 as usize * dimensions.1 as usize {
        return Err(MaskError::inference("segmentation dimensions do not match inference image"));
    }
    Ok(())
}

fn infer_rotations<E: MaskEngine + ?Sized>(engine: &E, image: &DynamicImage, request: &MaskRequest,
    mask_sky: bool, cancel: &CancelToken) -> MaskResult<SegmentationMask> {
    if cancel.is_cancelled() { return Err(MaskError::Cancelled); }
    let mut result = engine.generate_exclusion_mask(image, &request.classes, request.confidence, mask_sky, cancel)?;
    validate_segmentation(&result, image.dimensions())?;
    if !request.classes.is_empty() && request.rotations == 4 {
        for turn in 1..4 {
            if cancel.is_cancelled() { return Err(MaskError::Cancelled); }
            let rotated = match turn { 1 => image.rotate90(), 2 => image.rotate180(), _ => image.rotate270() };
            let mask = engine.generate_exclusion_mask(&rotated, &request.classes, request.confidence, false, cancel)?;
            validate_segmentation(&mask, rotated.dimensions())?;
            let gray = image::GrayImage::from_raw(mask.width, mask.height, mask.data)
                .ok_or_else(|| MaskError::inference("invalid rotated mask"))?;
            let restored = match turn { 1 => image::imageops::rotate270(&gray), 2 => image::imageops::rotate180(&gray), _ => image::imageops::rotate90(&gray) };
            for (pixel, rotated) in result.data.iter_mut().zip(restored.into_raw()) { *pixel |= rotated; }
        }
    }
    for value in &mut result.data { *value = if *value == 0 { 0 } else { 255 }; }
    Ok(result)
}

fn dilate_exclusions(mask: &mut SegmentationMask, radius: u32) {
    if radius == 0 { return; }
    let (w, h) = (mask.width as usize, mask.height as usize);
    let r = radius as usize;
    // Two separable passes give a square dilation in O(radius * pixels).
    let mut horizontal = vec![0; mask.data.len()];
    for y in 0..h { for x in 0..w {
        if mask.data[y*w + x.saturating_sub(r)..y*w + (x+r+1).min(w)].iter().any(|v| *v != 0) { horizontal[y*w+x] = 255; }
    }}
    for y in 0..h { for x in 0..w {
        mask.data[y*w+x] = if (y.saturating_sub(r)..(y+r+1).min(h)).any(|row| horizontal[row*w+x] != 0) { 255 } else { 0 };
    }}
}

fn emit_progress(
    callback: &impl Fn(MaskProgress),
    index: usize,
    total: usize,
    completed: usize,
    input: &Path,
    mask_path: &Path,
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
            ..MaskRequest::default()
        }
    }

    #[test]
    fn writes_lossless_png_by_stem_with_circle_black() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let mut request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        let source = request.images_dir.join("lens0/frame.jpg");
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
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
        assert_eq!(mask.get_pixel(2, 2)[0], 255);
        assert!(!request.masks_dir.join("lens0/.frame.png.part").exists());
        Ok(())
    }

    #[test]
    fn terminal_coverage_check_rejects_a_missing_mask() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        let source = request.images_dir.join("lens0/frame.jpg");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(8, 6, Rgb([100, 100, 100])).save(&source)?;
        let engine = FakeEngine {
            calls: AtomicUsize::new(0),
            exclusion: 0,
        };
        process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(verify_mask_output_coverage(&request)?, 1);

        fs::remove_file(request.masks_dir.join("lens0/frame.png"))?;
        let error = verify_mask_output_coverage(&request).unwrap_err();
        assert!(error.to_string().contains("missing or unreadable mask"));
        Ok(())
    }

    #[test]
    fn rejects_sources_that_map_to_the_same_png_mask() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        ImageBuffer::<Rgb<u8>, _>::from_pixel(2, 2, Rgb([100, 100, 100]))
            .save(request.images_dir.join("lens0/frame.jpg"))?;
        ImageBuffer::<Rgb<u8>, _>::from_pixel(2, 2, Rgb([100, 100, 100]))
            .save(request.images_dir.join("lens0/frame.png"))?;
        let engine = FakeEngine {
            calls: AtomicUsize::new(0),
            exclusion: 0,
        };

        let error = process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})
            .unwrap_err();

        assert!(error
            .to_string()
            .replace('\\', "/")
            .contains("map to the same mask lens0/frame.png"));
        assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
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
    fn mask_receipts_invalidate_changed_settings_images_weights_and_nonbinary_outputs() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let mut request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        let input = request.images_dir.join("lens0/frame.png");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(8, 8, Rgb([100, 100, 100])).save(&input)?;
        request.yolo_model = Some(dir.path().join("model.onnx"));
        fs::write(request.yolo_model.as_ref().unwrap(), b"version one")?;
        let engine = FakeEngine { calls: AtomicUsize::new(0), exclusion: 0 };
        let run = |request: &MaskRequest| process_mask_batch_with_engine(request, &CancelToken::new(), &engine, |_| {});
        assert_eq!(run(&request)?.skipped, 0);
        assert_eq!(run(&request)?.skipped, 1);
        request.confidence = 0.10;
        assert_eq!(run(&request)?.skipped, 0);
        fs::write(request.yolo_model.as_ref().unwrap(), b"version two")?;
        assert_eq!(run(&request)?.skipped, 0);
        ImageBuffer::<Rgb<u8>, _>::from_pixel(8, 8, Rgb([110, 100, 100])).save(&input)?;
        assert_eq!(run(&request)?.skipped, 0);
        ImageBuffer::<Luma<u8>, _>::from_pixel(8, 8, Luma([128])).save(request.masks_dir.join("lens0/frame.png"))?;
        assert!(verify_mask_output_coverage(&request).is_err());
        assert_eq!(run(&request)?.skipped, 0);
        Ok(())
    }

    #[test]
    fn rotations_restore_rectangular_coordinates_and_dilation_stays_semantic() -> MaskResult<()> {
        struct Corner;
        impl MaskEngine for Corner {
            fn generate_exclusion_mask(&self, image: &DynamicImage, _: &[String], _: f32, _: bool, _: &CancelToken) -> MaskResult<SegmentationMask> {
                let mut data = vec![0; (image.width()*image.height()) as usize]; data[0] = 255;
                SegmentationMask::new(image.width(), image.height(), data)
            }
        }
        let request = MaskRequest { classes: vec!["person".into()], rotations: 4, ..Default::default() };
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 4, Rgb([100,100,100])));
        let mut mask = infer_rotations(&Corner, &image, &request, false, &CancelToken::new())?;
        assert_eq!(mask.data.iter().enumerate().filter_map(|(i,v)| (*v != 0).then_some(i)).collect::<Vec<_>>(), vec![0,7,24,31]);
        dilate_exclusions(&mut mask, 1);
        assert_eq!(mask.data.iter().filter(|v| **v != 0).count(), 16);
        Ok(())
    }

    #[test]
    fn extra_masks_preserve_reviewed_exclusions_and_invalidate_resume() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let mut request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        request.additional_masks_dir = Some(dir.path().join("reviewed"));
        let extra = request.additional_masks_dir.as_ref().unwrap().join("lens0");
        fs::create_dir_all(&extra)?;
        let input = request.images_dir.join("lens0/frame.png");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(8, 8, Rgb([100,100,100])).save(&input)?;
        let mut manual = ImageBuffer::<Luma<u8>, _>::from_pixel(8, 8, Luma([255]));
        manual.put_pixel(4,4,Luma([0])); manual.save(extra.join("frame.png"))?;
        let engine = FakeEngine { calls: AtomicUsize::new(0), exclusion: 0 };
        process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(image::open(request.masks_dir.join("lens0/frame.png"))?.into_luma8().get_pixel(4,4)[0], 0);
        manual.put_pixel(3,3,Luma([0])); manual.save(extra.join("frame.png"))?;
        assert_eq!(process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?.skipped, 0);
        assert_eq!(image::open(request.masks_dir.join("lens0/frame.png"))?.into_luma8().get_pixel(3,3)[0], 0);
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
    fn skips_only_when_the_png_mask_is_valid() -> MaskResult<()> {
        let dir = TempDir::new()?;
        let request = request(&dir);
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        let source = request.images_dir.join("lens0/frame.jpg");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(2, 2, Rgb([100, 100, 100])).save(&source)?;
        let engine = FakeEngine {
            calls: AtomicUsize::new(0),
            exclusion: 0,
        };
        process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        let first_calls = engine.calls.load(Ordering::Relaxed);
        let mask_path = request.masks_dir.join("lens0/frame.png");
        let mask_before = fs::read(&mask_path)?;
        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(summary.skipped, 1);
        assert_eq!(engine.calls.load(Ordering::Relaxed), first_calls);
        assert_eq!(fs::read(&mask_path)?, mask_before);
        fs::write(&mask_path, b"broken")?;
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
        request.yolo_model = Some(dir.path().join("fixture-yolo.onnx"));
        fs::write(request.yolo_model.as_ref().unwrap(), b"fixture not a loadable ONNX model")?;
        fs::create_dir_all(request.images_dir.join("lens0"))?;
        fs::create_dir_all(request.masks_dir.join("lens0"))?;

        ImageBuffer::<Rgb<u8>, _>::from_pixel(2, 2, Rgb([100, 100, 100]))
            .save(request.images_dir.join("lens0/frame.jpg"))?;
        let engine = FakeEngine { calls: AtomicUsize::new(0), exclusion: 0 };
        process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;

        // Resolving/hashing the weights is required. A session must not be
        // loaded for an unchanged batch (the dummy weights cannot be loaded).
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
    #[ignore = "requires production YOLO, calibrated model, native image, and GPU"]
    fn processes_calibrated_views_with_production_model_and_reuses_receipt() -> MaskResult<()> {
        let image = PathBuf::from(std::env::var_os("GS360_TEST_PERSON_IMAGE").expect("native image required"));
        let model = PathBuf::from(std::env::var_os("GS360_TEST_CALIBRATION_MODEL").expect("calibration required"));
        let lens = std::env::var("GS360_TEST_LENS").unwrap_or_else(|_| "lens1".into());
        let temp = TempDir::new()?;
        let mut request = request(&temp);
        let input = request.images_dir.join(&lens).join(image.file_name().unwrap());
        fs::create_dir_all(input.parent().unwrap())?;
        fs::copy(&image, &input)?;
        request.classes = vec!["person".into()];
        request.yolo_model = Some(PathBuf::from(std::env::var_os("GS360_TEST_YOLO_MODEL").expect("YOLO required")));
        request.execution_provider = Some(std::env::var("GS360_TEST_GPU_PROVIDER").unwrap_or_else(|_| "DirectML".into()));
        request.calibrated_cameras = read_calibration(&model)?;
        request.rotations = 4;
        request.confidence = 0.10;
        request.dilation = 2;
        let summary = process_mask_batch(&request, &CancelToken::new(), |_| {})?;
        assert_eq!(summary.succeeded,1, "{:?}",summary.failures);
        assert_eq!(summary.failed,0);
        let dimensions = image::image_dimensions(&input)?;
        assert!(is_valid_mask_file(&output_path(&request,&input)?,dimensions.0,dimensions.1));
        let resumed = process_mask_batch(&request, &CancelToken::new(), |_| {})?;
        assert_eq!(resumed.skipped,1);
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
            let first = ImageReader::open(request.masks_dir.join("frame-0.png"))?
                .decode()?
                .to_luma8();
            let second = ImageReader::open(request.masks_dir.join("frame-1.png"))?
                .decode()?
                .to_luma8();
            assert_eq!(first, second);
        }
        Ok(())
    }
}
