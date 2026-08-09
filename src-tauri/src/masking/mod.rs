//! Native-fisheye instance masks for the studio backend.
//!
//! The module treats every image below `images_dir` (including `lens0/`,
//! `lens1/`, and future `lens*` directories) as an independent camera frame.
//! `masks/` keeps the exact relative filename and uses white for pixels retained
//! by SfM/training and black for excluded pixels.  `masks_colmap/` keeps the same
//! relative hierarchy but appends `.png` to the complete image filename (for
//! example `lens0/frame.jpg.png`), matching COLMAP's `ImageReader.mask_path`
//! contract.  The portion outside the configured fisheye circle is always black.
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

use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageReader, Luma};
use serde::Serialize;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

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
    /// Radius as a ratio of the shorter source dimension (normally ~0.497).
    pub valid_radius_ratio: f32,
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
            confidence: 0.25,
            valid_radius_ratio: 0.497,
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
    pub yolo: YoloSegPipeline,
    pub skyseg: Option<SkysegPipeline>,
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
            request.mask_sky,
            cancel,
            on_download,
        )?;
        let yolo = YoloSegPipeline::load(&paths, request.execution_provider.as_deref())?;
        let skyseg = if request.mask_sky {
            Some(SkysegPipeline::load(&paths, &yolo.execution_provider)?)
        } else {
            None
        };
        Ok(Self { yolo, skyseg })
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
        let mut result = self
            .yolo
            .generate_exclusion_mask(image, classes, confidence, cancel)?;
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
    on_progress: impl Fn(MaskProgress),
) -> MaskResult<MaskSummary> {
    validate_request(request)?;
    if request.classes.is_empty() && !request.mask_sky {
        return process_with_engine(request, cancel, &NoExclusionsEngine, on_progress);
    }
    on_progress(MaskProgress {
        index: 0,
        total: 0,
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
        input: request.images_dir.clone(),
        mask_path: request.masks_dir.clone(),
        colmap_mask_path: request.colmap_masks_dir.clone(),
        stage: MaskStage::LoadingModel,
        fraction: 0.0,
        message: format!(
            "模型已載入 {}，CPU 推論回退已停用",
            engine.yolo.execution_provider
        ),
    });
    process_with_engine(request, cancel, &engine, on_progress)
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
    on_progress: impl Fn(MaskProgress),
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
    on_progress: impl Fn(MaskProgress),
) -> MaskResult<MaskSummary>
where
    E: MaskEngine + ?Sized,
{
    let files = collect_images(&request.images_dir)?;
    let total = files.len();
    let mut summary = MaskSummary {
        total,
        succeeded: 0,
        skipped: 0,
        failed: 0,
        cancelled: false,
        failures: Vec::new(),
    };

    for (index, input) in files.iter().enumerate() {
        let (mask_path, colmap_path) = output_paths(request, input)?;
        if cancel.is_cancelled() {
            summary.cancelled = true;
            emit_progress(
                &on_progress,
                index,
                total,
                input,
                &mask_path,
                &colmap_path,
                MaskStage::Cancelled,
                index as f32 / total.max(1) as f32,
                "遮罩處理已取消",
            );
            break;
        }

        let image = match ImageReader::open(input)
            .and_then(|reader| reader.with_guessed_format())
            .map_err(MaskError::from)
            .and_then(|reader| reader.decode().map_err(MaskError::from))
        {
            Ok(image) => image,
            Err(error) => {
                summary.failed += 1;
                summary.failures.push(MaskFailure {
                    input: input.clone(),
                    error: error.to_string(),
                });
                emit_progress(
                    &on_progress,
                    index,
                    total,
                    input,
                    &mask_path,
                    &colmap_path,
                    MaskStage::Failed,
                    (index + 1) as f32 / total.max(1) as f32,
                    &error.to_string(),
                );
                continue;
            }
        };

        let (width, height) = image.dimensions();
        if request.skip_verified
            && is_valid_mask_file(&mask_path, width, height)
            && is_valid_mask_file(&colmap_path, width, height)
        {
            summary.succeeded += 1;
            summary.skipped += 1;
            emit_progress(
                &on_progress,
                index,
                total,
                input,
                &mask_path,
                &colmap_path,
                MaskStage::Skipped,
                (index + 1) as f32 / total.max(1) as f32,
                "已確認遮罩存在，已略過",
            );
            continue;
        }

        emit_progress(
            &on_progress,
            index,
            total,
            input,
            &mask_path,
            &colmap_path,
            MaskStage::Inference,
            index as f32 / total.max(1) as f32,
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
                summary.cancelled = true;
                emit_progress(
                    &on_progress,
                    index,
                    total,
                    input,
                    &mask_path,
                    &colmap_path,
                    MaskStage::Cancelled,
                    index as f32 / total.max(1) as f32,
                    "遮罩處理已取消",
                );
                break;
            }
            Err(error) => {
                summary.failed += 1;
                summary.failures.push(MaskFailure {
                    input: input.clone(),
                    error: error.to_string(),
                });
                emit_progress(
                    &on_progress,
                    index,
                    total,
                    input,
                    &mask_path,
                    &colmap_path,
                    MaskStage::Failed,
                    (index + 1) as f32 / total.max(1) as f32,
                    &error.to_string(),
                );
                continue;
            }
        };

        if exclusions.width != width || exclusions.height != height {
            summary.failed += 1;
            let error = format!(
                "backend returned {}x{} for source {}x{}",
                exclusions.width, exclusions.height, width, height
            );
            summary.failures.push(MaskFailure {
                input: input.clone(),
                error: error.clone(),
            });
            emit_progress(
                &on_progress,
                index,
                total,
                input,
                &mask_path,
                &colmap_path,
                MaskStage::Failed,
                (index + 1) as f32 / total.max(1) as f32,
                &error,
            );
            continue;
        }

        if cancel.is_cancelled() {
            summary.cancelled = true;
            emit_progress(
                &on_progress,
                index,
                total,
                input,
                &mask_path,
                &colmap_path,
                MaskStage::Cancelled,
                index as f32 / total.max(1) as f32,
                "已在寫入遮罩前取消",
            );
            break;
        }

        let keep = build_keep_mask(width, height, request.valid_radius_ratio, &exclusions.data);
        emit_progress(
            &on_progress,
            index,
            total,
            input,
            &mask_path,
            &colmap_path,
            MaskStage::Writing,
            index as f32 / total.max(1) as f32,
            "正在寫入遮罩檔案",
        );
        if let Err(error) = write_mask_atomic(&mask_path, width, height, &keep, false)
            .and_then(|_| write_mask_atomic(&colmap_path, width, height, &keep, true))
        {
            summary.failed += 1;
            summary.failures.push(MaskFailure {
                input: input.clone(),
                error: error.to_string(),
            });
            emit_progress(
                &on_progress,
                index,
                total,
                input,
                &mask_path,
                &colmap_path,
                MaskStage::Failed,
                (index + 1) as f32 / total.max(1) as f32,
                &error.to_string(),
            );
            continue;
        }
        summary.succeeded += 1;
        emit_progress(
            &on_progress,
            index,
            total,
            input,
            &mask_path,
            &colmap_path,
            MaskStage::Completed,
            (index + 1) as f32 / total.max(1) as f32,
            "遮罩處理完成",
        );
    }

    Ok(summary)
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

fn build_keep_mask(width: u32, height: u32, radius_ratio: f32, exclusions: &[u8]) -> Vec<u8> {
    let mut keep = vec![0u8; (width as usize).saturating_mul(height as usize)];
    let radius = (width.min(height) as f32 * radius_ratio).floor();
    let center_x = (width / 2) as f32;
    let center_y = (height / 2) as f32;
    let radius_squared = radius * radius;
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            if dx * dx + dy * dy <= radius_squared && exclusions[index] == 0 {
                keep[index] = 255;
            }
        }
    }
    keep
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
    let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(width, height, data.to_vec())
        .ok_or_else(|| MaskError::image("failed to build output mask"))?;
    let write_result = image.save_with_format(&temporary, format);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(MaskError::from(error));
    }
    // Keep rename on the same directory/filesystem.  A complete temporary file
    // is never visible at the final path, and a stale `.part` can be removed on
    // the next run without invalidating an existing mask.
    let sync_result = fs::File::open(&temporary).and_then(|file| file.sync_all());
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
    use tempfile::TempDir;

    struct FakeEngine {
        calls: AtomicUsize,
        exclusion: u8,
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
        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(summary.skipped, 1);
        assert_eq!(engine.calls.load(Ordering::Relaxed), first_calls);
        fs::write(
            request.colmap_masks_dir.join("lens0/frame.png.png"),
            b"broken",
        )?;
        let summary =
            process_mask_batch_with_engine(&request, &CancelToken::new(), &engine, |_| {})?;
        assert_eq!(summary.succeeded, 1);
        assert_eq!(summary.skipped, 0);
        assert_eq!(engine.calls.load(Ordering::Relaxed), first_calls + 1);
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
}
