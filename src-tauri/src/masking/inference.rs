//! YOLO11 segmentation inference used by the native-fisheye masker.
//!
//! This is intentionally kept independent from the old equirectangular pipeline:
//! native lens frames are loaded and decoded at their original aspect ratio, then
//! letterboxed for YOLO and projected back to the original pixel grid.

use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, TryLockError,
};
use std::{fs::File, io::Read};

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use ndarray::{Array4, ArrayD, Axis, Ix3, Ix4};
#[cfg(any(
    target_os = "macos",
    target_os = "windows",
    feature = "cuda",
    feature = "coreml",
    feature = "directml",
    feature = "xnnpack",
    feature = "tensorrt",
    feature = "nvrtx",
    feature = "webgpu"
))]
use ort::ep::ExecutionProvider;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::{OutputSelector, RunOptions, Session, SessionInputValue};
use ort::value::Tensor;
use sha2::{Digest, Sha256};

use super::{CancelToken, MaskError, MaskResult, SegmentationMask};
use crate::masking::models::ModelPaths;

const INPUT_SIZE: u32 = 640;
const MASK_DIM: usize = 32;
const CLASS_COUNT: usize = 80;
const IOU_THRESHOLD: f32 = 0.45;
const MAX_DETECTIONS: usize = 300;

/// COCO names emitted by the Ultralytics YOLO11 segmentation export.
pub const COCO_CLASSES: [&str; CLASS_COUNT] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[derive(Debug, Clone, Copy)]
struct LetterboxInfo {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    original_width: u32,
    original_height: u32,
}

#[derive(Debug, Clone)]
struct Detection {
    bbox_xyxy: [f32; 4],
    score: f32,
    class_id: usize,
    mask_coeffs: [f32; MASK_DIM],
}

/// A small pool of independently-created sessions.  ORT's `Run` API requires
/// exclusive access to each `Session`; the pool therefore serializes runs per
/// session while allowing different sessions to run in parallel where the
/// provider supports it (CUDA and DirectML).
pub(crate) struct SessionPool {
    sessions: Vec<Mutex<Session>>,
    next: AtomicUsize,
}

impl SessionPool {
    pub(crate) fn new(sessions: Vec<Session>) -> MaskResult<Self> {
        if sessions.is_empty() {
            return Err(MaskError::inference("session pool cannot be empty"));
        }
        Ok(Self {
            sessions: sessions.into_iter().map(Mutex::new).collect(),
            next: AtomicUsize::new(0),
        })
    }

    pub(crate) fn with_session<T, F>(&self, f: F) -> MaskResult<T>
    where
        F: FnOnce(&mut Session) -> MaskResult<T>,
    {
        let session_count = self.sessions.len();
        if session_count == 0 {
            return Err(MaskError::inference("session pool cannot be empty"));
        }
        let start = self.next.fetch_add(1, Ordering::Relaxed) % session_count;
        let mut guard = None;
        for offset in 0..session_count {
            let index = (start + offset) % session_count;
            match self.sessions[index].try_lock() {
                Ok(session) => {
                    guard = Some(session);
                    break;
                }
                Err(TryLockError::WouldBlock) => {}
                Err(TryLockError::Poisoned(_)) => {
                    return Err(MaskError::inference("ONNX session lock poisoned"));
                }
            }
        }
        let mut guard = match guard {
            Some(guard) => guard,
            None => self.sessions[start]
                .lock()
                .map_err(|_| MaskError::inference("ONNX session lock poisoned"))?,
        };
        f(&mut guard)
    }
}

/// Build the provider-specific session pool.  The first session is required;
/// additional GPU replicas are optional and fall back to the sessions
/// already created when a driver or VRAM limit rejects another session.
pub(crate) fn load_session_pool(
    model_path: &Path,
    provider: &str,
    cache_dir: Option<&Path>,
) -> MaskResult<SessionPool> {
    let provider = normalize_provider_name(provider);
    if provider == "CPU" {
        return Err(MaskError::inference(
            "CPU mask inference is disabled; select a GPU execution provider",
        ));
    }
    let mut cache_dir = prepare_coreml_cache_dir(&provider, cache_dir);
    let requested_pool_size = session_pool_size_for_provider(&provider);
    let mut sessions = Vec::with_capacity(requested_pool_size);
    let mut index = 0;
    let mut cache_disabled_after_failure = false;
    while index < requested_pool_size {
        let result = (|| {
            let mut builder = session_builder_for_provider(&provider)?;
            register_execution_provider_with_cache(&mut builder, &provider, cache_dir.as_deref())?;
            builder
                .commit_from_file(model_path)
                .map_err(|error| MaskError::inference(format!("{provider} session: {error}")))
        })();
        match result {
            Ok(session) => {
                sessions.push(session);
                index += 1;
            }
            Err(_error)
                if index == 0
                    && cache_dir.is_some()
                    && !cache_disabled_after_failure
                    && provider == "CoreML" =>
            {
                // A read-only cache directory may exist already, so
                // create_dir_all alone cannot prove that CoreML can write it.
                // Retry one time without caching to keep model loading safe.
                cache_dir = None;
                cache_disabled_after_failure = true;
            }
            Err(_error) if index > 0 => {
                // A second CUDA/DirectML session is only an optional throughput
                // optimization. Driver/VRAM limits must not make a model that
                // already loaded successfully unusable.
                break;
            }
            Err(error) => return Err(error),
        }
    }
    SessionPool::new(sessions)
}

/// YOLO11 ONNX segmentation pipeline.
#[derive(Clone)]
pub struct YoloSegPipeline {
    session: Arc<SessionPool>,
    input_name: String,
    output0_name: String,
    output1_name: String,
    pub execution_provider: String,
}

impl YoloSegPipeline {
    /// Load YOLO11 from resolved model paths. Provider loading is attempted in
    /// the requested/platform order, but never falls back to CPU.
    pub fn load(paths: &ModelPaths, requested_provider: Option<&str>) -> MaskResult<Self> {
        let model_path = paths.yolo.as_ref().ok_or_else(|| {
            MaskError::model("YOLO11 model is required when object masking is enabled")
        })?;
        let candidates = provider_candidates(requested_provider);
        if candidates.is_empty() {
            return Err(MaskError::model(
                "GPU execution provider is required; CPU mask inference is disabled",
            ));
        }
        let mut errors = Vec::new();
        for provider in candidates {
            match Self::load_with_provider(model_path, &provider) {
                Ok(pipeline) => return Ok(pipeline),
                Err(error) => errors.push(format!("{provider}: {error}")),
            }
        }
        Err(MaskError::model(format!(
            "unable to load YOLO11 segmentation model {} ({})",
            model_path.display(),
            errors.join("; ")
        )))
    }

    /// Load one provider explicitly.  The feature gates mirror the provider
    /// names used by the gs360masker build.  Unsupported providers return an
    /// actionable error and are subsequently handled by [`Self::load`].
    pub fn load_with_provider(model_path: &Path, provider: &str) -> MaskResult<Self> {
        let provider = normalize_provider_name(provider);
        let cache_dir = (provider == "CoreML")
            .then(|| default_coreml_cache_dir(model_path))
            .flatten();
        Self::load_with_cache(model_path, &provider, cache_dir.as_deref())
    }

    /// Load one provider explicitly, optionally reusing a compiled CoreML
    /// graph from `cache_dir`.  Cache setup is best-effort: an inaccessible
    /// directory simply disables the cache and does not prevent model loading.
    pub fn load_with_cache(
        model_path: &Path,
        provider: &str,
        cache_dir: Option<&Path>,
    ) -> MaskResult<Self> {
        let provider = normalize_provider_name(provider);
        if provider == "CPU" {
            return Err(MaskError::inference(
                "CPU mask inference is disabled; select a GPU execution provider",
            ));
        }
        let session = load_session_pool(model_path, &provider, cache_dir)?;
        let (input_name, output0_name, output1_name) = session.with_session(|session| {
            let input_name = session
                .inputs()
                .first()
                .map(|input| input.name().to_string())
                .ok_or_else(|| MaskError::model("YOLO11 model has no input"))?;
            let output0_name = session
                .outputs()
                .first()
                .map(|output| output.name().to_string())
                .ok_or_else(|| MaskError::model("YOLO11 model has no detection output"))?;
            let output1_name = session
                .outputs()
                .get(1)
                .map(|output| output.name().to_string())
                .ok_or_else(|| MaskError::model("YOLO11 model has no prototype output"))?;
            Ok((input_name, output0_name, output1_name))
        })?;

        Ok(Self {
            session: Arc::new(session),
            input_name,
            output0_name,
            output1_name,
            execution_provider: provider,
        })
    }

    /// Generate an exclusion mask: 255 means a detected object should be
    /// removed from the keep mask, and 0 means no matching object.
    pub fn generate_exclusion_mask(
        &self,
        image: &DynamicImage,
        classes: &[String],
        confidence: f32,
        cancel: &CancelToken,
    ) -> MaskResult<SegmentationMask> {
        if cancel.is_cancelled() {
            return Err(MaskError::Cancelled);
        }
        let has_requested_classes = classes
            .iter()
            .any(|class_name| !class_name.trim().is_empty());
        let class_ids = class_ids_for_targets(classes);
        // Empty means "no object classes" so a sky-only request does not mask
        // every COCO detection.  Likewise, unknown names must not silently
        // broaden the selection to all classes.
        if !has_requested_classes || class_ids.is_empty() {
            let (width, height) = image.dimensions();
            return SegmentationMask::new(width, height, vec![0; pixel_count(width, height)?]);
        }
        let (pred, proto, letterbox) = self.run_raw(image)?;
        if cancel.is_cancelled() {
            return Err(MaskError::Cancelled);
        }
        let detections =
            decode_detections(pred, &class_ids, confidence.clamp(0.0, 1.0), letterbox)?;
        let (width, height) = image.dimensions();
        if detections.is_empty() {
            return Ok(SegmentationMask::new(
                width,
                height,
                vec![0; pixel_count(width, height)?],
            )?);
        }

        let merged = decode_merged_mask(proto, &detections, letterbox)?;
        SegmentationMask::new(width, height, merged)
    }

    fn run_raw(
        &self,
        image: &DynamicImage,
    ) -> MaskResult<(ArrayD<f32>, ArrayD<f32>, LetterboxInfo)> {
        let (input, letterbox) = preprocess_yolo(image)?;
        let run_options = RunOptions::new()
            .map_err(|error| MaskError::inference(error.to_string()))?
            .with_outputs(
                OutputSelector::no_default()
                    .with(self.output0_name.clone())
                    .with(self.output1_name.clone()),
            );
        let (output0_value, output1_value) = self.session.with_session(|session| {
            let outputs = session
                .run_with_options(
                    vec![(
                        self.input_name.clone(),
                        SessionInputValue::from(
                            Tensor::from_array(input)
                                .map_err(|error| MaskError::inference(error.to_string()))?,
                        ),
                    )],
                    &run_options,
                )
                .map_err(|error| MaskError::inference(format!("YOLO11 inference: {error}")))?;
            let mut outputs = outputs;
            let output0 = outputs
                .remove(self.output0_name.as_str())
                .ok_or_else(|| MaskError::inference("YOLO11 detection output was not returned"))?;
            let output1 = outputs
                .remove(self.output1_name.as_str())
                .ok_or_else(|| MaskError::inference("YOLO11 prototype output was not returned"))?;
            Ok((output0, output1))
        })?;
        let output0 = output0_value
            .try_extract_array::<f32>()
            .map_err(|error| MaskError::inference(error.to_string()))?
            .to_owned();
        let output1 = output1_value
            .try_extract_array::<f32>()
            .map_err(|error| MaskError::inference(error.to_string()))?
            .to_owned();
        Ok((output0, output1, letterbox))
    }
}

fn preprocess_yolo(image: &DynamicImage) -> MaskResult<(Array4<f32>, LetterboxInfo)> {
    let rgb = image.to_rgb8();
    let (original_width, original_height) = rgb.dimensions();
    if original_width == 0 || original_height == 0 {
        return Err(MaskError::invalid_input("image has zero dimensions"));
    }
    let scale =
        (INPUT_SIZE as f32 / original_width as f32).min(INPUT_SIZE as f32 / original_height as f32);
    let resized_width = ((original_width as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let resized_height = ((original_height as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let resized = resize_yolo_source(rgb, resized_width, resized_height);
    let mut canvas =
        image::RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, image::Rgb([114, 114, 114]));
    let pad_x = ((INPUT_SIZE - resized_width) / 2) as i64;
    let pad_y = ((INPUT_SIZE - resized_height) / 2) as i64;
    image::imageops::replace(&mut canvas, &resized, pad_x, pad_y);

    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    let plane_len = (INPUT_SIZE * INPUT_SIZE) as usize;
    let input_data = input
        .as_slice_mut()
        .ok_or_else(|| MaskError::inference("YOLO input tensor is not contiguous"))?;
    for (index, pixel) in canvas.pixels().enumerate() {
        input_data[index] = pixel[0] as f32 / 255.0;
        input_data[plane_len + index] = pixel[1] as f32 / 255.0;
        input_data[plane_len * 2 + index] = pixel[2] as f32 / 255.0;
    }
    Ok((
        input,
        LetterboxInfo {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            original_width,
            original_height,
        },
    ))
}

fn resize_yolo_source(rgb: image::RgbImage, width: u32, height: u32) -> image::RgbImage {
    if rgb.width() == width && rgb.height() == height {
        rgb
    } else {
        image::imageops::resize(&rgb, width, height, FilterType::Triangle)
    }
}

fn decode_detections(
    output: ArrayD<f32>,
    target_class_ids: &[usize],
    confidence: f32,
    letterbox: LetterboxInfo,
) -> MaskResult<Vec<Detection>> {
    if target_class_ids.is_empty() {
        return Ok(Vec::new());
    }
    if target_class_ids
        .iter()
        .any(|class_id| *class_id >= CLASS_COUNT)
    {
        return Err(MaskError::invalid_input(
            "YOLO target class id is outside the COCO class range",
        ));
    }
    let output = output
        .into_dimensionality::<Ix3>()
        .map_err(|error| MaskError::inference(format!("unexpected YOLO output shape: {error}")))?;
    let output = output.index_axis(Axis(0), 0);
    let shape = output.shape();
    let (channels, predictions, transposed) = match shape {
        [a, b] if *a >= 4 + CLASS_COUNT + MASK_DIM => (*a, *b, false),
        [a, b] if *b >= 4 + CLASS_COUNT + MASK_DIM => (*b, *a, true),
        _ => {
            return Err(MaskError::inference(format!(
                "unexpected YOLO detection output shape: {:?}",
                shape
            )))
        }
    };
    if channels < 4 + CLASS_COUNT + MASK_DIM {
        return Err(MaskError::inference(
            "YOLO detection output has too few channels",
        ));
    }

    let at = |channel: usize, prediction: usize| {
        if transposed {
            output[[prediction, channel]]
        } else {
            output[[channel, prediction]]
        }
    };
    let mut detections = Vec::new();
    for prediction in 0..predictions {
        let cx = at(0, prediction);
        let cy = at(1, prediction);
        let width = at(2, prediction);
        let height = at(3, prediction);
        if ![cx, cy, width, height]
            .iter()
            .all(|value| value.is_finite())
        {
            continue;
        }
        let mut best_class = None;
        let mut best_score = f32::NEG_INFINITY;
        for &class_id in target_class_ids {
            let score = at(4 + class_id, prediction);
            if score > best_score {
                best_score = score;
                best_class = Some(class_id);
            }
        }
        let Some(class_id) = best_class else { continue };
        if !best_score.is_finite() || best_score < confidence {
            continue;
        }
        let mut mask_coeffs = [0.0f32; MASK_DIM];
        for (index, coeff) in mask_coeffs.iter_mut().enumerate() {
            *coeff = at(4 + CLASS_COUNT + index, prediction);
        }
        let bbox = scale_box_from_letterbox(
            [
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            ],
            letterbox,
        );
        if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            continue;
        }
        detections.push(Detection {
            bbox_xyxy: bbox,
            score: best_score,
            class_id,
            mask_coeffs,
        });
    }
    non_max_suppression(detections)
}

fn decode_merged_mask(
    proto: ArrayD<f32>,
    detections: &[Detection],
    letterbox: LetterboxInfo,
) -> MaskResult<Vec<u8>> {
    let proto = proto.into_dimensionality::<Ix4>().map_err(|error| {
        MaskError::inference(format!("unexpected YOLO prototype shape: {error}"))
    })?;
    let proto = proto.index_axis(Axis(0), 0);
    let shape = proto.shape();
    if shape.len() != 3 || shape[0] < MASK_DIM {
        return Err(MaskError::inference(format!(
            "unexpected YOLO prototype dimensions: {:?}",
            shape
        )));
    }
    let mask_h = shape[1];
    let mask_w = shape[2];
    let plane_len = mask_w * mask_h;
    let proto = proto
        .as_slice()
        .ok_or_else(|| MaskError::inference("YOLO prototype output is not contiguous"))?;
    let output_width = letterbox.original_width as usize;
    let output_height = letterbox.original_height as usize;
    let mut merged = vec![0u8; pixel_count(letterbox.original_width, letterbox.original_height)?];
    for detection in detections {
        let mut low_res = vec![0.0f32; plane_len];
        // Walk each contiguous prototype plane once. This avoids ndarray's
        // checked three-dimensional indexing in the innermost 32-channel dot
        // product while preserving the exact accumulation order per channel.
        for (channel, coefficient) in detection.mask_coeffs.iter().copied().enumerate() {
            let plane = &proto[channel * plane_len..(channel + 1) * plane_len];
            for (destination, source) in low_res.iter_mut().zip(plane) {
                *destination += coefficient * source;
            }
        }
        // YOLO prototypes describe the 640x640 letterboxed input. Remove that
        // padding before resizing to the source image, then crop in the same
        // original-image coordinate space as the decoded bounding box. This is
        // the ordering used by Ultralytics' process_mask_native/scale_masks.
        let upsampled = scale_mask_to_original(low_res, mask_w as u32, mask_h as u32, letterbox)?;
        let (x0, y0, x1, y1) = clipped_box_bounds(output_width, output_height, detection.bbox_xyxy);
        for y in y0..y1 {
            let row = y * output_width;
            for x in x0..x1 {
                // sigmoid(logit) > 0.5 is equivalent to logit > 0.
                if upsampled[row + x] > 0.0 {
                    merged[row + x] = 255;
                }
            }
        }
    }
    Ok(merged)
}

fn clipped_box_bounds(
    mask_width: usize,
    mask_height: usize,
    bbox_xyxy: [f32; 4],
) -> (usize, usize, usize, usize) {
    (
        bbox_xyxy[0].floor().clamp(0.0, mask_width as f32) as usize,
        bbox_xyxy[1].floor().clamp(0.0, mask_height as f32) as usize,
        bbox_xyxy[2].ceil().clamp(0.0, mask_width as f32) as usize,
        bbox_xyxy[3].ceil().clamp(0.0, mask_height as f32) as usize,
    )
}

fn scale_mask_to_original(
    mask: Vec<f32>,
    width: u32,
    height: u32,
    letterbox: LetterboxInfo,
) -> MaskResult<Vec<f32>> {
    let expected = pixel_count(width, height)?;
    if mask.len() != expected {
        return Err(MaskError::inference("mask dimensions do not match data"));
    }
    let buffer = ImageBuffer::<Luma<f32>, Vec<f32>>::from_vec(width, height, mask)
        .ok_or_else(|| MaskError::inference("failed to build intermediate mask"))?;
    let (left, top, right, bottom) = mask_content_bounds(
        width,
        height,
        letterbox.original_width,
        letterbox.original_height,
    );
    if right <= left || bottom <= top {
        return Err(MaskError::inference("letterbox mask content is empty"));
    }
    let content =
        image::imageops::crop_imm(&buffer, left, top, right - left, bottom - top).to_image();
    Ok(image::imageops::resize(
        &content,
        letterbox.original_width,
        letterbox.original_height,
        FilterType::Triangle,
    )
    .into_raw())
}

fn mask_content_bounds(
    mask_width: u32,
    mask_height: u32,
    original_width: u32,
    original_height: u32,
) -> (u32, u32, u32, u32) {
    let gain = (mask_height as f32 / original_height as f32)
        .min(mask_width as f32 / original_width as f32);
    let pad_width = (mask_width as f32 - (original_width as f32 * gain).round()) / 2.0;
    let pad_height = (mask_height as f32 - (original_height as f32 * gain).round()) / 2.0;
    let left = (pad_width - 0.1).round().clamp(0.0, mask_width as f32) as u32;
    let top = (pad_height - 0.1).round().clamp(0.0, mask_height as f32) as u32;
    let right =
        (mask_width as f32 - (pad_width + 0.1).round()).clamp(0.0, mask_width as f32) as u32;
    let bottom =
        (mask_height as f32 - (pad_height + 0.1).round()).clamp(0.0, mask_height as f32) as u32;
    (left, top, right, bottom)
}

fn scale_box_from_letterbox(bbox: [f32; 4], letterbox: LetterboxInfo) -> [f32; 4] {
    let mut x0 = (bbox[0] - letterbox.pad_x) / letterbox.scale;
    let mut y0 = (bbox[1] - letterbox.pad_y) / letterbox.scale;
    let mut x1 = (bbox[2] - letterbox.pad_x) / letterbox.scale;
    let mut y1 = (bbox[3] - letterbox.pad_y) / letterbox.scale;
    x0 = x0.clamp(0.0, letterbox.original_width as f32);
    y0 = y0.clamp(0.0, letterbox.original_height as f32);
    x1 = x1.clamp(0.0, letterbox.original_width as f32);
    y1 = y1.clamp(0.0, letterbox.original_height as f32);
    [x0, y0, x1, y1]
}

fn non_max_suppression(mut detections: Vec<Detection>) -> MaskResult<Vec<Detection>> {
    detections.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Detection> = Vec::new();
    'candidate: for detection in detections {
        for previous in &kept {
            if detection.class_id == previous.class_id
                && box_iou(detection.bbox_xyxy, previous.bbox_xyxy) > IOU_THRESHOLD
            {
                continue 'candidate;
            }
        }
        kept.push(detection);
        if kept.len() >= MAX_DETECTIONS {
            break;
        }
    }
    Ok(kept)
}

fn box_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    let intersection = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn class_ids_for_targets(targets: &[String]) -> Vec<usize> {
    let normalized: Vec<String> = targets
        .iter()
        .map(|target| target.trim().to_ascii_lowercase())
        .filter(|target| !target.is_empty())
        .collect();
    if normalized.is_empty() {
        return Vec::new();
    }
    COCO_CLASSES
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            normalized
                .iter()
                .any(|target| target == name)
                .then_some(index)
        })
        .collect()
}

fn pixel_count(width: u32, height: u32) -> MaskResult<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| MaskError::invalid_input("image dimensions overflow"))
}

pub(crate) fn session_pool_size_for_provider(provider: &str) -> usize {
    session_pool_size_for_target(provider, cfg!(target_os = "windows"))
}

fn session_pool_size_for_target(provider: &str, is_windows: bool) -> usize {
    if provider.eq_ignore_ascii_case("CUDA")
        || (is_windows && provider.eq_ignore_ascii_case("DirectML"))
    {
        // The Rust Session API requires exclusive access while Run is active.
        // Two replicas allow adjacent images to overlap GPU work without
        // tripling model memory for the three outer pipeline workers. DirectML
        // explicitly permits concurrent Run calls on different sessions; CUDA
        // likewise benefits from independent execution streams/sessions.
        2
    } else {
        // CoreML has a single session by design; keeping other providers at
        // one also avoids surprising memory growth and provider-specific
        // concurrency behavior.
        1
    }
}

pub(crate) fn default_coreml_cache_dir(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let mut model = File::open(model_path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = model.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    // CoreML's own invalidation follows model graph/metadata changes and can
    // reuse a compiled graph when only weights change. A content-addressed
    // subdirectory makes that reuse safe for replaced/custom ONNX files.
    let model_digest = format!("{:x}", hasher.finalize());
    Some(parent.join(".ort-coreml-cache").join(model_digest))
}

pub(crate) fn prepare_coreml_cache_dir(
    provider: &str,
    cache_dir: Option<&Path>,
) -> Option<PathBuf> {
    if !provider.eq_ignore_ascii_case("CoreML") {
        return None;
    }
    let cache_dir = cache_dir?;
    std::fs::create_dir_all(cache_dir).ok()?;
    Some(cache_dir.to_path_buf())
}

pub(crate) fn provider_candidates(requested: Option<&str>) -> Vec<String> {
    if let Some(provider) = requested
        .map(normalize_provider_name)
        .filter(|value| !value.is_empty())
    {
        if provider == "CPU" {
            return Vec::new();
        }
        return vec![provider];
    }

    let mut candidates = Vec::new();
    #[cfg(all(target_os = "windows", feature = "cuda"))]
    candidates.push("CUDA".to_string());
    #[cfg(target_os = "windows")]
    candidates.push("DirectML".to_string());
    #[cfg(target_os = "macos")]
    candidates.push("CoreML".to_string());
    candidates
}

pub(crate) fn normalize_provider_name(provider: &str) -> String {
    match provider.trim().to_ascii_lowercase().as_str() {
        "cuda" => "CUDA",
        "coreml" => "CoreML",
        "directml" => "DirectML",
        "tensorrt" => "TensorRT",
        "nvrtx" => "NVRTX",
        "xnnpack" => "XNNPACK",
        "webgpu" => "WebGPU",
        "cpu" => "CPU",
        other => other,
    }
    .to_string()
}

pub(crate) fn session_builder_for_provider(provider: &str) -> MaskResult<SessionBuilder> {
    let mut builder = Session::builder()
        .map_err(|error| MaskError::inference(format!("create ONNX session: {error}")))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|error| MaskError::inference(error.to_string()))?
        // DirectML does not support memory pattern optimization.  Keeping this
        // disabled is also safe for CPU and avoids global graph-state surprises.
        .with_memory_pattern(!provider.eq_ignore_ascii_case("DirectML"))
        .map_err(|error| MaskError::inference(error.to_string()))?
        .with_parallel_execution(false)
        .map_err(|error| MaskError::inference(error.to_string()))?;
    if !provider.eq_ignore_ascii_case("CPU") {
        builder = builder
            .with_disable_cpu_fallback()
            .map_err(|error| MaskError::inference(format!("disable CPU fallback: {error}")))?;
    }
    Ok(builder)
}

#[allow(unused_variables)]
pub(crate) fn register_execution_provider_with_cache(
    builder: &mut SessionBuilder,
    provider: &str,
    cache_dir: Option<&Path>,
) -> MaskResult<()> {
    let provider = normalize_provider_name(provider);
    match provider.as_str() {
        "CPU" => Ok(()),
        #[cfg(feature = "cuda")]
        "CUDA" => ort::ep::CUDA::default()
            .register(builder)
            .map_err(|error| MaskError::inference(format!("register CUDA: {error}"))),
        #[cfg(any(feature = "coreml", target_os = "macos"))]
        "CoreML" => {
            #[cfg(target_os = "macos")]
            {
                use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
                let mut execution_provider = ort::ep::CoreML::default()
                    .with_static_input_shapes(true)
                    .with_subgraphs(true)
                    .with_compute_units(ComputeUnits::All)
                    .with_model_format(ModelFormat::MLProgram)
                    .with_specialization_strategy(SpecializationStrategy::FastPrediction);
                if let Some(cache_dir) = cache_dir {
                    execution_provider = execution_provider
                        .with_model_cache_dir(cache_dir.to_string_lossy().into_owned());
                }
                return execution_provider
                    .register(builder)
                    .map_err(|error| MaskError::inference(format!("register CoreML: {error}")));
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err(MaskError::inference(
                    "CoreML is only available on macOS targets",
                ));
            }
        }
        #[cfg(any(feature = "directml", target_os = "windows"))]
        "DirectML" => {
            use ort::ep::directml::{DeviceFilter, PerformancePreference};
            return ort::ep::DirectML::default()
                .with_device_filter(DeviceFilter::Gpu)
                .with_performance_preference(PerformancePreference::HighPerformance)
                .register(builder)
                .map_err(|error| MaskError::inference(format!("register DirectML: {error}")));
        }
        #[cfg(feature = "xnnpack")]
        "XNNPACK" => ort::ep::XNNPACK::default()
            .register(builder)
            .map_err(|error| MaskError::inference(format!("register XNNPACK: {error}"))),
        #[cfg(feature = "tensorrt")]
        "TensorRT" => ort::ep::TensorRT::default()
            .register(builder)
            .map_err(|error| MaskError::inference(format!("register TensorRT: {error}"))),
        #[cfg(feature = "nvrtx")]
        "NVRTX" => ort::ep::NVRTX::default()
            .register(builder)
            .map_err(|error| MaskError::inference(format!("register NVRTX: {error}"))),
        #[cfg(feature = "webgpu")]
        "WebGPU" => ort::ep::WebGPU::default()
            .register(builder)
            .map_err(|error| MaskError::inference(format!("register WebGPU: {error}"))),
        _ => Err(MaskError::inference(format!(
            "execution provider {provider} is not enabled in this build"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_horizontal_letterbox_padding() {
        assert_eq!(mask_content_bounds(160, 160, 1280, 640), (0, 40, 160, 120));
    }

    #[test]
    fn removes_vertical_letterbox_padding() {
        assert_eq!(mask_content_bounds(160, 160, 640, 1280), (40, 0, 120, 160));
    }

    #[test]
    fn preserves_square_mask_content() {
        assert_eq!(mask_content_bounds(160, 160, 640, 640), (0, 0, 160, 160));
    }

    #[test]
    fn handles_odd_letterbox_padding_like_ultralytics() {
        assert_eq!(mask_content_bounds(160, 160, 1000, 333), (0, 53, 160, 106));
    }

    #[test]
    fn preprocesses_yolo_input_as_contiguous_chw_planes() {
        let mut pixels = image::RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, image::Rgb([0, 0, 0]));
        pixels.put_pixel(0, 0, image::Rgb([255, 128, 0]));
        pixels.put_pixel(1, 0, image::Rgb([64, 32, 16]));
        pixels.put_pixel(0, 1, image::Rgb([8, 4, 2]));
        let image = DynamicImage::ImageRgb8(pixels);
        let (input, letterbox) = preprocess_yolo(&image).unwrap();

        assert_eq!(letterbox.pad_y, 0.0);
        assert!((input[[0, 0, 0, 0]] - 1.0).abs() < f32::EPSILON);
        assert!((input[[0, 1, 0, 0]] - 128.0 / 255.0).abs() < f32::EPSILON);
        assert_eq!(input[[0, 2, 0, 0]], 0.0);
        assert!((input[[0, 0, 0, 1]] - 64.0 / 255.0).abs() < f32::EPSILON);
        assert!((input[[0, 1, 1, 0]] - 4.0 / 255.0).abs() < f32::EPSILON);
        assert!((input[[0, 2, 1, 0]] - 2.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reuses_rgb_when_yolo_resize_dimensions_match() {
        let mut pixels = image::RgbImage::from_pixel(4, 2, image::Rgb([0, 0, 0]));
        pixels.put_pixel(1, 1, image::Rgb([17, 31, 47]));
        let resized = resize_yolo_source(pixels, 4, 2);

        assert_eq!(resized.dimensions(), (4, 2));
        assert_eq!(resized.get_pixel(1, 1), &image::Rgb([17, 31, 47]));
    }

    #[test]
    fn resizes_rgb_when_yolo_resize_dimensions_differ() {
        let pixels = image::RgbImage::from_pixel(4, 2, image::Rgb([17, 31, 47]));
        let resized = resize_yolo_source(pixels, 2, 1);

        assert_eq!(resized.dimensions(), (2, 1));
    }

    #[test]
    fn directml_session_pool_policy_is_conservative() {
        assert_eq!(session_pool_size_for_target("DirectML", true), 2);
        assert_eq!(session_pool_size_for_target("DirectML", false), 1);
    }

    #[test]
    fn cuda_session_pool_policy_allows_two_in_flight_runs() {
        assert_eq!(session_pool_size_for_target("CUDA", true), 2);
        assert_eq!(session_pool_size_for_target("CUDA", false), 2);
    }

    #[test]
    fn coreml_session_pool_policy_stays_single_session() {
        assert_eq!(session_pool_size_for_target("CoreML", true), 1);
        assert_eq!(session_pool_size_for_target("CoreML", false), 1);
        assert_eq!(session_pool_size_for_provider("CoreML"), 1);
    }

    #[test]
    fn non_coreml_cache_requests_are_ignored() {
        let cache = tempfile::tempdir().unwrap();
        assert!(prepare_coreml_cache_dir("DirectML", Some(cache.path())).is_none());
    }

    #[test]
    fn coreml_cache_directory_is_best_effort() {
        let parent = tempfile::tempdir().unwrap();
        let cache_path = parent.path().join("coreml-cache");
        let prepared = prepare_coreml_cache_dir("CoreML", Some(&cache_path));

        assert_eq!(prepared.as_deref(), Some(cache_path.as_path()));
        assert!(cache_path.is_dir());
    }

    #[test]
    fn coreml_cache_path_is_keyed_by_model_contents() {
        let parent = tempfile::tempdir().unwrap();
        let model_path = parent.path().join("model.onnx");
        std::fs::write(&model_path, b"weights-a").unwrap();
        let first = default_coreml_cache_dir(&model_path).unwrap();

        // Keep the same path and byte length to prove the key is not based on
        // file identity or size alone.
        std::fs::write(&model_path, b"weights-b").unwrap();
        let second = default_coreml_cache_dir(&model_path).unwrap();

        assert_ne!(first, second);
        assert_eq!(first.parent(), second.parent());
        assert_eq!(
            first.parent(),
            Some(parent.path().join(".ort-coreml-cache").as_path())
        );
    }

    #[test]
    fn rejects_out_of_range_target_class_ids() {
        let output = Array4::<f32>::zeros((1, 1, 1, 1)).into_dyn();
        let letterbox = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            original_width: 1,
            original_height: 1,
        };

        let error = decode_detections(output, &[CLASS_COUNT], 0.25, letterbox).unwrap_err();

        assert!(error.to_string().contains("outside the COCO class range"));
    }

    #[test]
    fn merges_masks_only_inside_each_detection_box() {
        let mut proto = Array4::<f32>::zeros((1, MASK_DIM, 2, 2));
        proto[[0, 0, 0, 0]] = 1.0;
        proto[[0, 0, 0, 1]] = 1.0;
        proto[[0, 0, 1, 0]] = 1.0;
        proto[[0, 0, 1, 1]] = 1.0;
        let mut coefficients = [0.0; MASK_DIM];
        coefficients[0] = 1.0;
        let detections = vec![Detection {
            bbox_xyxy: [1.0, 1.0, 3.0, 3.0],
            score: 1.0,
            class_id: 0,
            mask_coeffs: coefficients,
        }];
        let letterbox = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            original_width: 4,
            original_height: 4,
        };

        let merged = decode_merged_mask(proto.into_dyn(), &detections, letterbox).unwrap();

        for y in 0..4 {
            for x in 0..4 {
                let expected = if (1..3).contains(&x) && (1..3).contains(&y) {
                    255
                } else {
                    0
                };
                assert_eq!(merged[y * 4 + x], expected);
            }
        }
    }

    #[test]
    fn explicit_cpu_provider_returns_actionable_error() {
        let paths = ModelPaths {
            yolo: Some(Path::new("unused.onnx").to_path_buf()),
            skyseg: None,
        };
        let error = match YoloSegPipeline::load(&paths, Some("CPU")) {
            Ok(_) => panic!("CPU provider must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("CPU mask inference is disabled"));
    }

    #[test]
    #[ignore = "requires GS360_TEST_YOLO_MODEL and a physical GPU"]
    fn loads_production_model_without_cpu_fallback() {
        let model = std::env::var_os("GS360_TEST_YOLO_MODEL")
            .expect("GS360_TEST_YOLO_MODEL must point to a YOLO11 segmentation model");
        let provider =
            std::env::var("GS360_TEST_GPU_PROVIDER").unwrap_or_else(|_| "CoreML".to_string());
        let pipeline = YoloSegPipeline::load_with_provider(Path::new(&model), &provider)
            .expect("the full graph must load on the selected GPU provider");
        assert_eq!(pipeline.execution_provider, provider);
        if let Some(image_path) = std::env::var_os("GS360_TEST_PERSON_IMAGE") {
            let image = image::open(image_path).expect("test image must decode");
            let mask = pipeline
                .generate_exclusion_mask(&image, &["person".to_string()], 0.25, &CancelToken::new())
                .expect("GPU person segmentation must run");
            assert!(mask.data.iter().any(|pixel| *pixel != 0));
        }
    }
}
