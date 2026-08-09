//! YOLO11 segmentation inference used by the native-fisheye masker.
//!
//! This is intentionally kept independent from the old equirectangular pipeline:
//! native lens frames are loaded and decoded at their original aspect ratio, then
//! letterboxed for YOLO and projected back to the original pixel grid.

use std::path::Path;
use std::sync::{Arc, Mutex};

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
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

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

/// YOLO11 ONNX segmentation pipeline.
#[derive(Clone)]
pub struct YoloSegPipeline {
    session: Arc<Mutex<Session>>,
    input_name: String,
    output0_name: String,
    output1_name: String,
    pub execution_provider: String,
}

impl YoloSegPipeline {
    /// Load YOLO11 from resolved model paths.  Provider loading is attempted in
    /// the requested/platform order and always falls back to CPU with a useful
    /// error if the CPU session cannot be created.
    pub fn load(paths: &ModelPaths, requested_provider: Option<&str>) -> MaskResult<Self> {
        let candidates = provider_candidates(requested_provider);
        let mut errors = Vec::new();
        for provider in candidates {
            match Self::load_with_provider(&paths.yolo, &provider) {
                Ok(pipeline) => return Ok(pipeline),
                Err(error) => errors.push(format!("{provider}: {error}")),
            }
        }
        Err(MaskError::model(format!(
            "unable to load YOLO11 segmentation model {} ({})",
            paths.yolo.display(),
            errors.join("; ")
        )))
    }

    /// Load one provider explicitly.  The feature gates mirror the provider
    /// names used by the gs360masker build.  Unsupported providers return an
    /// actionable error and are subsequently handled by [`Self::load`].
    pub fn load_with_provider(model_path: &Path, provider: &str) -> MaskResult<Self> {
        let mut builder = session_builder_for_provider(provider)?;
        register_execution_provider(&mut builder, provider)?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|error| MaskError::inference(format!("{provider} session: {error}")))?;
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

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            input_name,
            output0_name,
            output1_name,
            execution_provider: provider.to_string(),
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

        let masks = decode_masks(proto, &detections, letterbox)?;
        let mut merged = vec![0u8; pixel_count(width, height)?];
        for mask in masks {
            for (dst, src) in merged.iter_mut().zip(mask) {
                if src != 0 {
                    *dst = 255;
                }
            }
        }
        SegmentationMask::new(width, height, merged)
    }

    fn run_raw(
        &self,
        image: &DynamicImage,
    ) -> MaskResult<(ArrayD<f32>, ArrayD<f32>, LetterboxInfo)> {
        let (input, letterbox) = preprocess_yolo(image)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| MaskError::inference("YOLO11 session lock poisoned"))?;
        let outputs = session
            .run(vec![(
                self.input_name.clone(),
                SessionInputValue::from(
                    Tensor::from_array(input)
                        .map_err(|error| MaskError::inference(error.to_string()))?,
                ),
            )])
            .map_err(|error| MaskError::inference(format!("YOLO11 inference: {error}")))?;
        let output0 = outputs[self.output0_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|error| MaskError::inference(error.to_string()))?
            .to_owned();
        let output1 = outputs[self.output1_name.as_str()]
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
    let resized =
        image::imageops::resize(&rgb, resized_width, resized_height, FilterType::Triangle);
    let mut canvas =
        image::RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, image::Rgb([114, 114, 114]));
    let pad_x = ((INPUT_SIZE - resized_width) / 2) as i64;
    let pad_y = ((INPUT_SIZE - resized_height) / 2) as i64;
    image::imageops::replace(&mut canvas, &resized, pad_x, pad_y);

    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (x, y, pixel) in canvas.enumerate_pixels() {
        input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
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

fn decode_detections(
    output: ArrayD<f32>,
    target_class_ids: &[usize],
    confidence: f32,
    letterbox: LetterboxInfo,
) -> MaskResult<Vec<Detection>> {
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
        for class_id in 0..CLASS_COUNT {
            if !target_class_ids.is_empty() && !target_class_ids.contains(&class_id) {
                continue;
            }
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

fn decode_masks(
    proto: ArrayD<f32>,
    detections: &[Detection],
    letterbox: LetterboxInfo,
) -> MaskResult<Vec<Vec<u8>>> {
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
    let mut masks = Vec::with_capacity(detections.len());
    for detection in detections {
        let mut low_res = vec![0.0f32; mask_w * mask_h];
        for y in 0..mask_h {
            for x in 0..mask_w {
                let mut value = 0.0;
                for channel in 0..MASK_DIM {
                    value += detection.mask_coeffs[channel] * proto[[channel, y, x]];
                }
                // sigmoid(value) > .5 is equivalent to value > 0 and avoids an
                // unnecessary exp() for every prototype pixel.
                low_res[y * mask_w + x] = if value > 0.0 { 1.0 } else { 0.0 };
            }
        }
        crop_mask_to_box(
            &mut low_res,
            mask_w,
            mask_h,
            detection.bbox_xyxy,
            letterbox.original_width,
            letterbox.original_height,
        );
        let upsampled = resize_mask(
            &low_res,
            mask_w as u32,
            mask_h as u32,
            letterbox.original_width,
            letterbox.original_height,
        )?;
        masks.push(
            upsampled
                .into_iter()
                .map(|value| if value >= 0.5 { 255 } else { 0 })
                .collect(),
        );
    }
    Ok(masks)
}

fn crop_mask_to_box(
    mask: &mut [f32],
    mask_width: usize,
    mask_height: usize,
    bbox_xyxy: [f32; 4],
    image_width: u32,
    image_height: u32,
) {
    let x0 = (bbox_xyxy[0] * mask_width as f32 / image_width as f32)
        .floor()
        .clamp(0.0, mask_width as f32) as usize;
    let y0 = (bbox_xyxy[1] * mask_height as f32 / image_height as f32)
        .floor()
        .clamp(0.0, mask_height as f32) as usize;
    let x1 = (bbox_xyxy[2] * mask_width as f32 / image_width as f32)
        .ceil()
        .clamp(0.0, mask_width as f32) as usize;
    let y1 = (bbox_xyxy[3] * mask_height as f32 / image_height as f32)
        .ceil()
        .clamp(0.0, mask_height as f32) as usize;
    for y in 0..mask_height {
        for x in 0..mask_width {
            if x < x0 || x >= x1 || y < y0 || y >= y1 {
                mask[y * mask_width + x] = 0.0;
            }
        }
    }
}

fn resize_mask(
    mask: &[f32],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> MaskResult<Vec<f32>> {
    let expected = pixel_count(width, height)?;
    if mask.len() != expected {
        return Err(MaskError::inference("mask dimensions do not match data"));
    }
    let buffer = ImageBuffer::<Luma<f32>, Vec<f32>>::from_vec(width, height, mask.to_vec())
        .ok_or_else(|| MaskError::inference("failed to build intermediate mask"))?;
    Ok(
        image::imageops::resize(&buffer, target_width, target_height, FilterType::Triangle)
            .into_raw(),
    )
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

fn provider_candidates(requested: Option<&str>) -> Vec<String> {
    if let Some(provider) = requested
        .map(normalize_provider_name)
        .filter(|value| !value.is_empty())
    {
        let mut candidates = vec![provider];
        if !candidates.iter().any(|value| value == "CPU") {
            candidates.push("CPU".to_string());
        }
        return candidates;
    }

    let mut candidates = Vec::new();
    #[cfg(all(target_os = "windows", feature = "cuda"))]
    candidates.push("CUDA".to_string());
    #[cfg(target_os = "windows")]
    candidates.push("DirectML".to_string());
    #[cfg(target_os = "macos")]
    candidates.push("CoreML".to_string());
    candidates.push("CPU".to_string());
    candidates
}

fn normalize_provider_name(provider: &str) -> String {
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
    let builder = Session::builder()
        .map_err(|error| MaskError::inference(format!("create ONNX session: {error}")))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|error| MaskError::inference(error.to_string()))?
        // DirectML does not support memory pattern optimization.  Keeping this
        // disabled is also safe for CPU and avoids global graph-state surprises.
        .with_memory_pattern(!provider.eq_ignore_ascii_case("DirectML"))
        .map_err(|error| MaskError::inference(error.to_string()))?
        .with_parallel_execution(false)
        .map_err(|error| MaskError::inference(error.to_string()))?;
    Ok(builder)
}

#[allow(unused_variables)]
pub(crate) fn register_execution_provider(
    builder: &mut SessionBuilder,
    provider: &str,
) -> MaskResult<()> {
    match provider {
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
                return ort::ep::CoreML::default()
                    .with_static_input_shapes(true)
                    .with_subgraphs(true)
                    .with_compute_units(ComputeUnits::All)
                    .with_model_format(ModelFormat::MLProgram)
                    .with_specialization_strategy(SpecializationStrategy::FastPrediction)
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
        "DirectML" => ort::ep::DirectML::default()
            .register(builder)
            .map_err(|error| MaskError::inference(format!("register DirectML: {error}"))),
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
