//! U-2-Net/skyseg ONNX inference.

use std::path::Path;
use std::sync::{Arc, Mutex};

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use ndarray::Array4;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use super::inference::{register_execution_provider, session_builder_for_provider};
use super::{CancelToken, MaskError, MaskResult, SegmentationMask};
use crate::masking::models::ModelPaths;

const INPUT_SIZE: u32 = 320;
const THRESHOLD: f32 = 0.5;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Sky segmentation model.  The returned mask is an exclusion mask (255 = sky).
#[derive(Clone)]
pub struct SkysegPipeline {
    session: Arc<Mutex<Session>>,
    input_name: String,
    output_name: String,
}

impl SkysegPipeline {
    /// Load skyseg with the same provider selected for YOLO.  A failed GPU
    /// session falls back to CPU so enabling sky masking remains deterministic.
    pub fn load(paths: &ModelPaths, provider: &str) -> MaskResult<Self> {
        let model_path = paths.skyseg.as_ref().ok_or_else(|| {
            MaskError::model("skyseg model is required when sky masking is enabled")
        })?;
        match Self::load_with_provider(model_path, provider) {
            Ok(pipeline) => Ok(pipeline),
            Err(error) if !provider.eq_ignore_ascii_case("CPU") => {
                Self::load_with_provider(model_path, "CPU").map_err(|cpu_error| {
                    MaskError::model(format!(
                        "skyseg failed with {provider} ({error}) and CPU fallback ({cpu_error})"
                    ))
                })
            }
            Err(error) => Err(error),
        }
    }

    pub fn load_with_provider(model_path: &Path, provider: &str) -> MaskResult<Self> {
        let mut builder = session_builder_for_provider(provider)?;
        register_execution_provider(&mut builder, provider)?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|error| MaskError::inference(format!("skyseg session: {error}")))?;
        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .ok_or_else(|| MaskError::model("skyseg model has no input"))?;
        let output_name = session
            .outputs()
            .first()
            .map(|output| output.name().to_string())
            .ok_or_else(|| MaskError::model("skyseg model has no output"))?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            input_name,
            output_name,
        })
    }

    /// Generate a 255=sky exclusion mask at the source image dimensions.
    pub fn generate_exclusion_mask(
        &self,
        image: &DynamicImage,
        cancel: &CancelToken,
    ) -> MaskResult<SegmentationMask> {
        if cancel.is_cancelled() {
            return Err(MaskError::Cancelled);
        }
        let rgb = image
            .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
            .to_rgb8();
        let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let values = [pixel[0], pixel[1], pixel[2]];
            for channel in 0..3 {
                input[[0, channel, y as usize, x as usize]] =
                    (values[channel] as f32 / 255.0 - MEAN[channel]) / STD[channel];
            }
        }
        let raw_output = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| MaskError::inference("skyseg session lock poisoned"))?;
            let outputs = session
                .run(vec![(
                    self.input_name.clone(),
                    SessionInputValue::from(
                        Tensor::from_array(input)
                            .map_err(|error| MaskError::inference(error.to_string()))?,
                    ),
                )])
                .map_err(|error| MaskError::inference(format!("skyseg inference: {error}")))?;
            outputs[self.output_name.as_str()]
                .try_extract_array::<f32>()
                .map_err(|error| MaskError::inference(error.to_string()))?
                .to_owned()
        };
        if cancel.is_cancelled() {
            return Err(MaskError::Cancelled);
        }

        let values: Vec<f32> = raw_output.iter().copied().collect();
        let expected = (INPUT_SIZE * INPUT_SIZE) as usize;
        if values.len() < expected {
            return Err(MaskError::inference(format!(
                "skyseg output too small: got {}, expected at least {expected}",
                values.len()
            )));
        }
        let (min, max) = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                (min.min(*value), max.max(*value))
            });
        let range = (max - min).max(f32::EPSILON);
        let normalized = values[values.len() - expected..]
            .iter()
            .map(|value| {
                if value.is_finite() {
                    ((*value - min) / range).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let (width, height) = image.dimensions();
        let resized =
            ImageBuffer::<Luma<f32>, Vec<f32>>::from_vec(INPUT_SIZE, INPUT_SIZE, normalized)
                .ok_or_else(|| MaskError::inference("failed to build skyseg output"))?;
        let resized = image::imageops::resize(&resized, width, height, FilterType::Triangle);
        let data = resized
            .into_raw()
            .into_iter()
            .map(|value| if value >= THRESHOLD { 255 } else { 0 })
            .collect();
        SegmentationMask::new(width, height, data)
    }
}
