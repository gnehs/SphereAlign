//! U-2-Net/skyseg ONNX inference.

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use ndarray::Array4;
use ort::session::{OutputSelector, RunOptions, SessionInputValue};
use ort::value::Tensor;

use super::inference::{
    default_coreml_cache_dir, load_session_pool, normalize_provider_name, SessionPool,
};
use super::{CancelToken, MaskError, MaskResult, SegmentationMask};
use crate::masking::models::ModelPaths;

const INPUT_SIZE: u32 = 320;
const THRESHOLD: f32 = 0.5;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Sky segmentation model.  The returned mask is an exclusion mask (255 = sky).
#[derive(Clone)]
pub struct SkysegPipeline {
    session: Arc<SessionPool>,
    input_name: String,
    output_name: String,
}

impl SkysegPipeline {
    /// Load skyseg with the same provider selected for YOLO. GPU failures are
    /// surfaced instead of silently moving the model to CPU.
    pub fn load(paths: &ModelPaths, provider: &str) -> MaskResult<Self> {
        let model_path = paths.skyseg.as_ref().ok_or_else(|| {
            MaskError::model("skyseg model is required when sky masking is enabled")
        })?;
        Self::load_with_provider(model_path, provider)
    }

    pub fn load_with_provider(model_path: &Path, provider: &str) -> MaskResult<Self> {
        let provider = normalize_provider_name(provider);
        let cache_dir = (provider == "CoreML")
            .then(|| default_coreml_cache_dir(model_path))
            .flatten();
        Self::load_with_cache(model_path, &provider, cache_dir.as_deref())
    }

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
        let (input_name, output_name) = session.with_session(|session| {
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
            Ok((input_name, output_name))
        })?;
        Ok(Self {
            session: Arc::new(session),
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
        let input = preprocess_skyseg(image)?;
        let run_options = RunOptions::new()
            .map_err(|error| MaskError::inference(error.to_string()))?
            .with_outputs(OutputSelector::no_default().with(self.output_name.clone()));
        let output_value = self.session.with_session(|session| {
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
                .map_err(|error| MaskError::inference(format!("skyseg inference: {error}")))?;
            let mut outputs = outputs;
            outputs
                .remove(self.output_name.as_str())
                .ok_or_else(|| MaskError::inference("skyseg output was not returned"))
        })?;
        let raw_output = output_value
            .try_extract_array::<f32>()
            .map_err(|error| MaskError::inference(error.to_string()))?
            .to_owned();
        if cancel.is_cancelled() {
            return Err(MaskError::Cancelled);
        }

        let values: Cow<'_, [f32]> = match raw_output.as_slice() {
            Some(values) => Cow::Borrowed(values),
            None => Cow::Owned(raw_output.iter().copied().collect()),
        };
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

fn preprocess_skyseg(image: &DynamicImage) -> MaskResult<Array4<f32>> {
    let rgb = image
        .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
        .to_rgb8();
    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    let plane_len = (INPUT_SIZE * INPUT_SIZE) as usize;
    let input_data = input
        .as_slice_mut()
        .ok_or_else(|| MaskError::inference("skyseg input tensor is not contiguous"))?;
    for (index, pixel) in rgb.pixels().enumerate() {
        input_data[index] = (pixel[0] as f32 / 255.0 - MEAN[0]) / STD[0];
        input_data[plane_len + index] = (pixel[1] as f32 / 255.0 - MEAN[1]) / STD[1];
        input_data[plane_len * 2 + index] = (pixel[2] as f32 / 255.0 - MEAN[2]) / STD[2];
    }
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocesses_skyseg_input_as_contiguous_normalized_chw_planes() {
        let mut pixels = image::RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, image::Rgb([0, 0, 0]));
        pixels.put_pixel(0, 0, image::Rgb([255, 128, 0]));
        pixels.put_pixel(1, 0, image::Rgb([64, 32, 16]));
        pixels.put_pixel(0, 1, image::Rgb([8, 4, 2]));
        let image = DynamicImage::ImageRgb8(pixels);
        let input = preprocess_skyseg(&image).unwrap();

        assert!((input[[0, 0, 0, 0]] - (1.0 - MEAN[0]) / STD[0]).abs() < f32::EPSILON);
        assert!((input[[0, 1, 0, 0]] - (128.0 / 255.0 - MEAN[1]) / STD[1]).abs() < f32::EPSILON);
        assert!((input[[0, 2, 0, 0]] - (0.0 - MEAN[2]) / STD[2]).abs() < f32::EPSILON);
        assert!((input[[0, 0, 0, 1]] - (64.0 / 255.0 - MEAN[0]) / STD[0]).abs() < f32::EPSILON);
        assert!((input[[0, 1, 1, 0]] - (4.0 / 255.0 - MEAN[1]) / STD[1]).abs() < f32::EPSILON);
        assert!((input[[0, 2, 1, 0]] - (2.0 / 255.0 - MEAN[2]) / STD[2]).abs() < f32::EPSILON);
    }

    #[test]
    #[ignore = "requires GS360_TEST_SKYSEG_MODEL and a physical GPU"]
    fn loads_production_model_without_cpu_fallback() {
        let model = std::env::var_os("GS360_TEST_SKYSEG_MODEL")
            .expect("GS360_TEST_SKYSEG_MODEL must point to the SkySeg ONNX model");
        let provider =
            std::env::var("GS360_TEST_GPU_PROVIDER").unwrap_or_else(|_| "CoreML".to_string());
        SkysegPipeline::load_with_provider(Path::new(&model), &provider)
            .expect("the full graph must load on the selected GPU provider");
    }
}
