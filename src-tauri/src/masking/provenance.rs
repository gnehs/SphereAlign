//! A decoded PNG alone does not identify the settings or source that made it.
//! Each output carries a content-addressed receipt committed after the PNG.
use super::{MaskError, MaskRequest, MaskResult};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

fn hash_file(path: &Path) -> MaskResult<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 { break; }
        hash.update(&buffer[..size]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn request_signature(request: &MaskRequest) -> MaskResult<String> {
    let model_hash = |path: &Option<PathBuf>| -> MaskResult<Option<String>> {
        path.as_deref().map(hash_file).transpose()
    };
    let mut classes = request.classes.clone(); classes.sort(); classes.dedup();
    let config = json!({
        "revision": 2, "classes": classes, "maskSky": request.mask_sky,
        "confidence": request.confidence, "rotations": request.rotations,
        "dilationInferencePixels": request.dilation, "nativeWorkingLongEdge": super::MASK_WORKING_LONG_EDGE,
        "tileSize": super::tiles::TILE_SIZE, "tileFovDegrees": super::tiles::TILE_FOV_DEGREES,
        "tileViewsDegrees": [0, -60, 60, -60, 60], "pixelConvention": "colmap-corner-centers-at-0.5",
        "validRadiusRatio": request.valid_radius_ratio, "opticalOcclusions": request.optical_occlusions,
        "calibratedCameras": request.calibrated_cameras,
        "yoloSha256": model_hash(&request.yolo_model)?, "skysegSha256": model_hash(&request.skyseg_model)?,
        "executionProvider": request.execution_provider,
    });
    let bytes = serde_json::to_vec(&config).map_err(|e| MaskError::invalid_input(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn additional_path(request: &MaskRequest, input: &Path) -> Option<PathBuf> {
    let root = request.additional_masks_dir.as_ref()?;
    let relative = if request.images_dir.is_file() { PathBuf::from(input.file_name()?) }
        else { input.strip_prefix(&request.images_dir).ok()?.to_path_buf() };
    let path = root.join(relative.with_extension("png"));
    path.is_file().then_some(path)
}

pub(super) fn identity(request: &MaskRequest, input: &Path, signature: &str) -> MaskResult<Value> {
    Ok(json!({"requestSha256": signature, "imageSha256": hash_file(input)?,
        "additionalMaskSha256": additional_path(request, input).as_deref().map(hash_file).transpose()?}))
}

fn receipt_path(mask: &Path) -> PathBuf { mask.with_extension("png.receipt.json") }

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskReviewSummary {
    pub image_count: usize,
    pub heavily_masked_count: usize,
    pub nearly_empty_count: usize,
    pub examples: Vec<Value>,
}

/// These are review candidates, not labels of false positives. A large nearby
/// person can legitimately occupy most of a view; inspect static geometry.
pub fn review_summary(request: &MaskRequest) -> MaskResult<MaskReviewSummary> {
    let mut report = MaskReviewSummary { image_count: 0, heavily_masked_count: 0, nearly_empty_count: 0, examples: Vec::new() };
    for input in super::collect_images(&request.images_dir)? {
        let path = super::output_path(request, &input)?;
        let value: Value = serde_json::from_slice(&fs::read(receipt_path(&path))?).map_err(|e| MaskError::invalid_input(e.to_string()))?;
        let total = value.get("pixelCount").and_then(Value::as_u64).filter(|v| *v > 0).ok_or_else(|| MaskError::invalid_input("missing mask pixel count"))?;
        let excluded = value.get("excludedPixels").and_then(Value::as_u64).filter(|v| *v <= total).ok_or_else(|| MaskError::invalid_input("invalid excluded mask pixel count"))?;
        let ratio = excluded as f64 / total as f64;
        report.image_count += 1;
        report.heavily_masked_count += usize::from(ratio >= 0.60);
        report.nearly_empty_count += usize::from(ratio >= 0.90);
        if ratio >= 0.60 && report.examples.len() < 40 {
            report.examples.push(json!({"image": input.strip_prefix(&request.images_dir).unwrap_or(&input).to_string_lossy().replace('\\', "/"), "excludedRatio": ratio}));
        }
    }
    Ok(report)
}

pub(super) fn can_reuse(request: &MaskRequest, input: &Path, mask: &Path, signature: &str) -> bool {
    let check = || -> MaskResult<bool> {
        let receipt: Value = serde_json::from_slice(&fs::read(receipt_path(mask))?)
            .map_err(|e| MaskError::invalid_input(e.to_string()))?;
        Ok(receipt.get("identity") == Some(&identity(request, input, signature)?)
            && receipt.get("maskSha256").and_then(Value::as_str) == Some(hash_file(mask)?.as_str()))
    };
    check().unwrap_or(false)
}

pub(super) fn commit(request: &MaskRequest, input: &Path, mask: &Path, signature: &str, expected: &Value, keep: &[u8]) -> MaskResult<()> {
    if &identity(request, input, signature)? != expected {
        return Err(MaskError::invalid_input("input image or additional mask changed during inference; rerun this image"));
    }
    let output = receipt_path(mask);
    let temp = output.with_extension(format!("{}.part", std::process::id()));
    let receipt = json!({"schemaVersion": 1, "identity": expected,
        "maskSha256": hash_file(mask)?, "pixelCount": keep.len(),
        "excludedPixels": keep.iter().filter(|v| **v == 0).count(),
        "projection": if request.calibrated_cameras.is_empty() { "native" } else { "calibrated-tiles" },
        "confidence": request.confidence, "rotations": request.rotations,
        "dilationInferencePixels": request.dilation, "blackMeans": "exclude"});
    fs::write(&temp, serde_json::to_vec_pretty(&receipt).map_err(|e| MaskError::invalid_input(e.to_string()))?)?;
    super::rename_replace(&temp, &output)
}

pub(super) fn merge_additional_mask(request: &MaskRequest, input: &Path, width: u32, height: u32, keep: &mut [u8]) -> MaskResult<()> {
    let Some(path) = additional_path(request, input) else { return Ok(()); };
    if !super::is_valid_mask_file(&path, width, height) {
        return Err(MaskError::invalid_input(format!("additional mask must be binary L8 PNG at native dimensions: {}", path.display())));
    }
    let additional = image::open(&path)?.into_luma8();
    for (pixel, extra) in keep.iter_mut().zip(additional.into_raw()) {
        if extra == 0 { *pixel = 0; }
    }
    Ok(())
}
