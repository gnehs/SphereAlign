//! Model path discovery for the native-fisheye mask pipeline.
//!
//! The desktop application is intentionally not coupled to a downloader.  A model
//! can be installed by the backend/bootstrapper and this module only needs to
//! resolve the well-known layouts used by gs360masker and the Ultralytics release.

use std::env;
use std::path::{Path, PathBuf};

use super::{MaskError, MaskResult};

/// The repository name used by the Ultralytics YOLO11 segmentation export.
pub const YOLO11_SEG_REPO: &str = "ultralytics/yolo11s-seg-onnx";
/// The repository name used by the U-2-Net sky segmentation export.
pub const SKYSEG_REPO: &str = "JianyuanWang/skyseg";

/// Resolved ONNX model files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    /// YOLO11 segmentation model (`[1,116,8400]` + prototype output).
    pub yolo: PathBuf,
    /// Optional sky segmentation model.  It is required when `mask_sky` is true.
    pub skyseg: Option<PathBuf>,
}

impl ModelPaths {
    /// Discover model files from an explicit model directory and conventional
    /// cache locations.  Explicit paths always win; no network access occurs here.
    pub fn discover(
        model_dir: Option<&Path>,
        yolo_model: Option<&Path>,
        skyseg_model: Option<&Path>,
        require_skyseg: bool,
    ) -> MaskResult<Self> {
        let roots = model_roots(model_dir);
        let yolo = match yolo_model {
            Some(path) => validate_model_path(path, "YOLO11 segmentation")?,
            None => find_model(&roots, &YOLO_CANDIDATES).ok_or_else(|| {
                MaskError::model(format!(
                    "YOLO11 segmentation model ({YOLO11_SEG_REPO}) not found; searched {}",
                    format_roots(&roots)
                ))
            })?,
        };

        let skyseg = match skyseg_model {
            Some(path) => Some(validate_model_path(path, "skyseg")?),
            None => find_model(&roots, &SKYSEG_CANDIDATES),
        };

        if require_skyseg && skyseg.is_none() {
            return Err(MaskError::model(format!(
                "skyseg model ({SKYSEG_REPO}) not found; searched {}",
                format_roots(&roots)
            )));
        }

        Ok(Self { yolo, skyseg })
    }
}

// Keep these lists deliberately broad.  Older gs360masker builds cache under a
// Hugging Face style repository path while the studio bootstrapper may install
// a flat `models/` directory.
const YOLO_CANDIDATES: &[&str] = &[
    "ultralytics/yolo11s-seg-onnx/onnx/model.onnx",
    "ultralytics/yolo11s-seg-onnx/yolo11s-seg.onnx",
    "yolo11s-seg-onnx/onnx/model.onnx",
    "yolo11s-seg.onnx",
    "yolo11n-seg.onnx",
    "yolo11m-seg.onnx",
];

const SKYSEG_CANDIDATES: &[&str] = &[
    "JianyuanWang/skyseg/skyseg.onnx",
    "skyseg/skyseg.onnx",
    "skyseg.onnx",
];

fn model_roots(explicit: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(path) = explicit {
        roots.push(path.to_path_buf());
    }
    if let Some(path) = env::var_os("GS360_MODEL_DIR") {
        let path = PathBuf::from(path);
        if !roots.iter().any(|root| root == &path) {
            roots.push(path);
        }
    }
    // A relative cache is useful for portable studio bundles and tests.  Do not
    // use a broad home-directory scan: model discovery must remain deterministic.
    for path in [PathBuf::from("models"), PathBuf::from(".models")] {
        if !roots.iter().any(|root| root == &path) {
            roots.push(path);
        }
    }
    roots
}

fn find_model(roots: &[PathBuf], candidates: &[&str]) -> Option<PathBuf> {
    for root in roots {
        for relative in candidates {
            let path = root.join(relative);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn validate_model_path(path: &Path, label: &str) -> MaskResult<PathBuf> {
    if !path.is_file() {
        return Err(MaskError::model(format!(
            "{label} model path is not a file: {}",
            path.display()
        )));
    }
    Ok(path.to_path_buf())
}

fn format_roots(roots: &[PathBuf]) -> String {
    if roots.is_empty() {
        "(no model roots)".to_string()
    } else {
        roots
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
