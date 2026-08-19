//! Camera-specific color-profile detection and safe Log LUT resolution.
//!
//! The camera's transfer curve is not reliably inferable from image appearance
//! alone.  Detection therefore gives precedence to explicit container/stream
//! metadata and treats a DJI filename hint as weak evidence only. Extraction
//! remains fail-safe: an `unknown` profile, an ambiguous camera model, or a
//! mismatched Log curve never receives a LUT in `auto` mode.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::doctor::find_executable;
use crate::process::silent_command;

pub const DJI_DLOG_M_LUT_URL: &str = "https://terra-1-g.djicdn.com/851d20f7b9f64838a34cd02351370894/OQ101%20LUT/DJI%20Osmo%20360%20D-Log%20M%20to%20Rec.709%20V1.cube";
pub const DJI_DLOG_M_LUT_SHA256: &str =
    "b18162854ab47702068410c33afa98a8cb6eef159fc5a04ce0e65fad0fd8947e";
pub const DJI_DLOG_M_LUT_SIZE: u64 = 1_042_315;
pub const DJI_DLOG_M_LUT_FILE_NAME: &str = "dji-osmo-360-d-log-m-to-rec709-v1.cube";
pub const INSTA360_LUT_ARCHIVE_URL: &str =
    "https://file.insta360.com/static/25781783d5bca22fc519007723fe2ab1/Insta360-LUT.zip";
pub const INSTA360_LUT_ARCHIVE_SHA256: &str =
    "8c703d331012566bde233858305640a2ad2cf05dedbbc7f8950446f7aa3548bb";
pub const INSTA360_LUT_ARCHIVE_SIZE: u64 = 86_262_905;
pub const INSTA360_X5_ILOG_LUT_ENTRY: &str =
    "Insta360-LUT/Insta360 X5 LUT/X5_I-Log_To_Rec.709_V1.0.cube";
pub const INSTA360_X5_ILOG_LUT_SHA256: &str =
    "edba4055614bb2c3e8fa66435998aa18fc9ed8da4dbf2de23ce57038670f2e56";
pub const INSTA360_X5_ILOG_LUT_SIZE: u64 = 4_792_092;
pub const INSTA360_X5_ILOG_LUT_FILE_NAME: &str = "insta360-x5-i-log-to-rec709-v1.cube";

#[derive(Debug, Clone, Copy)]
struct Insta360ArchiveLutSpec {
    id: &'static str,
    display_name: &'static str,
    file_name: &'static str,
    entry: &'static str,
    size: u64,
    sha256: &'static str,
    input_profile: ColorProfile,
}

const INSTA360_ARCHIVE_LUTS: &[Insta360ArchiveLutSpec] = &[
    Insta360ArchiveLutSpec {
        id: "insta360-x5-ilog-rec709-v1",
        display_name: "Insta360 X5 I-Log to Rec.709 V1.0",
        file_name: INSTA360_X5_ILOG_LUT_FILE_NAME,
        entry: INSTA360_X5_ILOG_LUT_ENTRY,
        size: INSTA360_X5_ILOG_LUT_SIZE,
        sha256: INSTA360_X5_ILOG_LUT_SHA256,
        input_profile: ColorProfile::Ilog,
    },
    Insta360ArchiveLutSpec {
        id: "insta360-one-x-log-rec709-v1",
        display_name: "Insta360 ONE X LOG LUT V1.0.0",
        file_name: "insta360-one-x-log-lut-v1.cube",
        entry: "Insta360-LUT/ONE-X-LUT/ONE-X-LUT-Final-V1.0.0.cube",
        size: 10_223_660,
        sha256: "bab317aa511787c484b9e721955c43747ef4a6f6ad8ff8f77cd8fcaf76d9f098",
        input_profile: ColorProfile::InstaLog,
    },
    Insta360ArchiveLutSpec {
        id: "insta360-one-x2-log-rec709-v1",
        display_name: "Insta360 ONE X2 LOG LUT V1.0.0",
        file_name: "insta360-one-x2-log-lut-v1.cube",
        entry: "Insta360-LUT/ONE-X2-LUT/ONE-X2-LUT-Final-V1.0.0.cube",
        size: 7_415_049,
        sha256: "1e540e964333eb984481b95fffb3178dd08abc4c210644ca810d21a535c62b3e",
        input_profile: ColorProfile::InstaLog,
    },
    Insta360ArchiveLutSpec {
        id: "insta360-x3-log-rec709-v1",
        display_name: "Insta360 X3 LOG LUT V1.0.0",
        file_name: "insta360-x3-log-lut-v1.cube",
        entry: "Insta360-LUT/X3-LUT/X3-LUT-V1.0.0.cube",
        size: 7_415_049,
        sha256: "1e540e964333eb984481b95fffb3178dd08abc4c210644ca810d21a535c62b3e",
        input_profile: ColorProfile::InstaLog,
    },
    Insta360ArchiveLutSpec {
        id: "insta360-one-r-dual-lens-360-log-rec709-v1",
        display_name: "Insta360 ONE R Dual-Lens 360 LOG LUT",
        file_name: "insta360-one-r-dual-lens-360-log-lut.cube",
        entry: "Insta360-LUT/ONE-R-LUT/DUAL-LENS-360.CUBE",
        size: 6_409_805,
        sha256: "d8b3c80d53ff850dd23f546f1b6f8f8ccf4d302f7d4a471f5b06fc60bc926e89",
        input_profile: ColorProfile::InstaLog,
    },
    Insta360ArchiveLutSpec {
        id: "insta360-one-rs-dual-lens-360-log-rec709-v1",
        display_name: "Insta360 ONE RS Dual-Lens 360 LOG LUT (65-grid)",
        file_name: "insta360-one-rs-dual-lens-360-log-lut-65.cube",
        entry: "Insta360-LUT/ONE-RS-LUT/DUAL-LENS-360_65.CUBE",
        size: 6_409_867,
        sha256: "09329898ac27ac3a1a9edcae57386725e751643e781c1c608bfed153b4e923f1",
        input_profile: ColorProfile::InstaLog,
    },
    Insta360ArchiveLutSpec {
        id: "insta360-sphere-log-rec709-v1",
        display_name: "Insta360 Sphere Dual-Lens 360 LOG LUT (65-grid)",
        file_name: "insta360-sphere-dual-lens-360-log-lut-65.cube",
        entry: "Insta360-LUT/Sphere-LUT/DUAL-LENS-360_65.CUBE",
        size: 6_409_867,
        sha256: "09329898ac27ac3a1a9edcae57386725e751643e781c1c608bfed153b4e923f1",
        input_profile: ColorProfile::InstaLog,
    },
];
const MAX_CUSTOM_LUT_SIZE: u64 = 64 * 1024 * 1024;
const MAX_OFFICIAL_ARCHIVE_SIZE: u64 = 96 * 1024 * 1024;
const AUTO_APPLY_CONFIDENCE: f64 = 0.80;

/// The externally visible profile names are intentionally stable because they
/// are persisted in source inspection, capture metadata, and checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorProfile {
    #[serde(rename = "dlogM")]
    DlogM,
    #[serde(rename = "iLog")]
    Ilog,
    #[serde(rename = "instaLog")]
    InstaLog,
    #[serde(rename = "rec709")]
    Rec709,
    #[serde(rename = "unknown")]
    Unknown,
}

impl ColorProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DlogM => "dlogM",
            Self::Ilog => "iLog",
            Self::InstaLog => "instaLog",
            Self::Rec709 => "rec709",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LutRecommendation {
    pub id: String,
    pub display_name: String,
    pub file_name: String,
    pub source_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColorDetection {
    pub detected_profile: ColorProfile,
    pub confidence: f64,
    pub reasons: Vec<String>,
    pub should_apply: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommended_lut: Option<LutRecommendation>,
}

impl ColorDetection {
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            detected_profile: ColorProfile::Unknown,
            confidence: 0.0,
            reasons: vec![reason.into()],
            should_apply: false,
            camera_model: None,
            recommended_lut: None,
        }
    }
}

/// User-facing extraction setting. `auto` is deliberately the default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ColorMode {
    #[default]
    Auto,
    LogRec709,
    DlogMRec709,
    Native,
}

impl ColorMode {
    pub fn parse(value: Option<&Value>) -> Result<Self, String> {
        let value = value
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("auto");
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "logrec709" | "log-rec709" | "builtin" => Ok(Self::LogRec709),
            "dlogmrec709" | "dlog-m-rec709" | "dlog_m_rec709" => Ok(Self::DlogMRec709),
            "native" => Ok(Self::Native),
            _ => Err(format!(
                "extract.colorMode must be auto, logRec709, dlogMRec709, or native (got {value})"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::LogRec709 => "logRec709",
            Self::DlogMRec709 => "dlogMRec709",
            Self::Native => "native",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColorResolution {
    pub mode: ColorMode,
    pub detected_profile: ColorProfile,
    pub resolved_profile: ColorProfile,
    pub confidence: f64,
    pub applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lut_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lut_source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lut_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lut_file_name: Option<String>,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedLut {
    pub path: PathBuf,
    pub sha256: String,
    pub size: u64,
}

impl ValidatedLut {
    pub fn file_name(&self) -> Option<String> {
        self.path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
    }
    pub fn escaped_filter_path(&self) -> String {
        escape_filter_value(&self.path.to_string_lossy())
    }
}

/// Parse a color mode from the manifest settings.
pub fn mode_from_settings(settings: &Value) -> Result<ColorMode, String> {
    ColorMode::parse(settings.pointer("/extract/colorMode"))
}

/// Detect the profile from a full ffprobe JSON document. Explicit D-Log M
/// markers win over generic BT.709 defaults because some camera containers
/// advertise a display matrix while retaining a log transfer curve.
pub fn detect_from_probe(path: &Path, probe: &Value) -> ColorDetection {
    detect_from_probe_with_model(path, probe, None)
}

pub fn detect_from_probe_with_model(
    path: &Path,
    probe: &Value,
    camera_model: Option<&str>,
) -> ColorDetection {
    detect_from_probe_with_camera(path, probe, camera_model, None)
}

pub fn detect_from_probe_with_camera(
    path: &Path,
    probe: &Value,
    camera_model: Option<&str>,
    camera_color_profile: Option<&str>,
) -> ColorDetection {
    let mut metadata = Vec::new();
    collect_metadata(probe, &mut metadata);
    if let Some(profile) = camera_color_profile
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        metadata.push(("camera_color_profile".to_owned(), profile.to_owned()));
    }
    let camera_model = camera_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| infer_camera_model(path, &metadata));
    let normalized_model = camera_model
        .as_deref()
        .map(normalize_marker)
        .unwrap_or_default();
    let is_dji_osmo_360 = normalized_model.contains("djiosmo360") || normalized_model == "osmo360";
    let insta360_lut = insta360_lut_spec_for_model(&normalized_model);
    let model_recommendation = if is_dji_osmo_360 {
        Some(dji_lut_recommendation())
    } else {
        insta360_lut.map(insta360_lut_recommendation)
    };
    let filename_dlog = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem.to_ascii_lowercase().ends_with("_d"));
    let explicit_dlog = metadata.iter().filter(|(key, value)| {
        let key = key.to_ascii_lowercase();
        let value = value.to_ascii_lowercase();
        is_dlog_marker(&key, &value)
    });
    let dlog_count = explicit_dlog.count();
    if filename_dlog || dlog_count > 0 {
        let confidence = if dlog_count >= 2 || (filename_dlog && is_dji_osmo_360) {
            0.99
        } else if dlog_count > 0 {
            0.94
        } else {
            0.55
        };
        let reason = if filename_dlog {
            "filename stem ends in _D, the DJI D-Log M naming convention".to_owned()
        } else {
            format!(
                "ffprobe/DJI metadata contains explicit D-Log M marker ({dlog_count} evidence{})",
                if dlog_count == 1 { "" } else { "s" }
            )
        };
        return ColorDetection {
            detected_profile: ColorProfile::DlogM,
            confidence,
            reasons: vec![reason],
            should_apply: is_dji_osmo_360 && confidence >= AUTO_APPLY_CONFIDENCE,
            camera_model,
            recommended_lut: is_dji_osmo_360.then(dji_lut_recommendation),
        };
    }

    let ilog_count = metadata
        .iter()
        .filter(|(key, value)| is_ilog_marker(key, value))
        .count();
    if ilog_count > 0 {
        let matching_lut = insta360_lut.filter(|spec| spec.input_profile == ColorProfile::Ilog);
        let confidence = if matching_lut.is_some() { 0.99 } else { 0.92 };
        return ColorDetection {
            detected_profile: ColorProfile::Ilog,
            confidence,
            reasons: vec![format!(
                "ffprobe/Insta360 metadata contains explicit I-Log marker ({ilog_count} evidence{})",
                if ilog_count == 1 { "" } else { "s" }
            )],
            should_apply: matching_lut.is_some() && confidence >= AUTO_APPLY_CONFIDENCE,
            camera_model,
            recommended_lut: matching_lut.map(insta360_lut_recommendation),
        };
    }

    let insta_log_count = metadata
        .iter()
        .filter(|(key, value)| is_insta_log_marker(key, value))
        .count();
    if insta_log_count > 0 {
        let matching_lut = insta360_lut.filter(|spec| spec.input_profile == ColorProfile::InstaLog);
        let confidence = if matching_lut.is_some() { 0.99 } else { 0.90 };
        return ColorDetection {
            detected_profile: ColorProfile::InstaLog,
            confidence,
            reasons: vec![format!(
                "ffprobe/Insta360 metadata contains explicit LOG marker ({insta_log_count} evidence{})",
                if insta_log_count == 1 { "" } else { "s" }
            )],
            should_apply: matching_lut.is_some() && confidence >= AUTO_APPLY_CONFIDENCE,
            camera_model,
            recommended_lut: matching_lut.map(insta360_lut_recommendation),
        };
    }

    let non_log_profile_count = metadata
        .iter()
        .filter(|(key, value)| is_explicit_non_log_profile_marker(key, value))
        .count();
    if non_log_profile_count > 0 {
        return ColorDetection {
            detected_profile: ColorProfile::Unknown,
            confidence: 0.90,
            reasons: vec![
                "camera metadata explicitly selects a non-Log picture profile; no restoration LUT is applicable"
                    .to_owned(),
            ],
            should_apply: false,
            camera_model,
            recommended_lut: None,
        };
    }

    let explicit_rec709 = metadata.iter().filter(|(key, value)| {
        let key = key.to_ascii_lowercase();
        let value = value.to_ascii_lowercase();
        is_rec709_marker(&key, &value)
    });
    let rec709_count = explicit_rec709.count();
    if rec709_count > 0 {
        let confidence = if rec709_count >= 2 { 0.95 } else { 0.86 };
        return ColorDetection {
            detected_profile: ColorProfile::Rec709,
            confidence,
            reasons: vec![format!(
                "ffprobe color metadata identifies BT.709/Rec.709 ({rec709_count} evidence{})",
                if rec709_count == 1 { "" } else { "s" }
            )],
            should_apply: false,
            camera_model,
            recommended_lut: None,
        };
    }

    // Matrix/primaries alone are not enough: D-Log M HEVC can retain a BT.709
    // matrix while its transfer curve remains logarithmic. Treat a matching
    // transfer + primaries pair as medium evidence and keep it below the auto
    // threshold unless an explicit picture/color-profile tag was present.
    let has_bt709_transfer = metadata.iter().any(|(key, value)| {
        let key = normalize_marker(key);
        let value = normalize_marker(value);
        (key.contains("colortransfer") || key == "transfer") && value == "bt709"
    });
    let has_bt709_primaries = metadata.iter().any(|(key, value)| {
        normalize_marker(key).contains("colorprimaries") && normalize_marker(value) == "bt709"
    });
    if has_bt709_transfer && has_bt709_primaries {
        return ColorDetection {
            detected_profile: ColorProfile::Unknown,
            confidence: 0.45,
            reasons: vec![
                "ffprobe declares BT.709 transfer and primaries, but no explicit picture profile was provided".to_owned(),
                "D-Log M sources may retain a BT.709 matrix/primaries; auto mode stays conservative".to_owned(),
            ],
            should_apply: false,
            camera_model,
            recommended_lut: model_recommendation,
        };
    }

    // Camera-family metadata alone does not identify the selected picture
    // profile. Keep it visible without automatically transforming the source.
    let dji_hint = metadata.iter().any(|(key, value)| {
        let text = format!("{} {}", key, value).to_ascii_lowercase();
        text.contains("dji") || text.contains("osmo 360") || text.contains("osmo360")
    });
    if dji_hint {
        let reasons = vec![
            "DJI/Osmo metadata identifies the camera family, but no transfer profile was declared"
                .to_owned(),
        ];
        return ColorDetection {
            detected_profile: ColorProfile::Unknown,
            confidence: 0.35,
            reasons,
            should_apply: false,
            camera_model,
            recommended_lut: is_dji_osmo_360.then(dji_lut_recommendation),
        };
    }

    let mut detection = ColorDetection::unknown(
        "ffprobe metadata did not declare D-Log M or BT.709; auto mode keeps native pixels",
    );
    detection.camera_model = camera_model;
    detection.recommended_lut = model_recommendation;
    detection
}

/// Resolve a source's profile for one extract run. The caller may supply a
/// custom `.cube`; when no custom path is given, the model-specific pinned LUT
/// is fetched once into the application cache and reused after verification.
pub fn resolve_for_extract(
    mode: ColorMode,
    detection: &ColorDetection,
    lut_path: Option<&Path>,
    app_data_dir: Option<&Path>,
) -> Result<(ColorResolution, Option<ValidatedLut>), String> {
    let mut reasons = detection.reasons.clone();
    let mut warnings = Vec::new();
    let requested_apply = match mode {
        ColorMode::Native => false,
        ColorMode::LogRec709 => true,
        ColorMode::DlogMRec709 => true,
        ColorMode::Auto => detection.should_apply,
    };

    // An explicitly supplied LUT is validated whenever color processing is
    // enabled, even if auto eventually decides not to use it. This makes the
    // `lutPath` contract deterministic and prevents latent malformed files.
    let should_validate_custom = lut_path.is_some() && !matches!(mode, ColorMode::Native);
    let mut lut_id = None;
    let mut lut_source_url = None;
    let lut = if requested_apply || should_validate_custom {
        Some(match lut_path {
            Some(path) => {
                lut_id = Some("custom".to_owned());
                validate_lut_path(path)?
            }
            None => {
                let app_data_dir = app_data_dir.ok_or_else(|| {
                    "Log restoration needs an application data directory for the official LUT; select a valid extract.lutPath or retry after granting app-data access".to_owned()
                })?;
                let recommendation = detection
                    .recommended_lut
                    .clone()
                    .filter(|recommendation| {
                        !matches!(mode, ColorMode::DlogMRec709)
                            || recommendation.id == "dji-osmo-360-dlogm-rec709-v1"
                    })
                    .ok_or_else(|| {
                        "No verified built-in LUT is available for the detected camera/profile; choose a model-specific .cube file or keep native color".to_owned()
                    })?;
                lut_id = Some(recommendation.id.clone());
                lut_source_url = Some(recommendation.source_url.clone());
                download_or_reuse_official_lut(app_data_dir, &recommendation.id)?
            }
        })
    } else {
        None
    };

    let applied = requested_apply && lut.is_some();
    if matches!(mode, ColorMode::Auto) && detection.detected_profile == ColorProfile::Unknown {
        warnings.push(
            "auto mode did not apply a LUT because the input profile is unknown (fail-safe)"
                .to_owned(),
        );
    }
    if matches!(mode, ColorMode::Auto) && detection.detected_profile == ColorProfile::Rec709 {
        reasons.push("Rec.709 input is already display-referred; LUT was not applied".to_owned());
    }
    if matches!(mode, ColorMode::Native) {
        warnings.push("native mode keeps source transfer characteristics unchanged".to_owned());
    }
    if matches!(mode, ColorMode::DlogMRec709) && detection.detected_profile != ColorProfile::DlogM {
        warnings.push(
            "explicit dlogMRec709 mode overrides automatic detection and applies the LUT"
                .to_owned(),
        );
    }
    if matches!(mode, ColorMode::LogRec709)
        && detection.recommended_lut.is_none()
        && lut_path.is_none()
    {
        warnings.push(
            "explicit Log restoration needs a verified model-specific built-in LUT or custom .cube file"
                .to_owned(),
        );
    }
    let resolved_profile = if applied {
        ColorProfile::Rec709
    } else {
        detection.detected_profile
    };
    let resolution = ColorResolution {
        mode,
        detected_profile: detection.detected_profile,
        resolved_profile,
        confidence: detection.confidence,
        applied,
        lut_id,
        lut_source_url,
        lut_sha256: lut.as_ref().map(|lut| lut.sha256.clone()),
        lut_file_name: lut.as_ref().and_then(ValidatedLut::file_name),
        reasons,
        warnings,
    };
    Ok((resolution, lut))
}

/// Construct a direct FFmpeg filtergraph value. Arguments are still passed as
/// an argv array, while this quoting protects the filtergraph parser from
/// colons, commas, quotes, and backslashes in canonical user paths.
pub fn lut3d_filter(lut: &ValidatedLut) -> String {
    format!(
        "lut3d=file='{}':interp=tetrahedral",
        lut.escaped_filter_path()
    )
}

fn collect_metadata(value: &Value, output: &mut Vec<(String, String)>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                match value {
                    Value::String(value) => output.push((key.clone(), value.clone())),
                    Value::Number(value) => output.push((key.clone(), value.to_string())),
                    _ => collect_metadata(value, output),
                }
            }
        }
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_metadata(value, output)),
        _ => {}
    }
}

fn normalize_marker(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn infer_camera_model(_path: &Path, metadata: &[(String, String)]) -> Option<String> {
    let joined = metadata
        .iter()
        .map(|(key, value)| format!("{key} {value}"))
        .collect::<Vec<_>>()
        .join(" ");
    let normalized = normalize_marker(&joined);
    if normalized.contains("djiavata360") || normalized.contains("avata360") {
        return Some("DJI Avata 360".to_owned());
    }
    if normalized.contains("insta360x5") {
        return Some("Insta360 X5".to_owned());
    }
    if normalized.contains("insta360onex2") || normalized.contains("insta360x2") {
        return Some("Insta360 ONE X2".to_owned());
    }
    if normalized.contains("insta360onex") {
        return Some("Insta360 ONE X".to_owned());
    }
    if normalized.contains("insta360x3") {
        return Some("Insta360 X3".to_owned());
    }
    if normalized.contains("insta360oners") {
        return Some("Insta360 ONE RS".to_owned());
    }
    if normalized.contains("insta360oner") {
        return Some("Insta360 ONE R".to_owned());
    }
    if normalized.contains("insta360sphere") {
        return Some("Insta360 Sphere".to_owned());
    }
    if normalized.contains("djiosmo360") || normalized.contains("osmo360") {
        return Some("DJI Osmo 360".to_owned());
    }
    None
}

fn dji_lut_recommendation() -> LutRecommendation {
    LutRecommendation {
        id: "dji-osmo-360-dlogm-rec709-v1".to_owned(),
        display_name: "DJI Osmo 360 D-Log M to Rec.709 V1".to_owned(),
        file_name: DJI_DLOG_M_LUT_FILE_NAME.to_owned(),
        source_url: DJI_DLOG_M_LUT_URL.to_owned(),
    }
}

fn insta360_lut_recommendation(spec: &Insta360ArchiveLutSpec) -> LutRecommendation {
    LutRecommendation {
        id: spec.id.to_owned(),
        display_name: spec.display_name.to_owned(),
        file_name: spec.file_name.to_owned(),
        source_url: INSTA360_LUT_ARCHIVE_URL.to_owned(),
    }
}

fn insta360_lut_spec_for_model(normalized_model: &str) -> Option<&'static Insta360ArchiveLutSpec> {
    let id = if normalized_model.contains("insta360x5") || normalized_model == "x5" {
        "insta360-x5-ilog-rec709-v1"
    } else if normalized_model.contains("insta360onex2")
        || normalized_model.contains("insta360x2")
        || normalized_model == "onex2"
        || normalized_model == "x2"
    {
        "insta360-one-x2-log-rec709-v1"
    } else if normalized_model.contains("insta360onex") || normalized_model == "onex" {
        "insta360-one-x-log-rec709-v1"
    } else if normalized_model.contains("insta360x3") || normalized_model == "x3" {
        "insta360-x3-log-rec709-v1"
    } else if normalized_model.contains("insta360oners") || normalized_model == "oners" {
        "insta360-one-rs-dual-lens-360-log-rec709-v1"
    } else if normalized_model.contains("insta360oner") || normalized_model == "oner" {
        "insta360-one-r-dual-lens-360-log-rec709-v1"
    } else if normalized_model.contains("insta360sphere") || normalized_model == "sphere" {
        "insta360-sphere-log-rec709-v1"
    } else {
        return None;
    };
    INSTA360_ARCHIVE_LUTS.iter().find(|spec| spec.id == id)
}

fn is_ilog_marker(key: &str, value: &str) -> bool {
    let key = normalize_marker(key);
    let value = normalize_marker(value);
    let profile_key = key.contains("colorprofile")
        || key.contains("pictureprofile")
        || key.contains("gammamode")
        || key.contains("transfer")
        || key.contains("logcurve");
    value.contains("ilog") && (profile_key || value == "ilog")
}

fn is_insta_log_marker(key: &str, value: &str) -> bool {
    let key = normalize_marker(key);
    let value = normalize_marker(value);
    let profile_key = key.contains("colorprofile")
        || key.contains("pictureprofile")
        || key.contains("gammamode")
        || key.contains("transfer")
        || key.contains("logcurve");
    profile_key && matches!(value.as_str(), "log" | "instalog")
}

fn is_explicit_non_log_profile_marker(key: &str, value: &str) -> bool {
    let key = normalize_marker(key);
    let value = normalize_marker(value);
    let profile_key = key.contains("colorprofile")
        || key.contains("pictureprofile")
        || key.contains("gammamode")
        || key.contains("filter");
    profile_key
        && matches!(
            value.as_str(),
            "flat" | "stand" | "standard" | "vivid" | "vividbright"
        )
}

fn is_dlog_marker(key: &str, value: &str) -> bool {
    let normalized_key = normalize_marker(key);
    let normalized_value = normalize_marker(value);
    let explicit_value =
        normalized_value.contains("dlogm") || value.to_ascii_lowercase().contains("d-log m");
    let profile_key = normalized_key.contains("colorprofile")
        || normalized_key.contains("pictureprofile")
        || normalized_key.contains("gamma")
        || normalized_key.contains("transfer")
        || normalized_key.contains("logcurve")
        || normalized_key.contains("colorspace");
    let profile_value = normalized_value.contains("dlogm")
        || normalized_value == "dlog"
        || normalized_value == "dlogcurve";
    explicit_value || (profile_key && profile_value)
}

fn is_rec709_marker(key: &str, value: &str) -> bool {
    let normalized_key = normalize_marker(key);
    let normalized_value = normalize_marker(value);
    let explicit_profile_key = normalized_key.contains("colorprofile")
        || normalized_key.contains("pictureprofile")
        || normalized_key.contains("outputprofile")
        || normalized_key.contains("transferfunction");
    explicit_profile_key && matches!(normalized_value.as_str(), "bt709" | "rec709" | "bt709mpeg")
}

fn validate_lut_path(path: &Path) -> Result<ValidatedLut, String> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .ok_or_else(|| format!("LUT path must use the .cube extension: {}", path.display()))?;
    if !extension.eq_ignore_ascii_case("cube") {
        return Err(format!(
            "LUT path must use the .cube extension: {}",
            path.display()
        ));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("cannot resolve LUT path {}: {error}", path.display()))?;
    let metadata = fs::metadata(&canonical)
        .map_err(|error| format!("cannot inspect LUT {}: {error}", canonical.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "LUT path is not a regular file: {}",
            canonical.display()
        ));
    }
    if metadata.len() == 0 || metadata.len() > MAX_CUSTOM_LUT_SIZE {
        return Err(format!(
            "LUT size must be between 1 and {} bytes: {}",
            MAX_CUSTOM_LUT_SIZE,
            canonical.display()
        ));
    }
    let bytes = fs::read(&canonical)
        .map_err(|error| format!("cannot read LUT {}: {error}", canonical.display()))?;
    validate_cube_contents(&bytes, &canonical)?;
    Ok(ValidatedLut {
        path: canonical,
        sha256: sha256_hex(&bytes),
        size: bytes.len() as u64,
    })
}

fn validate_cube_contents(bytes: &[u8], path: &Path) -> Result<(), String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("LUT is not valid UTF-8 text ({}): {error}", path.display()))?;
    let mut size = None;
    let mut rows = 0usize;
    for (line_number, raw_line) in text.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let first = tokens.next().unwrap_or_default();
        match first.to_ascii_uppercase().as_str() {
            "TITLE" | "DOMAIN_MIN" | "DOMAIN_MAX" => continue,
            "LUT_3D_SIZE" => {
                if size.is_some() {
                    return Err(format!(
                        "LUT has duplicate LUT_3D_SIZE at line {}",
                        line_number + 1
                    ));
                }
                let value = tokens.next().ok_or_else(|| {
                    format!("LUT_3D_SIZE is missing a value at line {}", line_number + 1)
                })?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid LUT_3D_SIZE at line {}", line_number + 1))?;
                if !(2..=256).contains(&parsed) {
                    return Err(format!(
                        "LUT_3D_SIZE must be between 2 and 256 at line {}",
                        line_number + 1
                    ));
                }
                if tokens.next().is_some() {
                    return Err(format!(
                        "unexpected LUT_3D_SIZE tokens at line {}",
                        line_number + 1
                    ));
                }
                size = Some(parsed);
            }
            _ => {
                let values = std::iter::once(first).chain(tokens).collect::<Vec<_>>();
                if values.len() != 3 {
                    return Err(format!(
                        "LUT data row must contain exactly three values at line {}",
                        line_number + 1
                    ));
                }
                for value in values {
                    let parsed = value.parse::<f64>().map_err(|_| {
                        format!("invalid LUT data value at line {}", line_number + 1)
                    })?;
                    if !parsed.is_finite() {
                        return Err(format!(
                            "non-finite LUT data value at line {}",
                            line_number + 1
                        ));
                    }
                }
                rows += 1;
            }
        }
    }
    let size = size.ok_or_else(|| format!("LUT is missing LUT_3D_SIZE: {}", path.display()))?;
    let expected = size
        .checked_mul(size)
        .and_then(|value| value.checked_mul(size))
        .ok_or_else(|| "LUT data row count overflow".to_owned())?;
    if rows != expected {
        return Err(format!(
            "LUT has {rows} data rows but LUT_3D_SIZE {size} requires {expected}"
        ));
    }
    Ok(())
}

fn download_or_reuse_official_lut(
    app_data_dir: &Path,
    lut_id: &str,
) -> Result<ValidatedLut, String> {
    if lut_id == "dji-osmo-360-dlogm-rec709-v1" {
        return download_or_reuse_dji_lut(app_data_dir);
    }
    let spec = INSTA360_ARCHIVE_LUTS
        .iter()
        .find(|spec| spec.id == lut_id)
        .ok_or_else(|| format!("unknown built-in LUT id: {lut_id}"))?;
    download_or_reuse_insta360_lut(app_data_dir, spec)
}

fn download_or_reuse_dji_lut(app_data_dir: &Path) -> Result<ValidatedLut, String> {
    let destination = app_data_dir.join("luts").join(DJI_DLOG_M_LUT_FILE_NAME);
    if destination.is_file() {
        if let Ok(lut) = validate_lut_path(&destination) {
            if lut.size == DJI_DLOG_M_LUT_SIZE
                && lut.sha256.eq_ignore_ascii_case(DJI_DLOG_M_LUT_SHA256)
            {
                return Ok(lut);
            }
        }
    }
    let parent = destination
        .parent()
        .ok_or_else(|| "official LUT cache path has no parent".to_owned())?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "cannot create official LUT cache {}: {error}",
            parent.display()
        )
    })?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    let partial = parent.join(format!(
        ".{}.partial-{}-{stamp}",
        DJI_DLOG_M_LUT_FILE_NAME,
        std::process::id()
    ));
    let result = download_lut_to_partial(&partial);
    if result.is_err() {
        let _ = fs::remove_file(&partial);
        return result.and_then(|_| validate_lut_path(&destination));
    }
    if destination.exists() {
        fs::remove_file(&destination).map_err(|error| {
            format!(
                "cannot replace invalid cached official LUT {}: {error}",
                destination.display()
            )
        })?;
    }
    fs::rename(&partial, &destination).map_err(|error| {
        let _ = fs::remove_file(&partial);
        format!(
            "cannot commit official LUT {}: {error}",
            destination.display()
        )
    })?;
    validate_lut_path(&destination).and_then(|lut| {
        if lut.size != DJI_DLOG_M_LUT_SIZE
            || !lut.sha256.eq_ignore_ascii_case(DJI_DLOG_M_LUT_SHA256)
        {
            return Err(format!(
                "official DJI LUT verification failed after commit: expected {} bytes / {}, got {} bytes / {}",
                DJI_DLOG_M_LUT_SIZE, DJI_DLOG_M_LUT_SHA256, lut.size, lut.sha256
            ));
        }
        Ok(lut)
    })
}

fn download_or_reuse_insta360_lut(
    app_data_dir: &Path,
    spec: &Insta360ArchiveLutSpec,
) -> Result<ValidatedLut, String> {
    let destination = app_data_dir.join("luts").join(spec.file_name);
    if destination.is_file() {
        if let Ok(lut) = validate_lut_path(&destination) {
            if lut.size == spec.size && lut.sha256.eq_ignore_ascii_case(spec.sha256) {
                return Ok(lut);
            }
        }
    }
    let parent = destination
        .parent()
        .ok_or_else(|| "official Insta360 LUT cache path has no parent".to_owned())?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "cannot create official LUT cache {}: {error}",
            parent.display()
        )
    })?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    let archive_path = parent.join(format!(
        ".insta360-lut.partial-{}-{stamp}.zip",
        std::process::id()
    ));
    let result = download_verified_file(
        INSTA360_LUT_ARCHIVE_URL,
        &archive_path,
        INSTA360_LUT_ARCHIVE_SIZE,
        INSTA360_LUT_ARCHIVE_SHA256,
        "official Insta360 LUT archive",
        MAX_OFFICIAL_ARCHIVE_SIZE,
    )
    .and_then(|_| extract_verified_insta360_lut(&archive_path, &destination, spec));
    let _ = fs::remove_file(&archive_path);
    result?;
    validate_lut_path(&destination).and_then(|lut| {
        if lut.size != spec.size || !lut.sha256.eq_ignore_ascii_case(spec.sha256) {
            return Err(format!(
                "official Insta360 LUT verification failed after commit: {}",
                spec.id
            ));
        }
        Ok(lut)
    })
}

fn extract_verified_insta360_lut(
    archive_path: &Path,
    destination: &Path,
    spec: &Insta360ArchiveLutSpec,
) -> Result<(), String> {
    let archive_file = File::open(archive_path)
        .map_err(|error| format!("cannot open Insta360 LUT archive: {error}"))?;
    let mut archive = zip::ZipArchive::new(archive_file)
        .map_err(|error| format!("cannot read Insta360 LUT archive: {error}"))?;
    let mut entry = archive.by_name(spec.entry).map_err(|error| {
        format!(
            "official Insta360 LUT entry is missing ({}): {error}",
            spec.id
        )
    })?;
    if entry.size() != spec.size {
        return Err(format!(
            "official Insta360 LUT entry size mismatch for {}: expected {}, got {}",
            spec.id,
            spec.size,
            entry.size()
        ));
    }
    let parent = destination
        .parent()
        .ok_or_else(|| "official Insta360 LUT destination has no parent".to_owned())?;
    let partial = parent.join(format!(
        ".{}.partial-{}",
        spec.file_name,
        std::process::id()
    ));
    let write_result = (|| {
        let mut output = File::create(&partial)
            .map_err(|error| format!("cannot create Insta360 LUT cache file: {error}"))?;
        let mut hasher = Sha256::new();
        let mut copied = 0_u64;
        let mut buffer = [0_u8; 128 * 1024];
        loop {
            let count = entry
                .read(&mut buffer)
                .map_err(|error| format!("cannot extract official Insta360 LUT: {error}"))?;
            if count == 0 {
                break;
            }
            copied = copied.saturating_add(count as u64);
            if copied > spec.size {
                return Err(format!(
                    "official Insta360 LUT exceeded its pinned size: {}",
                    spec.id
                ));
            }
            hasher.update(&buffer[..count]);
            output
                .write_all(&buffer[..count])
                .map_err(|error| format!("cannot write Insta360 LUT cache file: {error}"))?;
        }
        output.flush().map_err(|error| error.to_string())?;
        output.sync_all().map_err(|error| error.to_string())?;
        let actual = format!("{:x}", hasher.finalize());
        if copied != spec.size || !actual.eq_ignore_ascii_case(spec.sha256) {
            return Err(format!(
                "official Insta360 LUT checksum/size mismatch for {}: got {copied} bytes / {actual}",
                spec.id
            ));
        }
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&partial);
        return Err(error);
    }
    if destination.exists() {
        fs::remove_file(destination).map_err(|error| error.to_string())?;
    }
    fs::rename(&partial, destination).map_err(|error| {
        let _ = fs::remove_file(&partial);
        format!("cannot commit official Insta360 LUT: {error}")
    })
}

fn download_lut_to_partial(partial: &Path) -> Result<(), String> {
    download_verified_file(
        DJI_DLOG_M_LUT_URL,
        partial,
        DJI_DLOG_M_LUT_SIZE,
        DJI_DLOG_M_LUT_SHA256,
        "official DJI D-Log M LUT",
        DJI_DLOG_M_LUT_SIZE,
    )
}

fn download_verified_file(
    url: &str,
    partial: &Path,
    expected_size: u64,
    expected_sha256: &str,
    label: &str,
    maximum_size: u64,
) -> Result<(), String> {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30 * 60)))
        .build();
    let agent: ureq::Agent = config.into();
    let mut response = agent
        .get(url)
        .header(
            "User-Agent",
            concat!("spherealign/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .map_err(|error| format!("cannot download {label} from {url}: {error}"))?;
    let mut reader = response.body_mut().as_reader();
    let mut output = File::create(partial)
        .map_err(|error| format!("cannot create LUT download {}: {error}", partial.display()))?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0_u64;
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("failed while downloading {label}: {error}"))?;
        if count == 0 {
            break;
        }
        downloaded = downloaded.saturating_add(count as u64);
        if downloaded > maximum_size {
            return Err(format!(
                "{label} download exceeded maximum size {maximum_size} bytes"
            ));
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("cannot write LUT download {}: {error}", partial.display()))?;
        hasher.update(&buffer[..count]);
    }
    output
        .flush()
        .map_err(|error| format!("cannot flush LUT download {}: {error}", partial.display()))?;
    output
        .sync_all()
        .map_err(|error| format!("cannot sync LUT download {}: {error}", partial.display()))?;
    let actual = format!("{:x}", hasher.finalize());
    if downloaded != expected_size {
        return Err(format!(
            "{label} size mismatch: expected {expected_size}, got {downloaded}"
        ));
    }
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        return Err(format!(
            "{label} checksum mismatch: expected {expected_sha256}, got {actual}"
        ));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// FFmpeg filtergraph escaping for one single-quoted option value.
fn escape_filter_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace(':', "\\:")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Probe one path for the standalone Tauri command. Extract uses its existing
/// full probe and calls `detect_from_probe` directly to avoid a second process.
pub fn detect_path(path: &Path) -> ColorDetection {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let Some(ffprobe) = find_executable("ffprobe") else {
        return ColorDetection::unknown("ffprobe is not available; cannot inspect color metadata");
    };
    let output = silent_command(ffprobe)
        .args([
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
        ])
        .arg(&canonical)
        .stderr(Stdio::piped())
        .output();
    let Ok(output) = output else {
        return ColorDetection::unknown("ffprobe could not be started for this source");
    };
    if !output.status.success() {
        return ColorDetection::unknown(format!(
            "ffprobe could not read this source: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice::<Value>(&output.stdout)
        .map(|probe| detect_from_probe(&canonical, &probe))
        .unwrap_or_else(|error| {
            ColorDetection::unknown(format!("ffprobe JSON was invalid: {error}"))
        })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ColorProfilePathInspection {
    pub path: String,
    #[serde(flatten)]
    pub detection: ColorDetection,
}

pub fn detect_paths(paths: Vec<String>) -> Vec<ColorProfilePathInspection> {
    paths
        .into_iter()
        .map(|path| {
            let source = PathBuf::from(&path);
            let canonical = fs::canonicalize(&source).unwrap_or(source);
            let detection = detect_path(&canonical);
            ColorProfilePathInspection {
                path: canonical.to_string_lossy().into_owned(),
                detection,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn cube(size: usize) -> String {
        let rows = size * size * size;
        let mut value = format!("TITLE \"test\"\nLUT_3D_SIZE {size}\n");
        for _ in 0..rows {
            value.push_str("0.0 0.0 0.0\n");
        }
        value
    }

    #[test]
    fn explicit_dlog_metadata_wins_over_bt709_defaults() {
        let probe = json!({
            "format": {"tags": {"DJI_ColorProfile": "D-Log M"}},
            "streams": [{"color_space": "bt709", "color_transfer": "bt709"}]
        });
        let result =
            detect_from_probe_with_model(Path::new("CAM_D.OSV"), &probe, Some("DJI Osmo 360"));
        assert_eq!(result.detected_profile, ColorProfile::DlogM);
        assert!(result.should_apply);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn bt709_metadata_is_not_lut_candidate() {
        let probe = json!({
            "format": {"tags": {"color_profile": "Rec.709"}},
            "streams": [{"color_primaries": "bt709", "color_transfer": "bt709"}]
        });
        let result = detect_from_probe(Path::new("capture.mp4"), &probe);
        assert_eq!(result.detected_profile, ColorProfile::Rec709);
        assert!(!result.should_apply);
    }

    #[test]
    fn filename_suffix_d_enables_dlog_restoration() {
        let result = detect_from_probe_with_model(
            Path::new("capture_D.OSV"),
            &json!({}),
            Some("DJI Osmo 360"),
        );
        assert_eq!(result.detected_profile, ColorProfile::DlogM);
        assert!(result.should_apply);
    }

    #[test]
    fn osv_extension_alone_never_selects_the_osmo_lut() {
        let result = detect_from_probe(Path::new("capture_D.OSV"), &json!({}));
        assert_eq!(result.detected_profile, ColorProfile::DlogM);
        assert!(!result.should_apply);
        assert!(result.camera_model.is_none());
        assert!(result.recommended_lut.is_none());
    }

    #[test]
    fn avata_360_dlog_never_selects_the_osmo_lut() {
        let result = detect_from_probe_with_camera(
            Path::new("capture_D.OSV"),
            &json!({"format": {"tags": {"DJI_ColorProfile": "D-Log M"}}}),
            Some("DJI Avata 360"),
            Some("D-Log M"),
        );
        assert_eq!(result.detected_profile, ColorProfile::DlogM);
        assert!(!result.should_apply);
        assert!(result.recommended_lut.is_none());
        assert_eq!(result.camera_model.as_deref(), Some("DJI Avata 360"));
    }

    #[test]
    fn filename_suffix_d_is_extension_agnostic_and_case_insensitive() {
        for path in ["capture_d.mp4", "capture_D.MOV"] {
            let result = detect_from_probe(Path::new(path), &json!({}));
            assert_eq!(result.detected_profile, ColorProfile::DlogM);
            assert!(
                !result.should_apply,
                "a renamed non-OSV file is only a hint"
            );
        }

        let result = detect_from_probe(Path::new("capture_D_backup.OSV"), &json!({}));
        assert_eq!(result.detected_profile, ColorProfile::Unknown);
        assert!(!result.should_apply);
    }

    #[test]
    fn x5_ilog_metadata_selects_the_pinned_official_lut() {
        let result = detect_from_probe_with_camera(
            Path::new("capture.insv"),
            &json!({}),
            Some("Insta360 X5"),
            Some("I-Log"),
        );
        assert_eq!(result.detected_profile, ColorProfile::Ilog);
        assert!(result.should_apply);
        assert_eq!(
            result.recommended_lut.as_ref().map(|lut| lut.id.as_str()),
            Some("insta360-x5-ilog-rec709-v1")
        );
        assert_eq!(result.camera_model.as_deref(), Some("Insta360 X5"));
    }

    #[test]
    fn camera_model_without_explicit_log_profile_only_recommends_a_lut() {
        let result = detect_from_probe_with_model(
            Path::new("capture.insv"),
            &json!({}),
            Some("Insta360 X5"),
        );
        assert_eq!(result.detected_profile, ColorProfile::Unknown);
        assert!(!result.should_apply);
        assert_eq!(
            result.recommended_lut.as_ref().map(|lut| lut.id.as_str()),
            Some("insta360-x5-ilog-rec709-v1")
        );
    }

    #[test]
    fn legacy_insta360_log_models_select_their_own_official_luts() {
        let cases = [
            ("Insta360 ONE X", "insta360-one-x-log-rec709-v1"),
            ("Insta360 ONE X2", "insta360-one-x2-log-rec709-v1"),
            ("Insta360 X3", "insta360-x3-log-rec709-v1"),
            (
                "Insta360 ONE R",
                "insta360-one-r-dual-lens-360-log-rec709-v1",
            ),
            (
                "Insta360 ONE RS",
                "insta360-one-rs-dual-lens-360-log-rec709-v1",
            ),
            ("Insta360 Sphere", "insta360-sphere-log-rec709-v1"),
        ];
        for (model, expected_lut) in cases {
            let result = detect_from_probe_with_camera(
                Path::new("capture.insv"),
                &json!({}),
                Some(model),
                Some("LOG"),
            );
            assert_eq!(result.detected_profile, ColorProfile::InstaLog, "{model}");
            assert!(result.should_apply, "{model}");
            assert_eq!(
                result.recommended_lut.as_ref().map(|lut| lut.id.as_str()),
                Some(expected_lut),
                "{model}"
            );
        }
    }

    #[test]
    fn insta360_luts_require_the_matching_explicit_profile() {
        for (model, profile) in [
            ("Insta360 X5", "LOG"),
            ("Insta360 X5", "FLAT"),
            ("Insta360 X3", "I-Log"),
            ("Insta360 X3", "FLAT"),
            ("Insta360 X4", "FLAT"),
            ("Insta360 X4 Air", "FLAT"),
        ] {
            let result = detect_from_probe_with_camera(
                Path::new("capture.insv"),
                &json!({}),
                Some(model),
                Some(profile),
            );
            assert!(!result.should_apply, "{model} {profile}");
            assert!(result.recommended_lut.is_none(), "{model} {profile}");
        }
    }

    #[test]
    fn official_insta360_lut_catalog_has_unique_ids_and_cache_names() {
        let mut ids = std::collections::BTreeSet::new();
        let mut file_names = std::collections::BTreeSet::new();
        for spec in INSTA360_ARCHIVE_LUTS {
            assert!(ids.insert(spec.id));
            assert!(file_names.insert(spec.file_name));
            assert!(spec.entry.starts_with("Insta360-LUT/"));
        }
    }

    #[test]
    fn cube_validation_rejects_wrong_row_count_and_accepts_valid_cube() {
        let directory = tempdir().unwrap();
        let valid = directory.path().join("valid.cube");
        fs::write(&valid, cube(2)).unwrap();
        let validated = validate_lut_path(&valid).unwrap();
        assert_eq!(validated.size, fs::metadata(&valid).unwrap().len());

        let invalid = directory.path().join("invalid.cube");
        fs::write(&invalid, "LUT_3D_SIZE 2\n0 0 0\n").unwrap();
        assert!(validate_lut_path(&invalid).is_err());
    }

    #[test]
    fn filter_path_is_escaped_without_shell_execution() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("quote'colon.cube");
        fs::write(&path, cube(2)).unwrap();
        let lut = validate_lut_path(&path).unwrap();
        let filter = lut3d_filter(&lut);
        assert!(filter.contains("quote\\'colon.cube"));
        assert!(filter.contains("interp=tetrahedral"));
    }

    #[test]
    fn filter_path_escapes_windows_drive_separator() {
        let lut = ValidatedLut {
            path: PathBuf::from(r"C:\Users\test\AppData\Roaming\SphereAlign\lut.cube"),
            sha256: "test".to_owned(),
            size: 1,
        };

        assert_eq!(
            lut3d_filter(&lut),
            r"lut3d=file='C\:\\Users\\test\\AppData\\Roaming\\SphereAlign\\lut.cube':interp=tetrahedral"
        );
    }

    #[test]
    fn explicit_mode_applies_custom_lut_even_when_metadata_is_unknown() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("custom.cube");
        fs::write(&path, cube(2)).unwrap();
        let detection = ColorDetection::unknown("test");
        let (resolution, lut) =
            resolve_for_extract(ColorMode::DlogMRec709, &detection, Some(&path), None).unwrap();
        assert!(resolution.applied);
        assert_eq!(resolution.resolved_profile, ColorProfile::Rec709);
        assert!(lut.is_some());
    }

    #[test]
    fn auto_unknown_custom_lut_is_validated_but_not_applied() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("custom.cube");
        fs::write(&path, cube(2)).unwrap();
        let detection = ColorDetection::unknown("test");
        let (resolution, lut) =
            resolve_for_extract(ColorMode::Auto, &detection, Some(&path), None).unwrap();
        assert!(!resolution.applied);
        assert_eq!(resolution.resolved_profile, ColorProfile::Unknown);
        assert!(lut.is_some(), "custom LUT should still be validated");
    }

    #[test]
    #[ignore = "requires GS360_TEST_INSTA_LUT_ARCHIVE"]
    fn extracts_all_pinned_insta360_luts_from_the_official_archive() {
        let archive = PathBuf::from(
            std::env::var("GS360_TEST_INSTA_LUT_ARCHIVE")
                .expect("GS360_TEST_INSTA_LUT_ARCHIVE is required"),
        );
        let directory = tempdir().unwrap();
        for spec in INSTA360_ARCHIVE_LUTS {
            let destination = directory.path().join(spec.file_name);
            extract_verified_insta360_lut(&archive, &destination, spec).unwrap();
            let lut = validate_lut_path(&destination).unwrap();
            assert_eq!(lut.size, spec.size, "{}", spec.id);
            assert_eq!(lut.sha256, spec.sha256, "{}", spec.id);
        }
    }
}
