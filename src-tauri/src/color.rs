//! Color-profile detection and safe D-Log M LUT resolution.
//!
//! The camera's transfer curve is not reliably inferable from image appearance
//! alone.  Detection therefore gives precedence to explicit container/stream
//! metadata and treats a DJI filename hint as weak evidence only.  Extraction
//! remains fail-safe: an `unknown` profile never receives a LUT in `auto` mode.

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
const MAX_CUSTOM_LUT_SIZE: u64 = 64 * 1024 * 1024;
const AUTO_APPLY_CONFIDENCE: f64 = 0.80;

/// The externally visible profile names are intentionally stable because they
/// are persisted in source inspection, capture metadata, and checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorProfile {
    #[serde(rename = "dlogM")]
    DlogM,
    #[serde(rename = "rec709")]
    Rec709,
    #[serde(rename = "unknown")]
    Unknown,
}

impl ColorProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DlogM => "dlogM",
            Self::Rec709 => "rec709",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColorDetection {
    pub detected_profile: ColorProfile,
    pub confidence: f64,
    pub reasons: Vec<String>,
    pub should_apply: bool,
}

impl ColorDetection {
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            detected_profile: ColorProfile::Unknown,
            confidence: 0.0,
            reasons: vec![reason.into()],
            should_apply: false,
        }
    }
}

/// User-facing extraction setting. `auto` is deliberately the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ColorMode {
    Auto,
    DlogMRec709,
    Native,
}

impl Default for ColorMode {
    fn default() -> Self {
        Self::Auto
    }
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
            "dlogmrec709" | "dlog-m-rec709" | "dlog_m_rec709" => Ok(Self::DlogMRec709),
            "native" => Ok(Self::Native),
            _ => Err(format!(
                "extract.colorMode must be auto, dlogMRec709, or native (got {value})"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
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
    let mut metadata = Vec::new();
    collect_metadata(probe, &mut metadata);
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
        let confidence = if filename_dlog || dlog_count >= 2 {
            0.99
        } else {
            0.94
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
            should_apply: confidence >= AUTO_APPLY_CONFIDENCE,
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
        };
    }

    ColorDetection::unknown(
        "ffprobe metadata did not declare D-Log M or BT.709; auto mode keeps native pixels",
    )
}

/// Resolve a source's profile for one extract run. The caller may supply a
/// custom `.cube`; when no custom path is given, the pinned DJI LUT is fetched
/// once into the application cache and then reused after size/hash validation.
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
        ColorMode::DlogMRec709 => true,
        ColorMode::Auto => detection.should_apply,
    };

    // An explicitly supplied LUT is validated whenever color processing is
    // enabled, even if auto eventually decides not to use it. This makes the
    // `lutPath` contract deterministic and prevents latent malformed files.
    let should_validate_custom = lut_path.is_some() && !matches!(mode, ColorMode::Native);
    let lut = if requested_apply || should_validate_custom {
        Some(match lut_path {
            Some(path) => validate_lut_path(path)?,
            None => {
                let app_data_dir = app_data_dir.ok_or_else(|| {
                    "D-Log M restoration needs an application data directory for the official LUT; select a valid extract.lutPath or retry after granting app-data access".to_owned()
                })?;
                download_or_reuse_official_lut(app_data_dir)?
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

fn download_or_reuse_official_lut(app_data_dir: &Path) -> Result<ValidatedLut, String> {
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

fn download_lut_to_partial(partial: &Path) -> Result<(), String> {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30 * 60)))
        .build();
    let agent: ureq::Agent = config.into();
    let mut response = agent
        .get(DJI_DLOG_M_LUT_URL)
        .header(
            "User-Agent",
            concat!("spherealign/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .map_err(|error| {
            format!("cannot download official DJI D-Log M LUT from {DJI_DLOG_M_LUT_URL}: {error}")
        })?;
    let mut reader = response.body_mut().as_reader();
    let mut output = File::create(partial)
        .map_err(|error| format!("cannot create LUT download {}: {error}", partial.display()))?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0_u64;
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("failed while downloading official DJI LUT: {error}"))?;
        if count == 0 {
            break;
        }
        downloaded = downloaded.saturating_add(count as u64);
        if downloaded > DJI_DLOG_M_LUT_SIZE {
            return Err(format!(
                "official DJI LUT download exceeded expected size {} bytes",
                DJI_DLOG_M_LUT_SIZE
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
    if downloaded != DJI_DLOG_M_LUT_SIZE {
        return Err(format!(
            "official DJI LUT size mismatch: expected {}, got {}",
            DJI_DLOG_M_LUT_SIZE, downloaded
        ));
    }
    if !actual.eq_ignore_ascii_case(DJI_DLOG_M_LUT_SHA256) {
        return Err(format!(
            "official DJI LUT checksum mismatch: expected {}, got {}",
            DJI_DLOG_M_LUT_SHA256, actual
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
        let result = detect_from_probe(Path::new("CAM_D.OSV"), &probe);
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
        let result = detect_from_probe(Path::new("capture_D.OSV"), &json!({}));
        assert_eq!(result.detected_profile, ColorProfile::DlogM);
        assert!(result.should_apply);
    }

    #[test]
    fn filename_suffix_d_is_extension_agnostic_and_case_insensitive() {
        for path in ["capture_d.mp4", "capture_D.MOV"] {
            let result = detect_from_probe(Path::new(path), &json!({}));
            assert_eq!(result.detected_profile, ColorProfile::DlogM);
            assert!(result.should_apply);
        }

        let result = detect_from_probe(Path::new("capture_D_backup.OSV"), &json!({}));
        assert_eq!(result.detected_profile, ColorProfile::Unknown);
        assert!(!result.should_apply);
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
}
