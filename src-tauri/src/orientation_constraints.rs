//! Calibrated orientation-prior interchange and optional visual-inertial
//! extensions.
//!
//! COLMAP's stock pose prior contains position, covariance, and gravity.  It
//! does **not** accept a complete rig quaternion as a bundle-adjustment
//! residual.  This module therefore keeps the calibrated quaternion contract
//! in a separate, versioned JSON manifest and exposes an explicit handshake
//! for an external orientation-aware BA executable.  The stock global mapper
//! validator only permits the gravity/fixed-rotation mode that COLMAP can
//! actually consume.
//!
//! The rolling-shutter helper below samples metadata only.  It never rewrites
//! image pixels; a future dewarper can consume the emitted trajectory after it
//! has independently validated the calibration and residual checks.

use crate::telemetry::{normalize_quaternion, AttitudeTimeline, Quaternion};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Schema version for `metadata/orientation_priors.json`.
pub const ORIENTATION_PRIOR_SCHEMA_VERSION: u32 = 1;
/// Protocol advertised by an external orientation-aware BA executable.
pub const EXTERNAL_ORIENTATION_BA_PROTOCOL: &str = "gs360.orientation-ba/v1";
/// Capability probe flag.  The executable must print one JSON object to
/// stdout and no orientation prior is sent to stock `colmap`.
pub const EXTERNAL_ORIENTATION_BA_CAPABILITY_FLAG: &str = "--gs360-orientation-capabilities";
/// JSON output flag used by the capability handshake.
pub const EXTERNAL_ORIENTATION_BA_JSON_FLAG: &str = "--format=json";
/// Marker documenting that this manifest is not a stock COLMAP quaternion
/// prior.  Keep the value stable so external tools can fail closed.
pub const STOCK_COLMAP_QUATERNION_SUPPORTED: bool = false;

/// A source description that is safe to persist in a project manifest.
/// Paths and user metadata are deliberately omitted; hashes and parser
/// revisions are sufficient to make the calibration provenance auditable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OrientationProvenance {
    /// For example, `dji_fused_attitude` or `calibrated_visual_inertial`.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser_revision: Option<String>,
    /// For example, `ffmpeg_pts_exposure_center`.
    pub timestamp_source: String,
    /// A human-readable calibration identifier, not a secret or file path.
    pub coordinate_transform: String,
}

impl OrientationProvenance {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("provenance.source", self.source.as_str()),
            ("provenance.timestampSource", self.timestamp_source.as_str()),
            (
                "provenance.coordinateTransform",
                self.coordinate_transform.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{name} must not be empty"));
            }
        }
        Ok(())
    }
}

/// One calibrated rig-frame orientation.  The quaternion convention is
/// explicit in the containing manifest (`rig_from_world`, scalar-first
/// Hamilton WXYZ); it must not be copied into COLMAP's `images.qvec` column
/// without an external residual implementation and a verified transform.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OrientationPrior {
    pub rig_frame_id: String,
    pub timestamp_ms: f64,
    pub rig_quaternion_wxyz: Quaternion,
    /// Optional row-major 3x3 covariance in squared radians.  If present it
    /// must be symmetric positive semidefinite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covariance_rad2: Option<[f64; 9]>,
    /// Optional scalar robust weight.  The external BA tool may combine this
    /// with covariance, but neither field is interpreted by stock COLMAP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    pub provenance: OrientationProvenance,
    pub calibration_version: String,
    /// Optional visual/IMU residual measured during calibration validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_angle_deg: Option<f64>,
}

impl OrientationPrior {
    /// Return a copy with a unit quaternion. This canonicalization is applied
    /// by [`OrientationPriorManifest::new`] before serialization.
    pub fn normalized(mut self) -> Result<Self, String> {
        self.rig_quaternion_wxyz = normalize_quaternion(self.rig_quaternion_wxyz)
            .ok_or_else(|| format!("{} has an invalid rig quaternion", self.rig_frame_id))?;
        Ok(self)
    }

    pub fn validate(&self, max_residual_angle_deg: f64) -> Result<(), String> {
        if self.rig_frame_id.trim().is_empty() {
            return Err("rigFrameId must not be empty".to_owned());
        }
        if !self.timestamp_ms.is_finite() {
            return Err(format!(
                "{} has a non-finite timestamp_ms",
                self.rig_frame_id
            ));
        }
        normalize_quaternion(self.rig_quaternion_wxyz).ok_or_else(|| {
            format!(
                "{} has a zero-length or non-finite rig quaternion",
                self.rig_frame_id
            )
        })?;
        if let Some(covariance) = self.covariance_rad2 {
            validate_covariance(covariance)
                .map_err(|error| format!("{} covariance_rad2: {error}", self.rig_frame_id))?;
        }
        if let Some(weight) = self.weight {
            if !weight.is_finite() || weight <= 0.0 {
                return Err(format!(
                    "{} weight must be finite and greater than zero",
                    self.rig_frame_id
                ));
            }
        }
        self.provenance.validate()?;
        if self.calibration_version.trim().is_empty() {
            return Err(format!(
                "{} calibrationVersion must not be empty",
                self.rig_frame_id
            ));
        }
        if let Some(residual) = self.residual_angle_deg {
            if !residual.is_finite() || residual < 0.0 || residual > max_residual_angle_deg {
                return Err(format!(
                    "{} residual_angle_deg {residual} exceeds {max_residual_angle_deg}",
                    self.rig_frame_id
                ));
            }
        }
        Ok(())
    }
}

/// A compact summary produced by [`validate_orientation_priors`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OrientationValidationReport {
    pub valid: bool,
    pub expected_frame_count: usize,
    pub covered_frame_count: usize,
    pub coverage_ratio: f64,
    pub max_residual_angle_deg: Option<f64>,
    pub time_offset_ms: f64,
    pub timestamp_monotonic: bool,
    pub issues: Vec<String>,
}

/// Validation thresholds for calibrated orientation priors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OrientationValidationConfig {
    pub min_coverage_ratio: f64,
    pub max_residual_angle_deg: f64,
    pub max_abs_time_offset_ms: f64,
    /// Maximum timestamp mismatch when associating a prior with an expected
    /// image timestamp.
    pub max_timestamp_error_ms: f64,
    #[serde(default)]
    pub expected_calibration_version: Option<String>,
}

impl Default for OrientationValidationConfig {
    fn default() -> Self {
        Self {
            min_coverage_ratio: 0.8,
            max_residual_angle_deg: 5.0,
            max_abs_time_offset_ms: 250.0,
            max_timestamp_error_ms: 5.0,
            expected_calibration_version: None,
        }
    }
}

impl OrientationValidationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.min_coverage_ratio.is_finite() || !(0.0..=1.0).contains(&self.min_coverage_ratio) {
            return Err("min_coverage_ratio must be finite and in [0, 1]".to_owned());
        }
        if !self.max_residual_angle_deg.is_finite() || self.max_residual_angle_deg < 0.0 {
            return Err("max_residual_angle_deg must be finite and non-negative".to_owned());
        }
        if !self.max_abs_time_offset_ms.is_finite() || self.max_abs_time_offset_ms < 0.0 {
            return Err("max_abs_time_offset_ms must be finite and non-negative".to_owned());
        }
        if !self.max_timestamp_error_ms.is_finite() || self.max_timestamp_error_ms < 0.0 {
            return Err("max_timestamp_error_ms must be finite and non-negative".to_owned());
        }
        if self
            .expected_calibration_version
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err("expected_calibration_version must not be empty".to_owned());
        }
        Ok(())
    }
}

/// Validate residuals, timestamp coverage, calibration identity, and time
/// offset before writing an orientation manifest or enabling quaternion BA.
/// `expected_timestamps_ms` are image exposure-center timestamps in image
/// time.  A prior is considered covered when its timestamp differs by at most
/// `max_timestamp_error_ms`.
pub fn validate_orientation_priors(
    priors: &[OrientationPrior],
    expected_timestamps_ms: &[f64],
    time_offset_ms: f64,
    calibration_version: &str,
    config: &OrientationValidationConfig,
) -> OrientationValidationReport {
    let expected_count = expected_timestamps_ms.len();
    let mut issues = Vec::new();
    if let Err(error) = config.validate() {
        issues.push(error);
    }
    if !time_offset_ms.is_finite() {
        issues.push("time_offset_ms must be finite".to_owned());
    } else if time_offset_ms.abs() > config.max_abs_time_offset_ms {
        issues.push(format!(
            "time_offset_ms {time_offset_ms} exceeds ±{} ms",
            config.max_abs_time_offset_ms
        ));
    }
    if calibration_version.trim().is_empty() {
        issues.push("calibration_version must not be empty".to_owned());
    }
    if let Some(expected) = config.expected_calibration_version.as_deref() {
        if expected != calibration_version {
            issues.push(format!(
                "calibration version mismatch: expected {expected}, got {calibration_version}"
            ));
        }
    }

    let mut seen_ids = HashSet::new();
    let mut previous_timestamp = None;
    let mut timestamp_monotonic = true;
    let mut max_residual = None;
    for prior in priors {
        if let Err(error) = prior.validate(config.max_residual_angle_deg) {
            issues.push(error);
        }
        if prior.calibration_version != calibration_version {
            issues.push(format!(
                "{} calibration version mismatch: expected {calibration_version}, got {}",
                prior.rig_frame_id, prior.calibration_version
            ));
        }
        if !seen_ids.insert(prior.rig_frame_id.clone()) {
            issues.push(format!("duplicate rigFrameId {}", prior.rig_frame_id));
        }
        if let Some(previous) = previous_timestamp {
            if prior.timestamp_ms < previous {
                timestamp_monotonic = false;
            }
        }
        if prior.timestamp_ms.is_finite() {
            previous_timestamp = Some(prior.timestamp_ms);
        }
        if let Some(residual) = prior.residual_angle_deg.filter(|value| value.is_finite()) {
            max_residual = Some(max_residual.map_or(residual, |value: f64| value.max(residual)));
        }
    }
    if !timestamp_monotonic {
        issues.push("orientation prior timestamps are not monotonic".to_owned());
    }

    let covered_frame_count = expected_timestamps_ms
        .iter()
        .filter(|timestamp| {
            timestamp.is_finite()
                && priors.iter().any(|prior| {
                    prior.timestamp_ms.is_finite()
                        && (prior.timestamp_ms - **timestamp).abs() <= config.max_timestamp_error_ms
                })
        })
        .count();
    let coverage_ratio = if expected_count == 0 {
        0.0
    } else {
        covered_frame_count as f64 / expected_count as f64
    };
    if expected_count == 0 {
        issues.push("expected_timestamps_ms must not be empty".to_owned());
    } else if coverage_ratio < config.min_coverage_ratio {
        issues.push(format!(
            "orientation coverage {:.3} is below {:.3}",
            coverage_ratio, config.min_coverage_ratio
        ));
    }

    OrientationValidationReport {
        valid: issues.is_empty(),
        expected_frame_count: expected_count,
        covered_frame_count,
        coverage_ratio,
        max_residual_angle_deg: max_residual,
        time_offset_ms,
        timestamp_monotonic,
        issues,
    }
}

/// Hamilton quaternion multiplication in scalar-first WXYZ order.
/// `left_from_middle * middle_from_world` yields `left_from_world` when both
/// operands use the same active-rotation convention.
pub fn multiply_quaternions(left: Quaternion, right: Quaternion) -> Option<Quaternion> {
    let left = normalize_quaternion(left)?;
    let right = normalize_quaternion(right)?;
    let result = [
        left[0] * right[0] - left[1] * right[1] - left[2] * right[2] - left[3] * right[3],
        left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2],
        left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
        left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0],
    ];
    normalize_quaternion(result)
}

/// Return the inverse of a unit quaternion. This is used only when the
/// calibration stage explicitly reports DJI's opposite world/sensor
/// orientation convention; it is never inferred from a single frame.
pub fn conjugate_quaternion(quaternion: Quaternion) -> Option<Quaternion> {
    let quaternion = normalize_quaternion(quaternion)?;
    normalize_quaternion([
        quaternion[0],
        -quaternion[1],
        -quaternion[2],
        -quaternion[3],
    ])
}

/// Apply the calibrated telemetry convention and sensor-to-rig transform.
/// Keeping this in one helper ensures orientation priors and rolling-shutter
/// trajectories cannot silently use different quaternion semantics.
fn calibrated_rig_quaternion(
    sensor_quaternion: Quaternion,
    invert_telemetry: bool,
    sensor_to_rig_quaternion: Quaternion,
) -> Option<Quaternion> {
    let sensor_quaternion = if invert_telemetry {
        conjugate_quaternion(sensor_quaternion)
    } else {
        normalize_quaternion(sensor_quaternion)
    }?;
    multiply_quaternions(sensor_to_rig_quaternion, sensor_quaternion)
}

/// Generate calibrated rig-frame priors from image exposure-centre times.
///
/// The timeline quaternion is first interpolated at
/// `image_timestamp_ms + telemetry_time_offset_ms`, then left-multiplied by
/// `sensor_to_rig_quaternion`.  This function deliberately requires the
/// caller to supply a calibrated transform; it never guesses DJI axes or
/// copies a telemetry quaternion directly into a COLMAP qvec.  Frames outside
/// telemetry coverage are omitted so the caller can report their ratio with
/// [`validate_orientation_priors`]. When `invert_telemetry` is true, the
/// interpolated telemetry quaternion is conjugated before applying the fixed
/// transform. This flag must come from a validated calibration result.
#[allow(clippy::too_many_arguments)] // Mirrors the persisted calibration contract explicitly.
pub fn build_orientation_priors(
    frame_ids_and_timestamps: &[(String, f64)],
    timeline: &AttitudeTimeline,
    telemetry_time_offset_ms: f64,
    invert_telemetry: bool,
    sensor_to_rig_quaternion: Quaternion,
    covariance_rad2: Option<[f64; 9]>,
    weight: Option<f64>,
    provenance: &OrientationProvenance,
    calibration_version: &str,
    residual_angle_deg: Option<&[f64]>,
) -> Result<Vec<OrientationPrior>, String> {
    if !telemetry_time_offset_ms.is_finite() {
        return Err("telemetry_time_offset_ms must be finite".to_owned());
    }
    normalize_quaternion(sensor_to_rig_quaternion)
        .ok_or_else(|| "sensor_to_rig_quaternion is invalid".to_owned())?;
    if calibration_version.trim().is_empty() {
        return Err("calibration_version must not be empty".to_owned());
    }
    provenance.validate()?;
    if let Some(residuals) = residual_angle_deg {
        if residuals.len() != frame_ids_and_timestamps.len() {
            return Err("residual_angle_deg length must match frame count".to_owned());
        }
    }
    if let Some(covariance) = covariance_rad2 {
        validate_covariance(covariance)?;
    }
    if let Some(weight) = weight {
        if !weight.is_finite() || weight <= 0.0 {
            return Err("weight must be finite and greater than zero".to_owned());
        }
    }

    let mut priors = Vec::with_capacity(frame_ids_and_timestamps.len());
    for (index, (frame_id, image_timestamp_ms)) in frame_ids_and_timestamps.iter().enumerate() {
        if frame_id.trim().is_empty() {
            return Err(format!("frame {index} has an empty rig frame id"));
        }
        if !image_timestamp_ms.is_finite() {
            return Err(format!("frame {frame_id} has a non-finite timestamp"));
        }
        let telemetry_timestamp_ms = image_timestamp_ms + telemetry_time_offset_ms;
        let Some(sensor_sample) = timeline.interpolate(telemetry_timestamp_ms) else {
            continue;
        };
        let rig_quaternion_wxyz = calibrated_rig_quaternion(
            sensor_sample.quaternion(),
            invert_telemetry,
            sensor_to_rig_quaternion,
        )
        .ok_or_else(|| format!("frame {frame_id} produced an invalid calibrated quaternion"))?;
        priors.push(OrientationPrior {
            rig_frame_id: frame_id.clone(),
            timestamp_ms: *image_timestamp_ms,
            rig_quaternion_wxyz,
            covariance_rad2,
            weight,
            provenance: provenance.clone(),
            calibration_version: calibration_version.to_owned(),
            residual_angle_deg: residual_angle_deg.and_then(|values| values.get(index).copied()),
        });
    }
    Ok(priors)
}

/// The versioned, standalone interchange file consumed by an external
/// orientation-aware BA executable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OrientationPriorManifest {
    pub schema_version: u32,
    pub format: String,
    /// This is deliberately not COLMAP's world-to-camera qvec convention.
    /// The external executable must declare how it maps this quaternion into
    /// its own residual before use.
    pub orientation_convention: String,
    pub quaternion_order: String,
    pub units: String,
    pub calibration_version: String,
    pub time_offset_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_coverage_ms: Option<[f64; 2]>,
    pub source: OrientationProvenance,
    pub validation: OrientationValidationReport,
    pub priors: Vec<OrientationPrior>,
    /// Explicitly false for stock COLMAP 4.1.x; this prevents a consumer from
    /// silently treating the JSON quaternion as a database qvec prior.
    pub stock_colmap_quaternion_supported: bool,
}

impl OrientationPriorManifest {
    pub const FORMAT: &'static str = "gs360.orientation-prior/v1";
    pub const ORIENTATION_CONVENTION: &'static str = "rig_from_world";
    pub const QUATERNION_ORDER: &'static str = "wxyz_hamilton";
    pub const UNITS: &'static str = "milliseconds_and_radians";

    pub fn new(
        priors: Vec<OrientationPrior>,
        expected_timestamps_ms: &[f64],
        telemetry_coverage_ms: Option<[f64; 2]>,
        time_offset_ms: f64,
        calibration_version: impl Into<String>,
        source: OrientationProvenance,
        config: &OrientationValidationConfig,
    ) -> Result<Self, String> {
        let calibration_version = calibration_version.into();
        let priors = priors
            .into_iter()
            .map(OrientationPrior::normalized)
            .collect::<Result<Vec<_>, _>>()?;
        let validation = validate_orientation_priors(
            &priors,
            expected_timestamps_ms,
            time_offset_ms,
            &calibration_version,
            config,
        );
        if !validation.valid {
            return Err(validation.issues.join("; "));
        }
        source.validate()?;
        if let Some(coverage) = telemetry_coverage_ms {
            if !coverage[0].is_finite() || !coverage[1].is_finite() || coverage[1] < coverage[0] {
                return Err("telemetry_coverage_ms must be finite and ordered".to_owned());
            }
        }
        Ok(Self {
            schema_version: ORIENTATION_PRIOR_SCHEMA_VERSION,
            format: Self::FORMAT.to_owned(),
            orientation_convention: Self::ORIENTATION_CONVENTION.to_owned(),
            quaternion_order: Self::QUATERNION_ORDER.to_owned(),
            units: Self::UNITS.to_owned(),
            calibration_version,
            time_offset_ms,
            telemetry_coverage_ms,
            source,
            validation,
            priors,
            stock_colmap_quaternion_supported: STOCK_COLMAP_QUATERNION_SUPPORTED,
        })
    }

    pub fn validate_contract(&self) -> Result<(), String> {
        if self.schema_version < ORIENTATION_PRIOR_SCHEMA_VERSION {
            return Err(format!(
                "unsupported orientation prior schema {}",
                self.schema_version
            ));
        }
        if self.format != Self::FORMAT {
            return Err(format!(
                "unsupported orientation prior format {}",
                self.format
            ));
        }
        if self.orientation_convention != Self::ORIENTATION_CONVENTION
            || self.quaternion_order != Self::QUATERNION_ORDER
            || self.units != Self::UNITS
        {
            return Err(
                "orientation prior convention/order/units do not match the v1 contract".to_owned(),
            );
        }
        if self.calibration_version.trim().is_empty() {
            return Err("calibrationVersion must not be empty".to_owned());
        }
        if !self.time_offset_ms.is_finite() {
            return Err("timeOffsetMs must be finite".to_owned());
        }
        if !self.validation.coverage_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.validation.coverage_ratio)
            || self.validation.covered_frame_count > self.validation.expected_frame_count
            || self.validation.expected_frame_count == 0
            || self.priors.is_empty()
        {
            return Err("orientation validation coverage fields are invalid".to_owned());
        }
        self.source.validate()?;
        if self.stock_colmap_quaternion_supported {
            return Err(
                "stock_colmap_quaternion_supported must remain false; use an external BA tool"
                    .to_owned(),
            );
        }
        if !self.validation.valid {
            return Err(format!(
                "orientation prior validation is invalid: {}",
                self.validation.issues.join("; ")
            ));
        }
        if !self.validation.timestamp_monotonic || !self.validation.issues.is_empty() {
            return Err("orientation validation report is internally inconsistent".to_owned());
        }
        if let Some(coverage) = self.telemetry_coverage_ms {
            if !coverage[0].is_finite() || !coverage[1].is_finite() || coverage[1] < coverage[0] {
                return Err("telemetryCoverageMs must be finite and ordered".to_owned());
            }
        }
        let config = OrientationValidationConfig {
            max_residual_angle_deg: self
                .validation
                .max_residual_angle_deg
                .unwrap_or(f64::INFINITY),
            min_coverage_ratio: self.validation.coverage_ratio,
            max_abs_time_offset_ms: self.time_offset_ms.abs(),
            max_timestamp_error_ms: f64::MAX,
            expected_calibration_version: Some(self.calibration_version.clone()),
        };
        for prior in &self.priors {
            prior.validate(config.max_residual_angle_deg)?;
            let normalized = normalize_quaternion(prior.rig_quaternion_wxyz)
                .ok_or_else(|| format!("{} has an invalid rig quaternion", prior.rig_frame_id))?;
            if prior
                .rig_quaternion_wxyz
                .iter()
                .zip(normalized.iter())
                .any(|(actual, expected)| (actual - expected).abs() > 1e-8)
            {
                return Err(format!(
                    "{} rig quaternion is not normalized",
                    prior.rig_frame_id
                ));
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn write_json(&self, path: &Path) -> Result<(), String> {
        self.validate_contract()?;
        let bytes = serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?;
        fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
    }

    #[allow(dead_code)]
    pub fn read_json(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "{} is not valid orientation prior JSON: {error}",
                path.display()
            )
        })?;
        manifest.validate_contract()?;
        Ok(manifest)
    }
}

/// Capabilities advertised by an external orientation-aware BA executable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ExternalOrientationBaCapability {
    pub protocol: String,
    pub executable_version: String,
    pub manifest_format: String,
    pub supports_orientation_priors: bool,
    pub supports_wxyz_hamilton: bool,
    pub supports_rig_frames: bool,
    pub supports_fixed_rotation: bool,
    #[serde(default)]
    pub supports_rolling_shutter_trajectory: bool,
    #[serde(default)]
    pub backend: String,
}

impl ExternalOrientationBaCapability {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol != EXTERNAL_ORIENTATION_BA_PROTOCOL {
            return Err(format!(
                "unsupported external BA protocol {}",
                self.protocol
            ));
        }
        if self.manifest_format != OrientationPriorManifest::FORMAT {
            return Err(format!(
                "external BA does not accept {}",
                self.manifest_format
            ));
        }
        if self.executable_version.trim().is_empty() {
            return Err("external BA executableVersion must not be empty".to_owned());
        }
        if self.backend.eq_ignore_ascii_case("stock_colmap")
            || self.backend.eq_ignore_ascii_case("colmap")
        {
            return Err(
                "stock COLMAP cannot consume complete orientation quaternion priors".to_owned(),
            );
        }
        if !self.supports_orientation_priors {
            return Err("external BA does not support orientation priors".to_owned());
        }
        if !self.supports_wxyz_hamilton {
            return Err(
                "external BA does not support the manifest WXYZ Hamilton convention".to_owned(),
            );
        }
        if !self.supports_rig_frames {
            return Err(
                "external BA does not support rig-frame orientation constraints".to_owned(),
            );
        }
        if !self.supports_fixed_rotation {
            return Err("external BA does not support fixed-rotation mode".to_owned());
        }
        Ok(())
    }
}

pub fn external_orientation_ba_capability_args() -> Vec<String> {
    vec![
        EXTERNAL_ORIENTATION_BA_CAPABILITY_FLAG.to_owned(),
        EXTERNAL_ORIENTATION_BA_JSON_FLAG.to_owned(),
    ]
}

/// Parse and validate the one-object JSON response from an external BA
/// capability handshake.
pub fn parse_external_orientation_ba_capability(
    stdout: &str,
) -> Result<ExternalOrientationBaCapability, String> {
    let capability: ExternalOrientationBaCapability = serde_json::from_str(stdout.trim())
        .map_err(|error| format!("external orientation BA capability is not JSON: {error}"))?;
    capability.validate()?;
    Ok(capability)
}

/// Arguments for the external BA protocol.  These arguments are never passed
/// to stock `colmap`; the caller must first complete the capability handshake.
pub fn external_orientation_ba_args(
    manifest_path: &Path,
    database_path: &Path,
    output_database_path: &Path,
    fixed_rotation: bool,
) -> Result<Vec<String>, String> {
    let paths = [manifest_path, database_path, output_database_path];
    if paths.iter().any(|path| path.as_os_str().is_empty()) {
        return Err("external BA paths must not be empty".to_owned());
    }
    Ok(vec![
        "--protocol".to_owned(),
        EXTERNAL_ORIENTATION_BA_PROTOCOL.to_owned(),
        "--orientation-prior-manifest".to_owned(),
        manifest_path.to_string_lossy().into_owned(),
        "--input-database".to_owned(),
        database_path.to_string_lossy().into_owned(),
        "--output-database".to_owned(),
        output_database_path.to_string_lossy().into_owned(),
        "--fixed-rotation".to_owned(),
        if fixed_rotation { "1" } else { "0" }.to_owned(),
        "--stock-colmap-quaternion-prior".to_owned(),
        "0".to_owned(),
    ])
}

/// Extended v1 invocation for an executable that optimizes an existing
/// COLMAP sparse model and writes a separate candidate model.  Keeping input
/// and output paths distinct lets the caller validate the result before an
/// atomic promotion.
pub fn external_orientation_ba_model_args(
    manifest_path: &Path,
    database_path: &Path,
    output_database_path: &Path,
    input_model_path: &Path,
    output_model_path: &Path,
    fixed_rotation: bool,
) -> Result<Vec<String>, String> {
    if input_model_path.as_os_str().is_empty() || output_model_path.as_os_str().is_empty() {
        return Err("external BA model paths must not be empty".to_owned());
    }
    if input_model_path == output_model_path {
        return Err("external BA input and output model paths must differ".to_owned());
    }
    let mut args = external_orientation_ba_args(
        manifest_path,
        database_path,
        output_database_path,
        fixed_rotation,
    )?;
    args.extend([
        "--input-model".to_owned(),
        input_model_path.to_string_lossy().into_owned(),
        "--output-model".to_owned(),
        output_model_path.to_string_lossy().into_owned(),
    ]);
    Ok(args)
}

/// Which implementation consumes a fixed-rotation global BA request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum FixedRotationBackend {
    /// COLMAP global mapper using its supported gravity/fixed-rotation stages.
    StockGlobalMapperGravity,
    /// A separate executable implementing quaternion residuals.
    ExternalOrientationBa,
}

/// Runtime capabilities and calibration facts needed before fixed-rotation
/// optimization is enabled.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FixedRotationGlobalBaPrerequisites {
    pub global_mapper_available: bool,
    pub fixed_rotation_option_available: bool,
    pub rig_extrinsics_complete: bool,
    pub focal_prior_valid: bool,
    pub orientation_manifest_valid: bool,
    pub gravity_prior_valid: bool,
    pub gravity_coverage_ratio: Option<f64>,
    #[serde(default)]
    pub external_capability: Option<ExternalOrientationBaCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FixedRotationValidationReport {
    pub backend: FixedRotationBackend,
    pub valid: bool,
    pub issues: Vec<String>,
}

pub fn validate_fixed_rotation_global_ba(
    backend: FixedRotationBackend,
    prerequisites: &FixedRotationGlobalBaPrerequisites,
) -> FixedRotationValidationReport {
    let mut issues = Vec::new();
    if !prerequisites.global_mapper_available {
        issues.push("COLMAP global_mapper is unavailable".to_owned());
    }
    if !prerequisites.fixed_rotation_option_available {
        issues.push("global mapper fixed-rotation BA option is unavailable".to_owned());
    }
    if !prerequisites.rig_extrinsics_complete {
        issues.push("complete sensor_from_rig extrinsics are required".to_owned());
    }
    if !prerequisites.focal_prior_valid {
        issues.push("validated focal priors are required".to_owned());
    }

    match backend {
        FixedRotationBackend::StockGlobalMapperGravity => {
            if !prerequisites.gravity_prior_valid {
                issues.push(
                    "stock global_mapper fixed-rotation mode requires gravity priors".to_owned(),
                );
            }
            if prerequisites
                .gravity_coverage_ratio
                .is_none_or(|coverage| !coverage.is_finite() || !(0.8..=1.0).contains(&coverage))
            {
                issues.push("gravity prior coverage must be at least 0.8".to_owned());
            }
            // Deliberately do not require orientation_manifest_valid here:
            // stock COLMAP can use gravity but cannot consume full quaternions.
        }
        FixedRotationBackend::ExternalOrientationBa => {
            if !prerequisites.orientation_manifest_valid {
                issues.push("validated orientation-prior manifest is required".to_owned());
            }
            match prerequisites.external_capability.as_ref() {
                Some(capability) => {
                    if let Err(error) = capability.validate() {
                        issues.push(error);
                    }
                }
                None => issues
                    .push("external orientation BA capability handshake is required".to_owned()),
            }
        }
    }

    FixedRotationValidationReport {
        backend,
        valid: issues.is_empty(),
        issues,
    }
}

/// One sampled row pose for rolling-shutter dewarp metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RollingShutterTrajectorySample {
    pub row: u32,
    pub row_fraction: f64,
    pub image_timestamp_ms: f64,
    pub telemetry_timestamp_ms: f64,
    /// Calibrated rig-from-world quaternion; raw sensor telemetry is never
    /// emitted in this field.
    pub rig_quaternion_wxyz: Option<Quaternion>,
    pub covered: bool,
}

/// Metadata-only rolling-shutter trajectory.  `pixels_modified` is always
/// false; consumers must explicitly pass this trajectory to a dewarper.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RollingShutterTrajectoryMetadata {
    pub schema_version: u32,
    pub frame_id: String,
    pub image_height: u32,
    pub requested_sample_count: usize,
    pub exposure_center_image_timestamp_ms: f64,
    pub exposure_center_telemetry_timestamp_ms: f64,
    pub readout_time_ms: f64,
    pub telemetry_time_offset_ms: f64,
    /// Calibrated rig-from-world quaternion at the exposure centre.
    pub center_rig_quaternion_wxyz: Option<Quaternion>,
    pub samples: Vec<RollingShutterTrajectorySample>,
    pub all_rows_covered: bool,
    pub pixels_modified: bool,
}

impl RollingShutterTrajectoryMetadata {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn validate_contract(&self) -> Result<(), String> {
        if self.schema_version < Self::SCHEMA_VERSION {
            return Err(format!(
                "unsupported rolling-shutter trajectory schema {}",
                self.schema_version
            ));
        }
        if self.frame_id.trim().is_empty() {
            return Err("rolling-shutter frameId must not be empty".to_owned());
        }
        if self.image_height == 0 || self.requested_sample_count == 0 {
            return Err("rolling-shutter image height/sample count must be positive".to_owned());
        }
        if !self.exposure_center_image_timestamp_ms.is_finite()
            || !self.exposure_center_telemetry_timestamp_ms.is_finite()
            || !self.readout_time_ms.is_finite()
            || self.readout_time_ms < 0.0
            || !self.telemetry_time_offset_ms.is_finite()
        {
            return Err("rolling-shutter timestamps/readout values are invalid".to_owned());
        }
        let expected_center_telemetry_timestamp =
            self.exposure_center_image_timestamp_ms + self.telemetry_time_offset_ms;
        if (self.exposure_center_telemetry_timestamp_ms - expected_center_telemetry_timestamp).abs()
            > 1e-7
        {
            return Err("rolling-shutter center timestamp does not apply time offset".to_owned());
        }
        if let Some(quaternion) = self.center_rig_quaternion_wxyz {
            let normalized = normalize_quaternion(quaternion)
                .ok_or_else(|| "rolling-shutter center quaternion is invalid".to_owned())?;
            if quaternion
                .iter()
                .zip(normalized.iter())
                .any(|(actual, expected)| (actual - expected).abs() > 1e-8)
            {
                return Err("rolling-shutter center quaternion is not normalized".to_owned());
            }
        }
        if self.pixels_modified {
            return Err("rolling-shutter trajectory metadata must not modify pixels".to_owned());
        }
        if self.samples.len() != self.requested_sample_count {
            return Err("rolling-shutter sample count does not match requested count".to_owned());
        }
        for sample in &self.samples {
            if sample.row >= self.image_height
                || !sample.row_fraction.is_finite()
                || !(0.0..=1.0).contains(&sample.row_fraction)
                || !sample.image_timestamp_ms.is_finite()
                || !sample.telemetry_timestamp_ms.is_finite()
            {
                return Err("rolling-shutter row sample is invalid".to_owned());
            }
            let expected_telemetry_timestamp =
                sample.image_timestamp_ms + self.telemetry_time_offset_ms;
            if (sample.telemetry_timestamp_ms - expected_telemetry_timestamp).abs() > 1e-7 {
                return Err("rolling-shutter row timestamp does not apply time offset".to_owned());
            }
            if let Some(quaternion) = sample.rig_quaternion_wxyz {
                normalize_quaternion(quaternion)
                    .ok_or_else(|| "rolling-shutter row quaternion is invalid".to_owned())?;
                if !sample.covered {
                    return Err("uncovered row must not carry an orientation".to_owned());
                }
            } else if sample.covered {
                return Err("covered row must carry an orientation".to_owned());
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn write_json(&self, path: &Path) -> Result<(), String> {
        self.validate_contract()?;
        let bytes = serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?;
        fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
    }

    #[allow(dead_code)]
    pub fn read_json(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let metadata: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "{} is not valid rolling-shutter metadata: {error}",
                path.display()
            )
        })?;
        metadata.validate_contract()?;
        Ok(metadata)
    }
}

/// Sample exposure-centre and row-time attitudes using the telemetry
/// timeline's shortest-path SLERP. The interpolated sensor quaternion is
/// optionally inverted and then transformed with `sensor_to_rig_quaternion`
/// before it is written as `rigQuaternionWxyz`. Timestamps outside telemetry
/// coverage are represented as `covered: false` instead of extrapolating.
#[allow(clippy::too_many_arguments)] // Keeps timing and calibration inputs explicit at the boundary.
pub fn sample_rolling_shutter_trajectory(
    frame_id: impl Into<String>,
    exposure_center_image_timestamp_ms: f64,
    readout_time_ms: f64,
    telemetry_time_offset_ms: f64,
    invert_telemetry: bool,
    sensor_to_rig_quaternion: Quaternion,
    image_height: u32,
    requested_sample_count: usize,
    timeline: &AttitudeTimeline,
) -> Result<RollingShutterTrajectoryMetadata, String> {
    let frame_id = frame_id.into();
    if frame_id.trim().is_empty() {
        return Err("frame_id must not be empty".to_owned());
    }
    if !exposure_center_image_timestamp_ms.is_finite() {
        return Err("exposure_center_image_timestamp_ms must be finite".to_owned());
    }
    if !readout_time_ms.is_finite() || readout_time_ms < 0.0 {
        return Err("readout_time_ms must be finite and non-negative".to_owned());
    }
    if !telemetry_time_offset_ms.is_finite() {
        return Err("telemetry_time_offset_ms must be finite".to_owned());
    }
    normalize_quaternion(sensor_to_rig_quaternion)
        .ok_or_else(|| "sensor_to_rig_quaternion is invalid".to_owned())?;
    if image_height == 0 {
        return Err("image_height must be greater than zero".to_owned());
    }
    if requested_sample_count == 0 {
        return Err("requested_sample_count must be greater than zero".to_owned());
    }

    let center_telemetry_timestamp_ms =
        exposure_center_image_timestamp_ms + telemetry_time_offset_ms;
    let center_quaternion = timeline
        .interpolate(center_telemetry_timestamp_ms)
        .and_then(|sample| {
            calibrated_rig_quaternion(
                sample.quaternion(),
                invert_telemetry,
                sensor_to_rig_quaternion,
            )
        });
    let sample_count = requested_sample_count.max(1);
    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let fraction = if sample_count == 1 {
            0.5
        } else {
            index as f64 / (sample_count - 1) as f64
        };
        let row = ((image_height.saturating_sub(1)) as f64 * fraction).round() as u32;
        let image_timestamp_ms =
            exposure_center_image_timestamp_ms + (fraction - 0.5) * readout_time_ms;
        let telemetry_timestamp_ms = image_timestamp_ms + telemetry_time_offset_ms;
        let quaternion = timeline
            .interpolate(telemetry_timestamp_ms)
            .and_then(|sample| {
                calibrated_rig_quaternion(
                    sample.quaternion(),
                    invert_telemetry,
                    sensor_to_rig_quaternion,
                )
            });
        samples.push(RollingShutterTrajectorySample {
            row,
            row_fraction: fraction,
            image_timestamp_ms,
            telemetry_timestamp_ms,
            covered: quaternion.is_some(),
            rig_quaternion_wxyz: quaternion,
        });
    }

    let metadata = RollingShutterTrajectoryMetadata {
        schema_version: RollingShutterTrajectoryMetadata::SCHEMA_VERSION,
        frame_id,
        image_height,
        requested_sample_count,
        exposure_center_image_timestamp_ms,
        exposure_center_telemetry_timestamp_ms: center_telemetry_timestamp_ms,
        readout_time_ms,
        telemetry_time_offset_ms,
        center_rig_quaternion_wxyz: center_quaternion,
        all_rows_covered: samples.iter().all(|sample| sample.covered),
        samples,
        pixels_modified: false,
    };
    metadata.validate_contract()?;
    Ok(metadata)
}

fn validate_covariance(covariance: [f64; 9]) -> Result<(), String> {
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err("contains non-finite values".to_owned());
    }
    let tolerance = 1e-10;
    for row in 0..3 {
        for column in (row + 1)..3 {
            if (covariance[row * 3 + column] - covariance[column * 3 + row]).abs() > tolerance {
                return Err("must be symmetric".to_owned());
            }
        }
    }
    if covariance[0] < -tolerance || covariance[4] < -tolerance || covariance[8] < -tolerance {
        return Err("diagonal entries must be non-negative".to_owned());
    }
    // Principal minors are sufficient for a symmetric 3x3 matrix to be PSD.
    let minor_01 = covariance[0] * covariance[4] - covariance[1] * covariance[3];
    let minor_02 = covariance[0] * covariance[8] - covariance[2] * covariance[6];
    let minor_12 = covariance[4] * covariance[8] - covariance[5] * covariance[7];
    let determinant = covariance[0]
        * (covariance[4] * covariance[8] - covariance[5] * covariance[7])
        - covariance[1] * (covariance[3] * covariance[8] - covariance[5] * covariance[6])
        + covariance[2] * (covariance[3] * covariance[7] - covariance[4] * covariance[6]);
    if [minor_01, minor_02, minor_12, determinant]
        .iter()
        .any(|value| *value < -tolerance)
    {
        return Err("must be positive semidefinite".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::QuaternionSample;
    use std::f64::consts::FRAC_1_SQRT_2;
    use tempfile::tempdir;

    fn provenance() -> OrientationProvenance {
        OrientationProvenance {
            source: "calibrated_visual_inertial".to_owned(),
            telemetry_sha256: Some("00".repeat(32)),
            parser_revision: Some("test-parser".to_owned()),
            timestamp_source: "ffmpeg_pts_exposure_center".to_owned(),
            coordinate_transform: "sensor_to_rig_hand_eye:v1".to_owned(),
        }
    }

    fn prior(sequence: u32, timestamp_ms: f64) -> OrientationPrior {
        OrientationPrior {
            rig_frame_id: format!("frame-{sequence:04}"),
            timestamp_ms,
            rig_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
            covariance_rad2: Some([1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]),
            weight: Some(1.0),
            provenance: provenance(),
            calibration_version: "hand-eye:v1".to_owned(),
            residual_angle_deg: Some(0.5),
        }
    }

    #[test]
    fn manifest_round_trips_and_explicitly_rejects_stock_quaternions() {
        let mut unnormalized = prior(0, 0.0);
        unnormalized.rig_quaternion_wxyz = [2.0, 0.0, 0.0, 0.0];
        let priors = vec![unnormalized, prior(1, 100.0), prior(2, 200.0)];
        let config = OrientationValidationConfig {
            expected_calibration_version: Some("hand-eye:v1".to_owned()),
            ..Default::default()
        };
        let manifest = OrientationPriorManifest::new(
            priors,
            &[0.0, 100.0, 200.0],
            Some([0.0, 200.0]),
            12.0,
            "hand-eye:v1",
            provenance(),
            &config,
        )
        .unwrap();
        assert_eq!(manifest.priors[0].rig_quaternion_wxyz, [1.0, 0.0, 0.0, 0.0]);
        assert!(!manifest.stock_colmap_quaternion_supported);
        let directory = tempdir().unwrap();
        let path = directory.path().join("orientation_priors.json");
        manifest.write_json(&path).unwrap();
        assert_eq!(
            OrientationPriorManifest::read_json(&path).unwrap(),
            manifest
        );
        let mut invalid = manifest;
        invalid.stock_colmap_quaternion_supported = true;
        assert!(invalid.validate_contract().is_err());
    }

    #[test]
    fn validation_catches_coverage_residual_offset_and_duplicates() {
        let mut second = prior(0, 100.0);
        second.residual_angle_deg = Some(7.0);
        let config = OrientationValidationConfig {
            max_residual_angle_deg: 5.0,
            max_abs_time_offset_ms: 50.0,
            min_coverage_ratio: 0.8,
            ..Default::default()
        };
        let report = validate_orientation_priors(
            &[prior(0, 0.0), second],
            &[0.0, 100.0, 200.0, 300.0],
            60.0,
            "hand-eye:v1",
            &config,
        );
        assert!(!report.valid);
        assert!(report.coverage_ratio < 0.8);
        assert!(report.issues.iter().any(|issue| issue.contains("residual")));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("time_offset")));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("duplicate")));

        let mut mismatched = prior(2, 200.0);
        mismatched.calibration_version = "hand-eye:v2".to_owned();
        let report = validate_orientation_priors(
            &[mismatched],
            &[200.0],
            0.0,
            "hand-eye:v1",
            &OrientationValidationConfig::default(),
        );
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("calibration version mismatch")));
    }

    #[test]
    fn external_capability_handshake_is_explicit_and_rejects_stock_colmap() {
        assert_eq!(
            external_orientation_ba_capability_args(),
            vec![
                EXTERNAL_ORIENTATION_BA_CAPABILITY_FLAG,
                EXTERNAL_ORIENTATION_BA_JSON_FLAG
            ]
        );
        let json = serde_json::json!({
            "protocol": EXTERNAL_ORIENTATION_BA_PROTOCOL,
            "executableVersion": "orientation-ba 0.1",
            "manifestFormat": OrientationPriorManifest::FORMAT,
            "supportsOrientationPriors": true,
            "supportsWxyzHamilton": true,
            "supportsRigFrames": true,
            "supportsFixedRotation": true,
            "supportsRollingShutterTrajectory": true,
            "backend": "ceres_orientation_ba"
        });
        let capability = parse_external_orientation_ba_capability(&json.to_string()).unwrap();
        assert!(capability.supports_rolling_shutter_trajectory);
        let args = external_orientation_ba_args(
            Path::new("metadata/orientation_priors.json"),
            Path::new("database.db"),
            Path::new("database.orientation-ba.db"),
            true,
        )
        .unwrap();
        assert!(args.contains(&"--stock-colmap-quaternion-prior".to_owned()));
        let model_args = external_orientation_ba_model_args(
            Path::new("metadata/orientation_priors.json"),
            Path::new("database.db"),
            Path::new("database.orientation-ba.db"),
            Path::new("sparse/0"),
            Path::new("sparse_orientation_candidate/0"),
            false,
        )
        .unwrap();
        assert!(model_args
            .windows(2)
            .any(|pair| { pair == ["--input-model".to_owned(), "sparse/0".to_owned()] }));
        assert!(external_orientation_ba_model_args(
            Path::new("metadata/orientation_priors.json"),
            Path::new("database.db"),
            Path::new("database.orientation-ba.db"),
            Path::new("sparse/0"),
            Path::new("sparse/0"),
            false,
        )
        .is_err());

        let mut stock = json;
        stock["backend"] = serde_json::Value::String("stock_colmap".to_owned());
        assert!(parse_external_orientation_ba_capability(&stock.to_string()).is_err());
    }

    #[test]
    fn fixed_rotation_validator_separates_stock_gravity_from_external_quaternion_ba() {
        let stock = FixedRotationGlobalBaPrerequisites {
            global_mapper_available: true,
            fixed_rotation_option_available: true,
            rig_extrinsics_complete: true,
            focal_prior_valid: true,
            orientation_manifest_valid: false,
            gravity_prior_valid: true,
            gravity_coverage_ratio: Some(0.9),
            external_capability: None,
        };
        assert!(
            validate_fixed_rotation_global_ba(
                FixedRotationBackend::StockGlobalMapperGravity,
                &stock
            )
            .valid
        );
        assert!(
            !validate_fixed_rotation_global_ba(FixedRotationBackend::ExternalOrientationBa, &stock)
                .valid
        );
    }

    #[test]
    fn rolling_shutter_samples_center_and_rows_without_pixel_mutation() {
        let samples = vec![
            QuaternionSample {
                timestamp_ms: 0.0,
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            QuaternionSample {
                timestamp_ms: 100.0,
                w: FRAC_1_SQRT_2,
                x: 0.0,
                y: 0.0,
                z: FRAC_1_SQRT_2,
            },
        ];
        let timeline = AttitudeTimeline::new(&samples);
        let trajectory = sample_rolling_shutter_trajectory(
            "frame-0001",
            50.0,
            20.0,
            0.0,
            false,
            [FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2],
            100,
            3,
            &timeline,
        )
        .unwrap();
        assert_eq!(trajectory.samples.len(), 3);
        assert_eq!(trajectory.samples[0].row, 0);
        assert_eq!(trajectory.samples[2].row, 99);
        assert!(trajectory.all_rows_covered);
        assert!(trajectory.center_rig_quaternion_wxyz.is_some());
        assert!(!trajectory.pixels_modified);
        assert!((trajectory.samples[1].image_timestamp_ms - 50.0).abs() < 1e-10);
        let center = trajectory.center_rig_quaternion_wxyz.unwrap();
        let expected_half_angle = 135.0_f64.to_radians() / 2.0;
        assert!((center[0] - expected_half_angle.cos()).abs() < 1e-10);
        assert!((center[3] - expected_half_angle.sin()).abs() < 1e-10);
        let center_row = trajectory.samples[1].rig_quaternion_wxyz.unwrap();
        assert!((center_row[0] - expected_half_angle.cos()).abs() < 1e-10);
        assert!((center_row[3] - expected_half_angle.sin()).abs() < 1e-10);
        let directory = tempdir().unwrap();
        let path = directory.path().join("trajectory.json");
        trajectory.write_json(&path).unwrap();
        let roundtrip = RollingShutterTrajectoryMetadata::read_json(&path).unwrap();
        assert_eq!(roundtrip.frame_id, trajectory.frame_id);
        assert_eq!(roundtrip.samples.len(), trajectory.samples.len());
        for (actual, expected) in roundtrip.samples.iter().zip(trajectory.samples.iter()) {
            assert!((actual.image_timestamp_ms - expected.image_timestamp_ms).abs() < 1e-12);
            assert_eq!(actual.covered, expected.covered);
            match (actual.rig_quaternion_wxyz, expected.rig_quaternion_wxyz) {
                (Some(actual), Some(expected)) => {
                    assert!(actual
                        .iter()
                        .zip(expected.iter())
                        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
                }
                (None, None) => {}
                _ => panic!("round-trip coverage mismatch"),
            }
        }

        let uncovered = sample_rolling_shutter_trajectory(
            "frame-0002",
            50.0,
            200.0,
            0.0,
            false,
            [FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2],
            100,
            3,
            &timeline,
        )
        .unwrap();
        assert!(!uncovered.all_rows_covered);
        assert!(uncovered.samples.iter().any(|sample| !sample.covered));

        let inverted = sample_rolling_shutter_trajectory(
            "frame-inverted",
            100.0,
            0.0,
            0.0,
            true,
            [1.0, 0.0, 0.0, 0.0],
            100,
            1,
            &timeline,
        )
        .unwrap();
        let inverted_center = inverted.center_rig_quaternion_wxyz.unwrap();
        assert!((inverted_center[0] - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((inverted_center[3] + FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn calibrated_prior_builder_left_multiplies_transform_and_skips_uncovered_frames() {
        let samples = vec![
            QuaternionSample {
                timestamp_ms: 0.0,
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            QuaternionSample {
                timestamp_ms: 100.0,
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        ];
        let timeline = AttitudeTimeline::new(&samples);
        let transform = [FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2];
        let ids = vec![
            ("frame-0".to_owned(), 0.0),
            ("frame-1".to_owned(), 50.0),
            ("frame-outside".to_owned(), 500.0),
        ];
        let priors = build_orientation_priors(
            &ids,
            &timeline,
            0.0,
            false,
            transform,
            None,
            Some(2.0),
            &provenance(),
            "hand-eye:v1",
            Some(&[0.25, 0.5, 0.75]),
        )
        .unwrap();
        assert_eq!(priors.len(), 2);
        assert!((priors[0].rig_quaternion_wxyz[3] - FRAC_1_SQRT_2).abs() < 1e-10);
        assert_eq!(priors[1].residual_angle_deg, Some(0.5));

        let inverted = build_orientation_priors(
            &[("frame-inverted".to_owned(), 50.0)],
            &timeline,
            0.0,
            true,
            [1.0, 0.0, 0.0, 0.0],
            None,
            None,
            &provenance(),
            "hand-eye:v1",
            None,
        )
        .unwrap();
        // The source is identity at the midpoint, so inversion remains
        // identity; use the non-identity endpoint to verify the conjugate.
        let inverted_samples = vec![
            QuaternionSample {
                timestamp_ms: 0.0,
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            QuaternionSample {
                timestamp_ms: 100.0,
                w: FRAC_1_SQRT_2,
                x: 0.0,
                y: 0.0,
                z: FRAC_1_SQRT_2,
            },
        ];
        let inverted_timeline = AttitudeTimeline::new(&inverted_samples);
        let inverted_endpoint = build_orientation_priors(
            &[("frame-inverted-end".to_owned(), 100.0)],
            &inverted_timeline,
            0.0,
            true,
            [1.0, 0.0, 0.0, 0.0],
            None,
            None,
            &provenance(),
            "hand-eye:v1",
            None,
        )
        .unwrap();
        assert_eq!(inverted.len(), 1);
        assert!((inverted_endpoint[0].rig_quaternion_wxyz[3] + FRAC_1_SQRT_2).abs() < 1e-10);
    }
}
