//! Safe, normalized telemetry export for supported video containers.
//!
//! DJI Osmo 360 currently exposes fused attitude through telemetry-parser's
//! `dvtm_oq101` decoder. Raw data streams are still preserved independently;
//! these quaternions must not be treated as COLMAP camera qvec values without
//! an explicit, verified sensor-to-camera coordinate transform.

use prost::Message;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::UNIX_EPOCH;
use telemetry_parser::tags_impl::{GroupId, TagId, TagValue};
use telemetry_parser::util::IMUData;

use crate::fisheye::{LensOpticalOcclusions, OpticalOcclusion};

const PARSER_REVISION: &str =
    "77a3b810a0e0f64688a90546c5aaf24c9dba00bd+spherealign-dji-protobuf-v1";
const NORMALIZED_TELEMETRY_SCHEMA_VERSION: u32 = 4;
const GYRO_INTEGRATION_GAP_WARNING_MS: f64 = 100.0;
const MAX_GYRO_INTEGRATION_GAP_MS: f64 = 2_000.0;
const MIN_GRAVITY_SAMPLES: usize = 16;
const MIN_GRAVITY_MAGNITUDE: f64 = 2.0;
const MAX_GRAVITY_MAGNITUDE: f64 = 50.0;
const MIN_GRAVITY_MAGNITUDE_RATIO: f64 = 0.65;
const MAX_GRAVITY_MAGNITUDE_RATIO: f64 = 1.35;
const MAX_GRAVITY_INLIER_ANGLE_DEG: f64 = 35.0;
const MAX_GRAVITY_RMS_ANGLE_DEG: f64 = 15.0;
const ACCELERATION_LOWPASS_TIME_CONSTANT_SECONDS: f64 = 0.25;
const GRAVITY_FILTER_WARMUP_SECONDS: f64 = 1.0;

/// A quaternion in scalar-first `(w, x, y, z)` order.
pub type Quaternion = [f64; 4];

/// A relative rotation larger than this is reported as a likely attitude
/// discontinuity by [`diagnose_attitude`].  This is intentionally diagnostic
/// only; it never removes a sample from a timeline.
pub const DEFAULT_ATTITUDE_DISCONTINUITY_DEG: f64 = 90.0;

/// Return a unit quaternion, or `None` for a non-finite or zero-length input.
///
/// Normalizing at the telemetry boundary keeps later angle and SLERP
/// calculations numerically stable without implying that the attitude has
/// already been transformed into a camera coordinate frame.
pub fn normalize_quaternion(quaternion: Quaternion) -> Option<Quaternion> {
    if quaternion.iter().any(|component| !component.is_finite()) {
        return None;
    }
    let norm = quaternion
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return None;
    }
    let normalized = quaternion.map(|component| component / norm);
    normalized
        .iter()
        .all(|component| component.is_finite())
        .then_some(normalized)
}

/// Return the shortest relative rotation angle between two quaternions.
///
/// `q` and `-q` represent the same rotation, hence the absolute dot product.
/// Invalid quaternions return `f64::INFINITY` so callers can treat them as a
/// mandatory keep/diagnostic case instead of silently pruning a frame.
pub fn quaternion_angle_deg(a: Quaternion, b: Quaternion) -> f64 {
    let Some(a) = normalize_quaternion(a) else {
        return f64::INFINITY;
    };
    let Some(b) = normalize_quaternion(b) else {
        return f64::INFINITY;
    };
    let dot = a
        .iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if !dot.is_finite() {
        return f64::INFINITY;
    }
    2.0 * dot.abs().clamp(0.0, 1.0).acos().to_degrees()
}

/// Interpolate two scalar-first quaternions using shortest-path SLERP.
///
/// The interpolation parameter must be finite and in `[0, 1]`; malformed
/// inputs return `None` rather than panicking or producing NaNs.
pub fn slerp_quaternion(a: Quaternion, b: Quaternion, t: f64) -> Option<Quaternion> {
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return None;
    }
    let a = normalize_quaternion(a)?;
    let mut b = normalize_quaternion(b)?;
    let mut dot = a
        .iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if !dot.is_finite() {
        return None;
    }
    // Pick the shortest arc.  Negating a quaternion does not change its
    // represented rotation.
    if dot < 0.0 {
        b = b.map(|component| -component);
        dot = -dot;
    }
    dot = dot.clamp(-1.0, 1.0);

    // For very small angles, the sine-based expression is ill-conditioned;
    // normalized linear interpolation is the continuous limit of SLERP.
    let result = if dot > 0.9995 {
        std::array::from_fn(|index| a[index] * (1.0 - t) + b[index] * t)
    } else {
        let theta = dot.acos();
        let sin_theta = theta.sin();
        if !sin_theta.is_finite() || sin_theta.abs() <= f64::EPSILON {
            return None;
        }
        let weight_a = ((1.0 - t) * theta).sin() / sin_theta;
        let weight_b = (t * theta).sin() / sin_theta;
        std::array::from_fn(|index| a[index] * weight_a + b[index] * weight_b)
    };
    normalize_quaternion(result)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QuaternionSample {
    pub timestamp_ms: f64,
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone)]
pub struct GravityDirectionTimeline {
    samples: Vec<(f64, [f64; 3])>,
    pub rms_innovation_deg: f64,
    pub correction_sample_count: usize,
}

impl GravityDirectionTimeline {
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn interpolate(&self, timestamp_ms: f64) -> Option<[f64; 3]> {
        if !timestamp_ms.is_finite() {
            return None;
        }
        let index = self
            .samples
            .binary_search_by(|sample| sample.0.total_cmp(&timestamp_ms));
        let (left, right) = match index {
            Ok(index) => return self.samples.get(index).map(|sample| sample.1),
            Err(index) => (
                self.samples.get(index.checked_sub(1)?)?,
                self.samples.get(index)?,
            ),
        };
        let duration_ms = right.0 - left.0;
        if duration_ms <= 0.0 || duration_ms > GYRO_INTEGRATION_GAP_WARNING_MS {
            return None;
        }
        let factor = (timestamp_ms - left.0) / duration_ms;
        normalize_vector(std::array::from_fn(|axis| {
            left.1[axis] * (1.0 - factor) + right.1[axis] * factor
        }))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AttitudeSource {
    #[default]
    Unavailable,
    NativeFused,
    GyroIntegratedRelative,
}

impl AttitudeSource {
    /// Only a native fused stream may carry a world/gravity reference.
    /// Gyro integration deliberately starts at identity and is relative-only.
    pub fn has_absolute_reference(self) -> bool {
        matches!(self, Self::NativeFused)
    }
}

impl QuaternionSample {
    pub fn quaternion(&self) -> Quaternion {
        [self.w, self.x, self.y, self.z]
    }

    /// Return a copy with a normalized quaternion, preserving its timestamp.
    pub fn normalized(&self) -> Option<Self> {
        if !self.timestamp_ms.is_finite() {
            return None;
        }
        let quaternion = normalize_quaternion(self.quaternion())?;
        Some(Self {
            timestamp_ms: self.timestamp_ms,
            w: quaternion[0],
            x: quaternion[1],
            y: quaternion[2],
            z: quaternion[3],
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AttitudeDiagnostics {
    pub sample_count: usize,
    pub valid_sample_count: usize,
    pub invalid_quaternion_count: usize,
    pub non_finite_timestamp_count: usize,
    pub non_monotonic_timestamp_count: usize,
    pub duplicate_timestamp_count: usize,
    pub discontinuity_count: usize,
    pub max_step_angle_deg: Option<f64>,
    pub first_timestamp_ms: Option<f64>,
    pub last_timestamp_ms: Option<f64>,
    pub coverage_duration_ms: Option<f64>,
    pub rate_hz: Option<f64>,
}

impl AttitudeDiagnostics {
    pub fn is_monotonic(&self) -> bool {
        self.non_monotonic_timestamp_count == 0
    }
}

/// Validate attitude samples without mutating the input or panicking on bad
/// telemetry.  Coverage uses the earliest/latest valid timestamp, while
/// monotonicity and discontinuity counters retain the source ordering.
pub fn diagnose_attitude(
    samples: &[QuaternionSample],
    discontinuity_threshold_deg: f64,
) -> AttitudeDiagnostics {
    let threshold = if discontinuity_threshold_deg.is_finite() {
        discontinuity_threshold_deg.max(0.0)
    } else {
        DEFAULT_ATTITUDE_DISCONTINUITY_DEG
    };
    let mut diagnostics = AttitudeDiagnostics {
        sample_count: samples.len(),
        ..AttitudeDiagnostics::default()
    };
    let mut previous_timestamp = None;
    let mut previous_quaternion = None;
    let mut min_timestamp = None;
    let mut max_timestamp = None;

    for sample in samples {
        if !sample.timestamp_ms.is_finite() {
            diagnostics.non_finite_timestamp_count += 1;
            continue;
        }
        if let Some(previous) = previous_timestamp {
            if sample.timestamp_ms < previous {
                diagnostics.non_monotonic_timestamp_count += 1;
            } else if sample.timestamp_ms == previous {
                diagnostics.duplicate_timestamp_count += 1;
            }
        }
        previous_timestamp = Some(sample.timestamp_ms);

        let Some(quaternion) = normalize_quaternion(sample.quaternion()) else {
            diagnostics.invalid_quaternion_count += 1;
            continue;
        };
        diagnostics.valid_sample_count += 1;
        min_timestamp = Some(min_timestamp.map_or(sample.timestamp_ms, |value: f64| {
            value.min(sample.timestamp_ms)
        }));
        max_timestamp = Some(max_timestamp.map_or(sample.timestamp_ms, |value: f64| {
            value.max(sample.timestamp_ms)
        }));

        if let Some(previous) = previous_quaternion {
            let angle = quaternion_angle_deg(previous, quaternion);
            if angle.is_finite() {
                diagnostics.max_step_angle_deg = Some(
                    diagnostics
                        .max_step_angle_deg
                        .map_or(angle, |value| value.max(angle)),
                );
                if angle > threshold {
                    diagnostics.discontinuity_count += 1;
                }
            }
        }
        previous_quaternion = Some(quaternion);
    }

    if let (Some(first), Some(last)) = (min_timestamp, max_timestamp) {
        diagnostics.first_timestamp_ms = Some(first);
        diagnostics.last_timestamp_ms = Some(last);
        let duration = last - first;
        if duration > 0.0 {
            diagnostics.coverage_duration_ms = Some(duration);
            diagnostics.rate_hz =
                Some(diagnostics.valid_sample_count.saturating_sub(1) as f64 * 1000.0 / duration);
        }
    }
    diagnostics
}

pub fn validate_attitude_samples(samples: &[QuaternionSample]) -> AttitudeDiagnostics {
    diagnose_attitude(samples, DEFAULT_ATTITUDE_DISCONTINUITY_DEG)
}

/// A normalized, timestamp-ordered view of attitude samples.
///
/// The source order is retained in [`AttitudeDiagnostics`] so a caller can
/// detect malformed timestamps, while interpolation uses a sorted copy and
/// therefore remains safe for otherwise recoverable telemetry.
#[derive(Debug, Clone)]
pub struct AttitudeTimeline {
    samples: Vec<QuaternionSample>,
    diagnostics: AttitudeDiagnostics,
    max_interpolation_gap_ms: Option<f64>,
}

impl AttitudeTimeline {
    #[cfg(test)]
    pub fn new(samples: &[QuaternionSample]) -> Self {
        Self::new_with_max_interpolation_gap(samples, None)
    }

    fn new_with_max_interpolation_gap(
        samples: &[QuaternionSample],
        max_interpolation_gap_ms: Option<f64>,
    ) -> Self {
        let diagnostics = validate_attitude_samples(samples);
        let mut normalized = samples
            .iter()
            .filter_map(QuaternionSample::normalized)
            .collect::<Vec<_>>();
        normalized.sort_by(|left, right| left.timestamp_ms.total_cmp(&right.timestamp_ms));
        let mut unique = Vec::<QuaternionSample>::with_capacity(normalized.len());
        for sample in normalized {
            if let Some(previous) = unique
                .last_mut()
                .filter(|previous| previous.timestamp_ms == sample.timestamp_ms)
            {
                // Prefer the latest source sample at an exact duplicate time;
                // this makes interpolation deterministic while diagnostics
                // still reports that the original stream was malformed.
                *previous = sample;
            } else {
                unique.push(sample);
            }
        }
        Self {
            samples: unique,
            diagnostics,
            max_interpolation_gap_ms,
        }
    }

    #[cfg(test)]
    pub fn samples(&self) -> &[QuaternionSample] {
        &self.samples
    }

    pub fn diagnostics(&self) -> &AttitudeDiagnostics {
        &self.diagnostics
    }

    pub fn coverage(&self) -> Option<(f64, f64)> {
        self.samples
            .first()
            .zip(self.samples.last())
            .map(|(first, last)| (first.timestamp_ms, last.timestamp_ms))
    }

    pub fn contains_timestamp(&self, timestamp_ms: f64) -> bool {
        timestamp_ms.is_finite()
            && self
                .coverage()
                .is_some_and(|(first, last)| timestamp_ms >= first && timestamp_ms <= last)
    }

    /// Interpolate at `timestamp_ms`; returns `None` outside telemetry
    /// coverage, for non-finite timestamps, or when no valid samples exist.
    pub fn interpolate(&self, timestamp_ms: f64) -> Option<QuaternionSample> {
        if !self.contains_timestamp(timestamp_ms) {
            return None;
        }
        let index = self
            .samples
            .binary_search_by(|sample| sample.timestamp_ms.total_cmp(&timestamp_ms));
        let (left, right) = match index {
            Ok(index) => {
                let sample = self.samples.get(index)?.clone();
                return Some(sample);
            }
            Err(index) => (
                self.samples.get(index.checked_sub(1)?)?,
                self.samples.get(index)?,
            ),
        };
        let duration = right.timestamp_ms - left.timestamp_ms;
        if duration <= f64::EPSILON {
            return Some(right.clone());
        }
        if self
            .max_interpolation_gap_ms
            .is_some_and(|maximum| duration > maximum)
        {
            return None;
        }
        let factor = (timestamp_ms - left.timestamp_ms) / duration;
        let quaternion = slerp_quaternion(left.quaternion(), right.quaternion(), factor)?;
        Some(QuaternionSample {
            timestamp_ms,
            w: quaternion[0],
            x: quaternion[1],
            y: quaternion[2],
            z: quaternion[3],
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedTelemetry {
    pub schema_version: u32,
    pub parser: String,
    pub parser_revision: String,
    pub camera_type: String,
    pub camera_model: Option<String>,
    pub source_size: u64,
    pub source_modified_nanos: Option<String>,
    pub timestamps_accurate: bool,
    pub sensor_readout_time_ms: Option<f64>,
    pub timebase: String,
    pub normalized_imu_sample_count: usize,
    pub normalized_imu: Vec<IMUData>,
    pub fused_attitude_sample_count: usize,
    pub fused_attitude_rate_hz: Option<f64>,
    pub fused_attitude: Vec<QuaternionSample>,
    pub integrated_gyro_attitude_sample_count: usize,
    pub integrated_gyro_attitude_rate_hz: Option<f64>,
    pub integrated_gyro_attitude: Vec<QuaternionSample>,
    #[serde(default)]
    pub attitude_source: AttitudeSource,
    #[serde(default)]
    pub attitude_diagnostics: AttitudeDiagnostics,
    pub coordinate_frame: String,
    pub applied_to_colmap: bool,
    pub warnings: Vec<String>,
}

impl NormalizedTelemetry {
    pub fn attitude_timeline(&self) -> AttitudeTimeline {
        let maximum_gap = (self.attitude_source == AttitudeSource::GyroIntegratedRelative)
            .then_some(GYRO_INTEGRATION_GAP_WARNING_MS);
        AttitudeTimeline::new_with_max_interpolation_gap(self.attitude_samples(), maximum_gap)
    }

    pub fn attitude_samples(&self) -> &[QuaternionSample] {
        if self.fused_attitude.is_empty() {
            &self.integrated_gyro_attitude
        } else {
            &self.fused_attitude
        }
    }
}

fn telemetry_source_label(parser_name: &str, camera_type: &str) -> String {
    let parser_name = parser_name.trim();
    let camera_type = camera_type.trim();
    match (camera_type.is_empty(), parser_name.is_empty()) {
        (true, true) => "source telemetry".to_owned(),
        (true, false) => format!("{parser_name} telemetry"),
        (false, true) => format!("{camera_type} telemetry"),
        (false, false) => format!("{camera_type} telemetry ({parser_name})"),
    }
}

fn describe_timebase(parser_name: &str, camera_type: &str, timestamps_accurate: bool) -> String {
    let origin = if camera_type.eq_ignore_ascii_case("Insta360") {
        "the first frame/gyro timestamp reported by telemetry-parser"
    } else if camera_type.eq_ignore_ascii_case("DJI") {
        "the first metadata frame reported by telemetry-parser"
    } else if timestamps_accurate {
        "the source timestamp origin reported by telemetry-parser"
    } else {
        "the first sample on the parser-provided relative timeline"
    };
    let parser = if parser_name.trim().is_empty() {
        "telemetry-parser"
    } else {
        parser_name.trim()
    };
    format!("milliseconds relative to {origin} ({parser}); leading samples may be negative")
}

fn describe_coordinate_frame(
    parser_name: &str,
    camera_type: &str,
    normalized_imu_sample_count: usize,
    attitude_source: AttitudeSource,
) -> String {
    let source = telemetry_source_label(parser_name, camera_type);
    match attitude_source {
        AttitudeSource::NativeFused => format!(
            "{source} fused attitude normalized by telemetry-parser; sensor-to-camera transform is unverified; not a COLMAP camera qvec"
        ),
        AttitudeSource::GyroIntegratedRelative => format!(
            "{source} normalized gyro integrated from an arbitrary identity orientation; relative rotation only; sensor-to-camera transform is unverified; not a gravity reference or COLMAP camera qvec"
        ),
        AttitudeSource::Unavailable if normalized_imu_sample_count > 0 => format!(
            "{source} normalized IMU axes; no usable attitude timeline was produced; not a camera pose or COLMAP qvec"
        ),
        AttitudeSource::Unavailable => {
            "No verified telemetry coordinate frame; not a camera pose or COLMAP qvec".to_owned()
        }
    }
}

fn base_telemetry_warnings(
    parser_name: &str,
    camera_type: &str,
    normalized_imu_sample_count: usize,
    attitude_source: AttitudeSource,
) -> Vec<String> {
    let source = telemetry_source_label(parser_name, camera_type);
    let mut warnings = vec![format!("Raw {source} streams remain the source of truth.")];
    match attitude_source {
        AttitudeSource::NativeFused => warnings.push(
            "A verified sensor-to-camera transform is required before using fused attitude as a COLMAP prior."
                .to_owned(),
        ),
        AttitudeSource::GyroIntegratedRelative => warnings.push(
            "Gyroscope samples were integrated for relative rotation and hand-eye calibration; the arbitrary initial orientation must not be used as a gravity or absolute camera-pose prior."
                .to_owned(),
        ),
        AttitudeSource::Unavailable if normalized_imu_sample_count > 0 => warnings.push(
            "Only normalized IMU samples were detected; no usable attitude timeline is available as a COLMAP prior."
                .to_owned(),
        ),
        AttitudeSource::Unavailable => warnings.push(
            "No usable IMU or fused-attitude samples are available as a COLMAP prior.".to_owned(),
        ),
    }
    warnings
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryExport {
    pub path: PathBuf,
    pub camera_model: Option<String>,
    pub normalized_imu_sample_count: usize,
    pub fused_attitude_sample_count: usize,
    pub integrated_gyro_attitude_sample_count: usize,
}

/// Lightweight metadata capability probe used by source inspection.
///
/// This intentionally reports only decoded sample availability.  It does not
/// claim that sensor coordinates are aligned with camera coordinates; that
/// requires the later, verified hand-eye calibration path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetryInspection {
    pub parser: String,
    pub camera_type: String,
    pub camera_model: Option<String>,
    pub color_profile: Option<String>,
    pub samples_available: bool,
    pub normalized_imu_sample_count: usize,
    pub gyro_sample_count: usize,
    pub accelerometer_sample_count: usize,
    pub gyro_attitude_available: bool,
    pub gravity_estimation_available: bool,
    pub fused_attitude_sample_count: usize,
}

/// Inspect source metadata without creating a normalized telemetry artifact.
///
/// telemetry-parser reads bounded beginning/end chunks for format detection
/// and parses the embedded metadata stream.  A parser error means metadata
/// could not be decoded; an identified input with zero samples is reported as
/// available-but-empty so callers can distinguish those cases.
pub fn inspect_source(input_path: &Path) -> Result<TelemetryInspection, String> {
    let mut stream = fs::File::open(input_path).map_err(|error| error.to_string())?;
    let source_size = stream.metadata().map_err(|error| error.to_string())?.len();
    let size = usize::try_from(source_size)
        .map_err(|_| "source is too large for telemetry inspection".to_owned())?;
    let input = telemetry_parser::Input::from_stream(
        &mut stream,
        size,
        input_path,
        |_| {},
        Arc::new(AtomicBool::new(false)),
    )
    .map_err(|error| error.to_string())?;
    let samples_available = input.samples.is_some();
    let normalized_imu =
        telemetry_parser::util::normalized_imu(&input, None).map_err(|error| error.to_string())?;
    let normalized_imu_sample_count = normalized_imu.len();
    let gyro_sample_count = normalized_imu
        .iter()
        .filter(|sample| {
            sample.timestamp_ms.is_finite()
                && sample
                    .gyro
                    .is_some_and(|gyro| gyro.iter().all(|value| value.is_finite()))
        })
        .count();
    let accelerometer_sample_count = normalized_imu
        .iter()
        .filter(|sample| {
            sample.timestamp_ms.is_finite()
                && sample
                    .accl
                    .is_some_and(|value| value.iter().all(|component| component.is_finite()))
        })
        .count();
    let integrated = input
        .camera_type()
        .eq_ignore_ascii_case("Insta360")
        .then(|| integrate_normalized_gyro(&normalized_imu))
        .transpose()
        .ok()
        .flatten();
    let gyro_attitude_available = integrated.is_some();
    let gravity_estimation_available =
        integrated.is_some() && build_gravity_direction_timeline(&normalized_imu).is_some();
    let mut fused_attitude_sample_count = 0usize;
    let mut color_profiles = Vec::new();
    for sample in input.samples.iter().flatten() {
        let Some(groups) = sample.tag_map.as_ref() else {
            continue;
        };
        if let Some(profile) = groups
            .get(&GroupId::Default)
            .and_then(|tags| tags.get(&TagId::Metadata))
            .and_then(|tag| match &tag.value {
                TagValue::Json(value) => {
                    let metadata = value.get();
                    metadata
                        .get("gamma_mode")
                        .or_else(|| metadata.get("gammaMode"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::trim)
                        .filter(|profile| !profile.is_empty())
                        .map(str::to_owned)
                }
                _ => None,
            })
        {
            let normalized = profile
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase();
            if !color_profiles
                .iter()
                .any(|(_, existing): &(String, String)| existing == &normalized)
            {
                color_profiles.push((profile, normalized));
            }
        }
        let Some(group) = groups.get(&GroupId::Quaternion) else {
            continue;
        };
        let Some(tag) = group.get(&TagId::Data) else {
            continue;
        };
        if let TagValue::Vec_TimeQuaternion_f64(values) = &tag.value {
            fused_attitude_sample_count = fused_attitude_sample_count.saturating_add(
                values
                    .get()
                    .iter()
                    .filter(|value| {
                        QuaternionSample {
                            timestamp_ms: value.t,
                            w: value.v.w,
                            x: value.v.x,
                            y: value.v.y,
                            z: value.v.z,
                        }
                        .normalized()
                        .is_some()
                    })
                    .count(),
            );
        }
    }
    let color_profile = (color_profiles.len() == 1).then(|| color_profiles.remove(0).0);
    Ok(TelemetryInspection {
        parser: input.parser_name().to_owned(),
        camera_type: input.camera_type(),
        camera_model: input.camera_model().cloned(),
        color_profile,
        samples_available,
        normalized_imu_sample_count,
        gyro_sample_count,
        accelerometer_sample_count,
        gyro_attitude_available,
        gravity_estimation_available,
        fused_attitude_sample_count,
    })
}

#[derive(Clone, PartialEq, Message)]
struct DjiProductMeta {
    #[prost(message, optional, tag = "2")]
    stream_meta: Option<DjiStreamMeta>,
}

/// DJI product schemas share a stable protobuf envelope even when a new
/// camera ships before telemetry-parser has added its generated product
/// module.  Keep this deliberately partial: prost ignores unknown fields and
/// we only decode the timestamped fused-attitude fields that the downstream
/// hand-eye validator can independently verify.
#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryProductMeta {
    #[prost(message, optional, tag = "1")]
    clip_meta: Option<DjiTelemetryClipMeta>,
    #[prost(message, optional, tag = "3")]
    frame_meta: Option<DjiTelemetryFrameMeta>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryClipMeta {
    #[prost(message, optional, tag = "1")]
    clip_meta_header: Option<DjiTelemetryClipMetaHeader>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryClipMetaHeader {
    #[prost(string, tag = "1")]
    proto_file_name: String,
    #[prost(string, tag = "10")]
    product_name: String,
}

#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryFrameMeta {
    #[prost(message, optional, tag = "1")]
    frame_meta_header: Option<DjiTelemetryFrameMetaHeader>,
    // Camera-frame fields vary significantly between DJI products. Keep the
    // payload opaque so an unrelated field-number reuse cannot reject the
    // entire frame before the IMU message is reached.
    #[prost(bytes = "vec", optional, tag = "2")]
    camera_frame_meta: Option<Vec<u8>>,
    #[prost(message, optional, tag = "3")]
    imu_frame_meta: Option<DjiTelemetryImuFrameMeta>,
    #[prost(bytes = "vec", optional, tag = "4")]
    product_frame_meta: Option<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Message)]
struct DjiTelemetryFrameMetaHeader {
    #[prost(uint64, tag = "1")]
    frame_sequence: u64,
    #[prost(uint64, tag = "2")]
    frame_timestamp_us: u64,
    #[prost(uint32, tag = "3")]
    stream_id: u32,
}

#[derive(Clone, PartialEq, Message)]
struct DjiOq101CameraFrameMeta {
    #[prost(message, optional, tag = "9")]
    camera_attitude: Option<DjiTelemetryQuaternion>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiAvata360CameraFrameMeta {
    #[prost(message, optional, tag = "5")]
    camera_attitude: Option<DjiTelemetryQuaternion>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiAvata360ProductFrameMeta {
    #[prost(message, optional, tag = "4")]
    gps: Option<DjiTelemetryGpsBasic>,
    #[prost(message, optional, tag = "5")]
    relative_altitude: Option<DjiTelemetryRelativeAltitude>,
    #[prost(message, optional, tag = "14")]
    stabilization_meta: Option<DjiAvata360StabilizationMeta>,
}

#[derive(Clone, Copy, PartialEq, Message)]
struct DjiTelemetryPositionCoord {
    #[prost(enumeration = "DjiTelemetryPositionCoordUnit", tag = "1")]
    unit: i32,
    #[prost(double, tag = "2")]
    latitude: f64,
    #[prost(double, tag = "3")]
    longitude: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, prost::Enumeration)]
enum DjiTelemetryPositionCoordUnit {
    Radians = 0,
    Degrees = 1,
}

#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryGpsBasic {
    #[prost(message, optional, tag = "1")]
    coordinates: Option<DjiTelemetryPositionCoord>,
    #[prost(int32, tag = "2")]
    altitude_mm: i32,
    #[prost(int32, tag = "3")]
    status: i32,
    #[prost(int32, tag = "4")]
    altitude_type: i32,
    #[prost(bool, tag = "5")]
    has_gps_time: bool,
}

#[derive(Clone, Copy, PartialEq, Message)]
struct DjiTelemetryRelativeAltitude {
    #[prost(float, tag = "1")]
    altitude_mm: f32,
    #[prost(bool, tag = "2")]
    valid: bool,
}

#[derive(Clone, Copy, PartialEq, Message)]
struct DjiAvata360StabilizationMeta {
    #[prost(message, optional, tag = "4")]
    camera_attitude: Option<DjiTelemetryQuaternion>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryImuFrameMeta {
    // WM169 stores DeviceAttitude here; OQ101 and AVATA360 store
    // DeviceMultiAttitude. Decode the length-delimited payload after the
    // product schema is known instead of assigning the wrong generated type.
    #[prost(bytes = "vec", optional, tag = "2")]
    attitude_tag2: Option<Vec<u8>>,
    // WA530 exposes its single-frame fused attitude at field 4.
    #[prost(bytes = "vec", optional, tag = "4")]
    attitude_tag4: Option<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryDeviceMultiAttitude {
    #[prost(message, optional, tag = "1")]
    current_frame: Option<DjiTelemetryDeviceAttitude>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiTelemetryDeviceAttitude {
    #[prost(uint32, tag = "1")]
    timestamp: u32,
    #[prost(uint32, tag = "2")]
    vsync: u32,
    #[prost(message, repeated, tag = "3")]
    attitude: Vec<DjiTelemetryQuaternion>,
    #[prost(float, tag = "4")]
    offset: f32,
}

#[derive(Clone, Copy, PartialEq, Message)]
struct DjiTelemetryQuaternion {
    #[prost(float, tag = "1")]
    w: f32,
    #[prost(float, tag = "2")]
    x: f32,
    #[prost(float, tag = "3")]
    y: f32,
    #[prost(float, tag = "4")]
    z: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DjiAttitudeLayout {
    MultiTag2,
    SingleTag2,
    SingleTag4,
    Auto,
}

fn dji_attitude_layout(proto_file_name: &str) -> DjiAttitudeLayout {
    let schema = proto_file_name.to_ascii_lowercase();
    if schema.contains("oq101") || schema.contains("avata360") {
        DjiAttitudeLayout::MultiTag2
    } else if schema.contains("wa530") {
        DjiAttitudeLayout::SingleTag4
    } else if schema.contains("wm169") {
        DjiAttitudeLayout::SingleTag2
    } else {
        DjiAttitudeLayout::Auto
    }
}

fn decode_dji_camera_attitude(
    payload: &[u8],
    proto_file_name: &str,
) -> Option<DjiTelemetryQuaternion> {
    let schema = proto_file_name.to_ascii_lowercase();
    if schema.contains("avata360") {
        DjiAvata360CameraFrameMeta::decode(payload)
            .ok()?
            .camera_attitude
    } else if schema.contains("oq101") {
        DjiOq101CameraFrameMeta::decode(payload)
            .ok()?
            .camera_attitude
    } else {
        DjiAvata360CameraFrameMeta::decode(payload)
            .ok()
            .and_then(|camera| camera.camera_attitude)
            .or_else(|| {
                DjiOq101CameraFrameMeta::decode(payload)
                    .ok()
                    .and_then(|camera| camera.camera_attitude)
            })
    }
}

fn decode_avata360_stabilized_camera_attitude(payload: &[u8]) -> Option<DjiTelemetryQuaternion> {
    DjiAvata360ProductFrameMeta::decode(payload)
        .ok()?
        .stabilization_meta?
        .camera_attitude
}

fn decode_dji_device_attitude(
    imu: &DjiTelemetryImuFrameMeta,
    layout: DjiAttitudeLayout,
) -> Option<DjiTelemetryDeviceAttitude> {
    let decode_multi_tag2 = || {
        let payload = imu.attitude_tag2.as_deref()?;
        DjiTelemetryDeviceMultiAttitude::decode(payload)
            .ok()?
            .current_frame
    };
    let decode_single_tag2 =
        || DjiTelemetryDeviceAttitude::decode(imu.attitude_tag2.as_deref()?).ok();
    let decode_single_tag4 =
        || DjiTelemetryDeviceAttitude::decode(imu.attitude_tag4.as_deref()?).ok();
    let valid = |value: Option<DjiTelemetryDeviceAttitude>| {
        value.filter(|attitude| !attitude.attitude.is_empty())
    };

    match layout {
        DjiAttitudeLayout::MultiTag2 => valid(decode_multi_tag2()),
        DjiAttitudeLayout::SingleTag2 => valid(decode_single_tag2()),
        DjiAttitudeLayout::SingleTag4 => valid(decode_single_tag4()),
        DjiAttitudeLayout::Auto => valid(decode_multi_tag2())
            .or_else(|| valid(decode_single_tag4()))
            .or_else(|| valid(decode_single_tag2())),
    }
}

fn multiply_quaternions(left: Quaternion, right: Quaternion) -> Quaternion {
    let [lw, lx, ly, lz] = left;
    let [rw, rx, ry, rz] = right;
    [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]
}

/// Integrate normalized body-frame angular velocity into a relative attitude.
///
/// telemetry-parser's normalized IMU contract is degrees/second and
/// milliseconds. The initial orientation is intentionally identity, so the
/// result is suitable for relative-rotation hand-eye calibration but not for
/// gravity alignment or an absolute camera pose prior.
#[derive(Debug)]
struct GyroIntegration {
    samples: Vec<QuaternionSample>,
    gap_count: usize,
    max_gap_ms: f64,
    non_monotonic_timestamp_count: usize,
    duplicate_timestamp_count: usize,
}

fn integrate_normalized_gyro(samples: &[IMUData]) -> Result<GyroIntegration, String> {
    let mut gyro = samples
        .iter()
        .filter_map(|sample| {
            let value = sample.gyro?;
            (sample.timestamp_ms.is_finite() && value.iter().all(|component| component.is_finite()))
                .then_some((sample.timestamp_ms, value))
        })
        .collect::<Vec<_>>();
    let non_monotonic_timestamp_count =
        gyro.windows(2).filter(|pair| pair[1].0 < pair[0].0).count();
    let duplicate_timestamp_count = gyro
        .windows(2)
        .filter(|pair| pair[1].0 == pair[0].0)
        .count();
    gyro.sort_by(|left, right| left.0.total_cmp(&right.0));
    gyro.dedup_by(|left, right| left.0 == right.0);
    if gyro.len() < 2 {
        return Err("fewer than two finite gyro samples are available".to_owned());
    }

    let mut attitude = [1.0, 0.0, 0.0, 0.0];
    let mut integrated = Vec::with_capacity(gyro.len());
    let mut gap_count = 0usize;
    let mut max_gap_ms = 0.0_f64;
    integrated.push(QuaternionSample {
        timestamp_ms: gyro[0].0,
        w: attitude[0],
        x: attitude[1],
        y: attitude[2],
        z: attitude[3],
    });
    for pair in gyro.windows(2) {
        let [(previous_timestamp, previous), (timestamp, current)] = pair else {
            continue;
        };
        let dt_ms = timestamp - previous_timestamp;
        if !dt_ms.is_finite() || dt_ms <= 0.0 {
            return Err("gyro timestamps are not strictly increasing".to_owned());
        }
        if dt_ms > MAX_GYRO_INTEGRATION_GAP_MS {
            return Err(format!(
                "gyro timeline contains a {dt_ms:.3} ms gap (maximum supported gap is {MAX_GYRO_INTEGRATION_GAP_MS:.1} ms)"
            ));
        }
        if dt_ms > GYRO_INTEGRATION_GAP_WARNING_MS {
            gap_count += 1;
            max_gap_ms = max_gap_ms.max(dt_ms);
        }

        let dt_seconds = dt_ms / 1_000.0;
        let omega = std::array::from_fn::<_, 3, _>(|axis| {
            (previous[axis] + current[axis]) * 0.5_f64.to_radians()
        });
        let speed = omega.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !speed.is_finite() {
            return Err("gyro angular speed overflowed during integration".to_owned());
        }
        let delta = if speed <= f64::EPSILON {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            let half_angle = speed * dt_seconds * 0.5;
            let scale = half_angle.sin() / speed;
            [
                half_angle.cos(),
                omega[0] * scale,
                omega[1] * scale,
                omega[2] * scale,
            ]
        };
        attitude = normalize_quaternion(multiply_quaternions(attitude, delta))
            .ok_or_else(|| "gyro integration produced an invalid quaternion".to_owned())?;
        integrated.push(QuaternionSample {
            timestamp_ms: *timestamp,
            w: attitude[0],
            x: attitude[1],
            y: attitude[2],
            z: attitude[3],
        });
    }
    Ok(GyroIntegration {
        samples: integrated,
        gap_count,
        max_gap_ms,
        non_monotonic_timestamp_count,
        duplicate_timestamp_count,
    })
}

fn normalize_vector(vector: [f64; 3]) -> Option<[f64; 3]> {
    if vector.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return None;
    }
    Some(vector.map(|value| value / norm))
}

fn rotate_vector(quaternion: Quaternion, vector: [f64; 3]) -> Option<[f64; 3]> {
    let [w, x, y, z] = normalize_quaternion(quaternion)?;
    let [vx, vy, vz] = vector;
    let uv = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx];
    let uuv = [
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    ];
    normalize_vector([
        vx + 2.0 * (w * uv[0] + uuv[0]),
        vy + 2.0 * (w * uv[1] + uuv[1]),
        vz + 2.0 * (w * uv[2] + uuv[2]),
    ])
}

fn vector_angle_deg(left: [f64; 3], right: [f64; 3]) -> Option<f64> {
    let left = normalize_vector(left)?;
    let right = normalize_vector(right)?;
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    dot.is_finite()
        .then(|| dot.clamp(-1.0, 1.0).acos().to_degrees())
}

fn fuse_gravity_candidate(
    normalized_imu: &[IMUData],
    gravity_magnitude: f64,
    gyro_sign: f64,
) -> Option<GravityDirectionTimeline> {
    let mut ordered = normalized_imu
        .iter()
        .filter(|sample| sample.timestamp_ms.is_finite())
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.timestamp_ms.total_cmp(&right.timestamp_ms));

    let mut state = None;
    let mut filtered_acceleration: Option<[f64; 3]> = None;
    let mut previous_timestamp_ms: Option<f64> = None;
    let mut segment_elapsed_seconds = 0.0;
    let mut samples = Vec::with_capacity(ordered.len());
    let mut innovations = Vec::new();
    for sample in ordered {
        let dt_seconds = previous_timestamp_ms
            .map(|previous| (sample.timestamp_ms - previous) / 1_000.0)
            .filter(|dt| dt.is_finite() && *dt > 0.0);
        if dt_seconds.is_some_and(|dt| dt * 1_000.0 > GYRO_INTEGRATION_GAP_WARNING_MS) {
            state = None;
            filtered_acceleration = None;
            segment_elapsed_seconds = 0.0;
        } else if let (Some(current), Some(dt), Some(gyro)) = (state, dt_seconds, sample.gyro) {
            if gyro.iter().all(|value| value.is_finite()) {
                let omega = gyro.map(f64::to_radians);
                let speed = omega.iter().map(|value| value * value).sum::<f64>().sqrt();
                if speed.is_finite() && speed > f64::EPSILON {
                    let half_angle = gyro_sign * speed * dt * 0.5;
                    let scale = half_angle.sin() / speed;
                    let delta = [
                        half_angle.cos(),
                        omega[0] * scale,
                        omega[1] * scale,
                        omega[2] * scale,
                    ];
                    state = rotate_vector(delta, current);
                }
            }
        }

        let dt = dt_seconds.unwrap_or(0.001).clamp(0.000_1, 0.1);
        if let Some(raw) = sample
            .accl
            .filter(|value| value.iter().all(|axis| axis.is_finite()))
        {
            let alpha =
                (dt / (ACCELERATION_LOWPASS_TIME_CONSTANT_SECONDS + dt)).clamp(0.000_1, 1.0);
            filtered_acceleration = Some(match filtered_acceleration {
                Some(previous) => {
                    std::array::from_fn(|axis| previous[axis] * (1.0 - alpha) + raw[axis] * alpha)
                }
                None => raw,
            });
        }
        segment_elapsed_seconds += dt;
        let measurement = filtered_acceleration.and_then(|acceleration| {
            let magnitude = acceleration
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            (segment_elapsed_seconds >= GRAVITY_FILTER_WARMUP_SECONDS
                && magnitude.is_finite()
                && (gravity_magnitude * MIN_GRAVITY_MAGNITUDE_RATIO
                    ..=gravity_magnitude * MAX_GRAVITY_MAGNITUDE_RATIO)
                    .contains(&magnitude))
            .then(|| normalize_vector(acceleration.map(|value| -value)))
            .flatten()
        });

        match (state, measurement) {
            (None, Some(measurement)) => state = Some(measurement),
            (Some(predicted), Some(measurement)) => {
                let innovation = vector_angle_deg(predicted, measurement)?;
                if innovation <= MAX_GRAVITY_INLIER_ANGLE_DEG {
                    innovations.push(innovation);
                    let correction = (dt / (1.5 + dt)).clamp(0.000_1, 0.08);
                    state = normalize_vector(std::array::from_fn(|axis| {
                        predicted[axis] * (1.0 - correction) + measurement[axis] * correction
                    }));
                }
            }
            _ => {}
        }
        previous_timestamp_ms = Some(sample.timestamp_ms);
        if let Some(gravity_sensor) = state {
            samples.push((sample.timestamp_ms, gravity_sensor));
        }
    }
    if samples.len() < MIN_GRAVITY_SAMPLES || innovations.len() < MIN_GRAVITY_SAMPLES {
        return None;
    }
    let rms_innovation_deg = (innovations.iter().map(|angle| angle * angle).sum::<f64>()
        / innovations.len() as f64)
        .sqrt();
    if !rms_innovation_deg.is_finite() || rms_innovation_deg > MAX_GRAVITY_RMS_ANGLE_DEG {
        return None;
    }
    Some(GravityDirectionTimeline {
        samples,
        rms_innovation_deg,
        correction_sample_count: innovations.len(),
    })
}

/// Fuse gyro propagation with accelerometer correction into a downward
/// gravity direction in sensor coordinates. Both gyro sign conventions are
/// evaluated; the one agreeing best with measured acceleration is retained.
pub fn build_gravity_direction_timeline(
    normalized_imu: &[IMUData],
) -> Option<GravityDirectionTimeline> {
    let gravity_magnitude = estimate_gravity_magnitude(normalized_imu)?;
    [-1.0, 1.0]
        .into_iter()
        .filter_map(|gyro_sign| {
            fuse_gravity_candidate(normalized_imu, gravity_magnitude, gyro_sign)
        })
        .max_by(|left, right| {
            left.correction_sample_count
                .cmp(&right.correction_sample_count)
                .then_with(|| right.rms_innovation_deg.total_cmp(&left.rms_innovation_deg))
        })
}

/// Estimate the recording's nominal gravity magnitude after low-pass
/// filtering. Some Insta360 models report a different raw acceleration scale,
/// so quality gates use this robust per-capture baseline instead of assuming
/// exactly 9.80665 m/s².
pub fn estimate_gravity_magnitude(normalized_imu: &[IMUData]) -> Option<f64> {
    let mut ordered = normalized_imu
        .iter()
        .filter(|sample| sample.timestamp_ms.is_finite())
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.timestamp_ms.total_cmp(&right.timestamp_ms));
    let stride = (ordered.len() / 4_096).max(1);
    let mut filtered: Option<[f64; 3]> = None;
    let mut previous_timestamp_ms: Option<f64> = None;
    let mut segment_elapsed_seconds = 0.0;
    let mut magnitudes = Vec::new();
    for (index, sample) in ordered.into_iter().enumerate() {
        let dt = previous_timestamp_ms
            .map(|previous| (sample.timestamp_ms - previous) / 1_000.0)
            .filter(|dt| dt.is_finite() && *dt > 0.0)
            .unwrap_or(0.001);
        if dt * 1_000.0 > GYRO_INTEGRATION_GAP_WARNING_MS {
            filtered = None;
            segment_elapsed_seconds = 0.0;
        }
        if let Some(raw) = sample
            .accl
            .filter(|value| value.iter().all(|axis| axis.is_finite()))
        {
            let alpha =
                (dt / (ACCELERATION_LOWPASS_TIME_CONSTANT_SECONDS + dt)).clamp(0.000_1, 1.0);
            filtered = Some(match filtered {
                Some(previous) => {
                    std::array::from_fn(|axis| previous[axis] * (1.0 - alpha) + raw[axis] * alpha)
                }
                None => raw,
            });
        }
        segment_elapsed_seconds += dt;
        previous_timestamp_ms = Some(sample.timestamp_ms);
        if segment_elapsed_seconds < GRAVITY_FILTER_WARMUP_SECONDS || index % stride != 0 {
            continue;
        }
        if let Some(magnitude) = filtered
            .map(|value| value.iter().map(|axis| axis * axis).sum::<f64>().sqrt())
            .filter(|value| {
                value.is_finite() && (MIN_GRAVITY_MAGNITUDE..=MAX_GRAVITY_MAGNITUDE).contains(value)
            })
        {
            magnitudes.push(magnitude);
        }
    }
    if magnitudes.len() < MIN_GRAVITY_SAMPLES {
        return None;
    }
    magnitudes.sort_by(f64::total_cmp);
    magnitudes.get(magnitudes.len() / 2).copied()
}

/// Match telemetry-parser's DJI normalized-attitude convention. This is still
/// not a COLMAP camera qvec; rotational hand-eye calibration remains mandatory.
fn normalize_dji_attitude(value: DjiTelemetryQuaternion) -> Option<Quaternion> {
    let raw = [
        f64::from(value.w),
        f64::from(value.x),
        f64::from(value.y),
        f64::from(value.z),
    ];
    let camera_basis = multiply_quaternions(raw, [0.5, -0.5, -0.5, 0.5]);
    normalize_quaternion(multiply_quaternions([0.0, 0.0, 1.0, 0.0], camera_basis))
}

#[derive(Debug)]
struct DjiFallbackAttitude {
    proto_file_name: String,
    camera_model: Option<String>,
    attitude_source: String,
    samples: Vec<QuaternionSample>,
}

fn parse_dji_fused_attitude_fallback(
    input_path: &Path,
    source_size: usize,
    cancel_flag: Arc<AtomicBool>,
) -> Result<Option<DjiFallbackAttitude>, String> {
    let mut stream = fs::File::open(input_path).map_err(|error| error.to_string())?;
    let mut proto_file_name = String::new();
    let mut camera_model = None;
    let mut attitude_source = String::new();
    let mut samples = Vec::new();
    let mut first_frame_timestamp_us = None;

    telemetry_parser::util::get_metadata_track_samples(
        &mut stream,
        source_size,
        true,
        |info, data, _, _| {
            let Ok(meta) = DjiTelemetryProductMeta::decode(data) else {
                return;
            };
            if let Some(header) = meta.clip_meta.and_then(|clip| clip.clip_meta_header) {
                if !header.proto_file_name.trim().is_empty() {
                    proto_file_name = header.proto_file_name;
                }
                if !header.product_name.trim().is_empty() {
                    camera_model = Some(header.product_name);
                }
            }
            let Some(frame) = meta.frame_meta else {
                return;
            };
            let frame_timestamp_ms = frame
                .frame_meta_header
                .map(|header| {
                    let first = *first_frame_timestamp_us.get_or_insert(header.frame_timestamp_us);
                    (i128::from(header.frame_timestamp_us) - i128::from(first)) as f64 / 1000.0
                })
                .unwrap_or(info.timestamp_ms);
            let layout = dji_attitude_layout(&proto_file_name);
            let camera_attitude = frame
                .camera_frame_meta
                .as_deref()
                .and_then(|payload| decode_dji_camera_attitude(payload, &proto_file_name));
            let stabilized_camera_attitude = frame
                .product_frame_meta
                .as_deref()
                .and_then(decode_avata360_stabilized_camera_attitude);
            // Avata360 carries a body/IMU high-rate attitude and a distinct
            // camera attitude. A fixed hand-eye transform cannot absorb gimbal
            // motion, so the per-frame camera attitude must take precedence.
            if proto_file_name.to_ascii_lowercase().contains("avata360")
                && stabilized_camera_attitude.is_some()
            {
                let quaternion = stabilized_camera_attitude.and_then(normalize_dji_attitude);
                if let Some(quaternion) = quaternion {
                    attitude_source = "product-frame/stabilization-meta/camera-attitude".to_owned();
                    samples.push(QuaternionSample {
                        timestamp_ms: frame_timestamp_ms,
                        w: quaternion[0],
                        x: quaternion[1],
                        y: quaternion[2],
                        z: quaternion[3],
                    });
                }
            } else if let Some(attitude) = frame
                .imu_frame_meta
                .as_ref()
                .and_then(|imu| decode_dji_device_attitude(imu, layout))
            {
                let count = attitude.attitude.len();
                let duration_ms = if info.duration_ms.is_finite() && info.duration_ms > 0.0 {
                    info.duration_ms
                } else {
                    0.0
                };
                for (index, value) in attitude.attitude.into_iter().enumerate() {
                    let Some(quaternion) = normalize_dji_attitude(value) else {
                        continue;
                    };
                    let offset = (index as f64 - f64::from(attitude.offset)) / count.max(1) as f64
                        * duration_ms;
                    attitude_source = "imu-frame/fused-attitude".to_owned();
                    samples.push(QuaternionSample {
                        timestamp_ms: frame_timestamp_ms + offset,
                        w: quaternion[0],
                        x: quaternion[1],
                        y: quaternion[2],
                        z: quaternion[3],
                    });
                }
            } else if let Some(value) = camera_attitude {
                if let Some(quaternion) = normalize_dji_attitude(value) {
                    attitude_source = "camera-frame/camera-attitude".to_owned();
                    samples.push(QuaternionSample {
                        timestamp_ms: frame_timestamp_ms,
                        w: quaternion[0],
                        x: quaternion[1],
                        y: quaternion[2],
                        z: quaternion[3],
                    });
                }
            }
        },
        cancel_flag.clone(),
    )
    .map_err(|error| error.to_string())?;
    if cancel_flag.load(Ordering::Acquire) {
        return Err("cancelled".to_owned());
    }

    if samples.is_empty() && proto_file_name.trim().is_empty() {
        return Ok(None);
    }
    samples.sort_by(|left, right| left.timestamp_ms.total_cmp(&right.timestamp_ms));
    samples.dedup_by(|left, right| left.timestamp_ms == right.timestamp_ms);
    Ok(Some(DjiFallbackAttitude {
        proto_file_name,
        camera_model,
        attitude_source,
        samples,
    }))
}

#[derive(Clone, PartialEq, Message)]
struct DjiStreamMeta {
    #[prost(message, optional, tag = "6")]
    pano_dewarp_params: Option<DjiPanoDewarpParams>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiPanoDewarpParams {
    #[prost(message, optional, tag = "1")]
    native_refine_slave: Option<DjiDewarpParams>,
    #[prost(message, optional, tag = "2")]
    native_refine_master: Option<DjiDewarpParams>,
    #[prost(message, optional, tag = "11")]
    native_slave: Option<DjiDewarpParams>,
    #[prost(message, optional, tag = "12")]
    native_master: Option<DjiDewarpParams>,
}

#[derive(Clone, PartialEq, Message)]
struct DjiDewarpParams {
    #[prost(float, tag = "3")]
    cx: f32,
    #[prost(float, tag = "4")]
    cy: f32,
    #[prost(float, tag = "10")]
    width: f32,
    #[prost(float, tag = "11")]
    height: f32,
    #[prost(float, repeated, tag = "22")]
    occlusion_pt_x: Vec<f32>,
    #[prost(float, repeated, tag = "23")]
    occlusion_pt_y: Vec<f32>,
}

impl DjiDewarpParams {
    fn optical_occlusion(self) -> Option<OpticalOcclusion> {
        OpticalOcclusion::from_source_pixels(
            self.width,
            self.height,
            self.cx,
            self.cy,
            &self.occlusion_pt_x,
            &self.occlusion_pt_y,
        )
    }
}

fn optical_occlusions_from_pano(params: DjiPanoDewarpParams) -> Option<LensOpticalOcclusions> {
    let lens0 = params
        .native_refine_master
        .or(params.native_master)?
        .optical_occlusion()?;
    let lens1 = params
        .native_refine_slave
        .or(params.native_slave)?
        .optical_occlusion()?;
    Some(LensOpticalOcclusions { lens0, lens1 })
}

/// Read DJI's per-lens native occlusion curves from an OSV container.
///
/// The first video stream is DJI's master lens and becomes `lens0`; the second
/// is the slave lens and becomes `lens1`, matching the extraction stream order.
pub fn read_dji_optical_occlusions(
    input_path: &Path,
) -> Result<Option<LensOpticalOcclusions>, String> {
    let mut stream = fs::File::open(input_path).map_err(|error| error.to_string())?;
    let size = stream.metadata().map_err(|error| error.to_string())?.len() as usize;
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_after_found = cancel.clone();
    let mut found = None;
    telemetry_parser::util::get_metadata_track_samples(
        &mut stream,
        size,
        true,
        |_, data, _, _| {
            if found.is_some() {
                return;
            }
            let Ok(parsed) = DjiProductMeta::decode(data) else {
                return;
            };
            let Some(params) = parsed
                .stream_meta
                .and_then(|stream| stream.pano_dewarp_params)
            else {
                return;
            };
            found = optical_occlusions_from_pano(params);
            if found.is_some() {
                cancel_after_found.store(true, Ordering::Release);
            }
        },
        cancel,
    )
    .map_err(|error| error.to_string())?;
    Ok(found)
}

pub fn parse_and_write(
    input_path: &Path,
    output_path: &Path,
    cancel_flag: Arc<AtomicBool>,
) -> Result<TelemetryExport, String> {
    parse_and_write_with_progress(input_path, output_path, cancel_flag, |_| {})
}

pub fn parse_and_write_with_progress<F>(
    input_path: &Path,
    output_path: &Path,
    cancel_flag: Arc<AtomicBool>,
    progress: F,
) -> Result<TelemetryExport, String>
where
    F: Fn(f64),
{
    let mut stream = fs::File::open(input_path).map_err(|error| error.to_string())?;
    let source_metadata = stream.metadata().map_err(|error| error.to_string())?;
    let source_size = source_metadata.len();
    let source_modified_nanos = source_metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().to_string());
    if let Some(export) =
        existing_export(output_path, source_size, source_modified_nanos.as_deref())
    {
        return Ok(export);
    }
    let size = source_size as usize;
    let input = telemetry_parser::Input::from_stream(
        &mut stream,
        size,
        input_path,
        progress,
        cancel_flag.clone(),
    )
    .map_err(|error| error.to_string())?;
    if cancel_flag.load(Ordering::Acquire) {
        return Err("cancelled".to_owned());
    }

    let mut normalized_imu =
        telemetry_parser::util::normalized_imu(&input, None).map_err(|error| error.to_string())?;
    let mut parser_name = input.parser_name().to_owned();
    let camera_type = input.camera_type();
    let timestamps_accurate = input.has_accurate_timestamps();
    let mut camera_model = input.camera_model().cloned();
    let mut fallback_proto = None;
    let mut fused_attitude = Vec::new();
    let mut invalid_fused_attitude_samples = 0usize;
    for sample in input.samples.iter().flatten() {
        if cancel_flag.load(Ordering::Acquire) {
            return Err("cancelled".to_owned());
        }
        let Some(groups) = sample.tag_map.as_ref() else {
            continue;
        };
        let Some(group) = groups.get(&GroupId::Quaternion) else {
            continue;
        };
        let Some(tag) = group.get(&TagId::Data) else {
            continue;
        };
        if let TagValue::Vec_TimeQuaternion_f64(values) = &tag.value {
            for value in values.get() {
                let sample = QuaternionSample {
                    timestamp_ms: value.t,
                    w: value.v.w,
                    x: value.v.x,
                    y: value.v.y,
                    z: value.v.z,
                };
                if let Some(normalized) = sample.normalized() {
                    fused_attitude.push(normalized);
                } else {
                    invalid_fused_attitude_samples += 1;
                }
            }
        }
    }
    let dji_fallback = if camera_type.eq_ignore_ascii_case("DJI") {
        parse_dji_fused_attitude_fallback(input_path, size, cancel_flag.clone())?
    } else {
        None
    };
    if let Some(fallback) = dji_fallback {
        let schema = fallback.proto_file_name.to_ascii_lowercase();
        let upstream_schema_supported =
            schema.contains("oq101") || schema.contains("wm169") || schema.contains("wa530");
        if !upstream_schema_supported || (normalized_imu.is_empty() && fused_attitude.is_empty()) {
            // telemetry-parser's pinned DJI dispatcher defaults unknown product
            // schemas to WM169. Never accept that ambiguous result: use the
            // schema-aware partial decoder and let hand-eye validation decide
            // whether its attitude is safe for COLMAP.
            if !upstream_schema_supported {
                normalized_imu.clear();
            }
            parser_name = format!(
                "spherealign-dji-protobuf:{}:{}",
                fallback.proto_file_name, fallback.attitude_source
            );
            fallback_proto = Some(fallback.proto_file_name);
            camera_model = fallback.camera_model.or(camera_model);
            fused_attitude = fallback.samples;
        }
    }
    let mut attitude_source = if fused_attitude.is_empty() {
        AttitudeSource::Unavailable
    } else {
        AttitudeSource::NativeFused
    };
    let mut integrated_gyro_attitude = Vec::new();
    let mut gyro_integration_gap = None;
    let mut gyro_timestamp_repairs = None;
    let mut gyro_integration_error = None;
    if attitude_source == AttitudeSource::Unavailable
        && camera_type.eq_ignore_ascii_case("Insta360")
    {
        match integrate_normalized_gyro(&normalized_imu) {
            Ok(integrated) => {
                if integrated.gap_count > 0 {
                    gyro_integration_gap = Some((integrated.gap_count, integrated.max_gap_ms));
                }
                if integrated.non_monotonic_timestamp_count > 0
                    || integrated.duplicate_timestamp_count > 0
                {
                    gyro_timestamp_repairs = Some((
                        integrated.non_monotonic_timestamp_count,
                        integrated.duplicate_timestamp_count,
                    ));
                }
                integrated_gyro_attitude = integrated.samples;
                attitude_source = AttitudeSource::GyroIntegratedRelative;
            }
            Err(error) => gyro_integration_error = Some(error),
        }
    }
    if normalized_imu.is_empty() && fused_attitude.is_empty() && integrated_gyro_attitude.is_empty()
    {
        return Err(
            "supported container was detected, but it contained no usable IMU or fused-attitude samples; no compatible DJI protobuf attitude layout was found"
                .to_owned(),
        );
    }

    let attitude_samples = if fused_attitude.is_empty() {
        &integrated_gyro_attitude
    } else {
        &fused_attitude
    };
    let attitude_diagnostics = validate_attitude_samples(attitude_samples);
    let fused_attitude_rate_hz = validate_attitude_samples(&fused_attitude).rate_hz;
    let integrated_gyro_attitude_rate_hz =
        validate_attitude_samples(&integrated_gyro_attitude).rate_hz;
    let normalized_imu_sample_count = normalized_imu.len();
    let mut warnings = base_telemetry_warnings(
        &parser_name,
        &camera_type,
        normalized_imu_sample_count,
        attitude_source,
    );
    if invalid_fused_attitude_samples > 0 {
        warnings.push("Invalid fused-attitude quaternion samples were dropped.".to_owned());
    }
    if let Some(proto) = fallback_proto {
        warnings.push(format!(
            "Fused attitude was decoded through SphereAlign's schema-aware DJI protobuf fallback ({proto}); rotational hand-eye calibration is still required before COLMAP use."
        ));
    }
    if let Some(error) = gyro_integration_error {
        warnings.push(format!(
            "Normalized gyro could not be integrated into a relative attitude timeline: {error}."
        ));
    }
    if let Some((gap_count, max_gap_ms)) = gyro_integration_gap {
        warnings.push(format!(
            "Integrated gyro contains {gap_count} telemetry gap(s) above {GYRO_INTEGRATION_GAP_WARNING_MS:.1} ms (maximum {max_gap_ms:.3} ms); hand-eye residual validation remains mandatory."
        ));
    }
    if let Some((non_monotonic_count, duplicate_count)) = gyro_timestamp_repairs {
        warnings.push(format!(
            "Gyro integration sorted {non_monotonic_count} non-monotonic timestamp pair(s) and removed {duplicate_count} duplicate timestamp pair(s)."
        ));
    }
    if !attitude_diagnostics.is_monotonic() {
        warnings.push(
            "Fused-attitude timestamps are not monotonic; interpolation will use a sorted view."
                .to_owned(),
        );
    }
    if attitude_diagnostics.discontinuity_count > 0 {
        warnings.push(format!(
            "Fused attitude contains {} step(s) above {} degrees; inspect attitudeDiagnostics before using it for frame selection.",
            attitude_diagnostics.discontinuity_count, DEFAULT_ATTITUDE_DISCONTINUITY_DEG
        ));
    }
    let timebase = describe_timebase(&parser_name, &camera_type, timestamps_accurate);
    let coordinate_frame = describe_coordinate_frame(
        &parser_name,
        &camera_type,
        normalized_imu_sample_count,
        attitude_source,
    );
    let normalized = NormalizedTelemetry {
        schema_version: NORMALIZED_TELEMETRY_SCHEMA_VERSION,
        parser: parser_name,
        parser_revision: PARSER_REVISION.to_owned(),
        camera_type,
        camera_model: camera_model.clone(),
        source_size,
        source_modified_nanos,
        timestamps_accurate,
        sensor_readout_time_ms: input.frame_readout_time(),
        timebase,
        normalized_imu_sample_count,
        normalized_imu,
        fused_attitude_sample_count: fused_attitude.len(),
        fused_attitude_rate_hz,
        fused_attitude,
        integrated_gyro_attitude_sample_count: integrated_gyro_attitude.len(),
        integrated_gyro_attitude_rate_hz,
        integrated_gyro_attitude,
        attitude_source,
        attitude_diagnostics,
        coordinate_frame,
        applied_to_colmap: false,
        warnings,
    };
    write_json_atomic(output_path, &normalized)?;
    Ok(TelemetryExport {
        path: output_path.to_path_buf(),
        camera_model,
        normalized_imu_sample_count,
        fused_attitude_sample_count: normalized.fused_attitude_sample_count,
        integrated_gyro_attitude_sample_count: normalized.integrated_gyro_attitude_sample_count,
    })
}

/// Read a normalized telemetry export written by [`parse_and_write`].
pub fn read_normalized_telemetry(path: &Path) -> Result<NormalizedTelemetry, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid normalized telemetry {}: {error}", path.display()))
}

fn existing_export(
    path: &Path,
    source_size: u64,
    source_modified_nanos: Option<&str>,
) -> Option<TelemetryExport> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    if value.get("schemaVersion")?.as_u64()? != u64::from(NORMALIZED_TELEMETRY_SCHEMA_VERSION)
        || value.get("parserRevision")?.as_str()? != PARSER_REVISION
        || value.get("sourceSize")?.as_u64()? != source_size
        || value
            .get("sourceModifiedNanos")
            .and_then(|value| value.as_str())
            != source_modified_nanos
    {
        return None;
    }
    let normalized_imu_sample_count = value.get("normalizedImuSampleCount")?.as_u64()? as usize;
    let fused_attitude_sample_count = value.get("fusedAttitudeSampleCount")?.as_u64()? as usize;
    let integrated_gyro_attitude_sample_count =
        value.get("integratedGyroAttitudeSampleCount")?.as_u64()? as usize;
    if normalized_imu_sample_count == 0
        && fused_attitude_sample_count == 0
        && integrated_gyro_attitude_sample_count == 0
    {
        return None;
    }
    Some(TelemetryExport {
        path: path.to_path_buf(),
        camera_model: value
            .get("cameraModel")
            .and_then(|value| value.as_str())
            .map(str::to_owned),
        normalized_imu_sample_count,
        fused_attitude_sample_count,
        integrated_gyro_attitude_sample_count,
    })
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "telemetry output has no parent directory".to_owned())?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let partial = path.with_extension("json.partial");
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    let result = (|| -> Result<(), String> {
        fs::write(&partial, bytes).map_err(|error| error.to_string())?;
        // sync_all maps to FlushFileBuffers on Windows, which rejects a
        // read-only handle with os error 5.
        fs::OpenOptions::new()
            .write(true)
            .open(&partial)
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
        rename_replace(&partial, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn rename_replace(temporary: &Path, destination: &Path) -> Result<(), String> {
    match fs::rename(temporary, destination) {
        Ok(()) => Ok(()),
        Err(_error) if destination.exists() => {
            let backup = destination.with_extension("json.backup");
            let _ = fs::remove_file(&backup);
            fs::rename(destination, &backup).map_err(|error| error.to_string())?;
            match fs::rename(temporary, destination) {
                Ok(()) => {
                    let _ = fs::remove_file(backup);
                    Ok(())
                }
                Err(replacement_error) => {
                    let _ = fs::rename(&backup, destination);
                    Err(replacement_error.to_string())
                }
            }
        }
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_insv(path: &Path, output: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_insv(&path, output);
            } else if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("insv"))
            {
                output.push(path);
            }
        }
    }

    fn sample(timestamp_ms: f64, quaternion: Quaternion) -> QuaternionSample {
        QuaternionSample {
            timestamp_ms,
            w: quaternion[0],
            x: quaternion[1],
            y: quaternion[2],
            z: quaternion[3],
        }
    }

    fn dewarp_with_curve() -> DjiDewarpParams {
        DjiDewarpParams {
            cx: 50.0,
            cy: 49.0,
            width: 100.0,
            height: 100.0,
            occlusion_pt_x: vec![20.0, 50.0, 80.0],
            occlusion_pt_y: vec![70.0, 90.0, 70.0],
        }
    }

    #[test]
    fn converts_dji_master_and_slave_curves() {
        let params = DjiPanoDewarpParams {
            native_refine_slave: Some(dewarp_with_curve()),
            native_refine_master: Some(dewarp_with_curve()),
            native_slave: None,
            native_master: None,
        };
        assert!(optical_occlusions_from_pano(params).is_some());
    }

    #[test]
    fn normalizes_quaternions_and_rejects_invalid_values() {
        let normalized = normalize_quaternion([2.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(normalized, [1.0, 0.0, 0.0, 0.0]);
        assert!(normalize_quaternion([0.0, 0.0, 0.0, 0.0]).is_none());
        assert!(normalize_quaternion([f64::NAN, 0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn quaternion_angle_handles_sign_equivalence_and_invalid_inputs() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        let quarter_turn_z = [2.0_f64.sqrt() / 2.0, 0.0, 0.0, 2.0_f64.sqrt() / 2.0];
        assert!((quaternion_angle_deg(identity, quarter_turn_z) - 90.0).abs() < 1e-10);
        assert!(quaternion_angle_deg(identity, identity.map(|component| -component)) < 1e-10);
        assert_eq!(quaternion_angle_deg([0.0; 4], identity), f64::INFINITY);
    }

    #[test]
    fn slerp_interpolates_shortest_path() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        let quarter_turn_z = [2.0_f64.sqrt() / 2.0, 0.0, 0.0, 2.0_f64.sqrt() / 2.0];
        let midpoint = slerp_quaternion(identity, quarter_turn_z, 0.5).unwrap();
        assert!((quaternion_angle_deg(identity, midpoint) - 45.0).abs() < 1e-10);
        assert!(slerp_quaternion(identity, quarter_turn_z, f64::NAN).is_none());
        assert!(slerp_quaternion(identity, quarter_turn_z, 1.1).is_none());
    }

    #[test]
    fn diagnostics_report_bad_timestamps_quaternions_and_jumps() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        let quarter_turn_z = [2.0_f64.sqrt() / 2.0, 0.0, 0.0, 2.0_f64.sqrt() / 2.0];
        let three_quarter_turn_z = [-2.0_f64.sqrt() / 2.0, 0.0, 0.0, 2.0_f64.sqrt() / 2.0];
        let samples = vec![
            sample(0.0, identity),
            sample(100.0, quarter_turn_z),
            sample(50.0, quarter_turn_z),
            sample(50.0, [0.0; 4]),
            sample(200.0, three_quarter_turn_z),
        ];
        let diagnostics = validate_attitude_samples(&samples);
        assert_eq!(diagnostics.sample_count, 5);
        assert_eq!(diagnostics.valid_sample_count, 4);
        assert_eq!(diagnostics.invalid_quaternion_count, 1);
        assert_eq!(diagnostics.non_monotonic_timestamp_count, 1);
        assert_eq!(diagnostics.duplicate_timestamp_count, 1);
        assert_eq!(diagnostics.first_timestamp_ms, Some(0.0));
        assert_eq!(diagnostics.last_timestamp_ms, Some(200.0));
        assert!(diagnostics
            .coverage_duration_ms
            .is_some_and(|value| (value - 200.0).abs() < 1e-10));
        assert!(diagnostics.discontinuity_count >= 1);
        let timeline = AttitudeTimeline::new(&samples);
        assert!(timeline.contains_timestamp(100.0));
        assert!(!timeline.contains_timestamp(201.0));
    }

    #[test]
    fn timeline_sorts_recoverable_samples_and_interpolates() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        let quarter_turn_z = [2.0_f64.sqrt() / 2.0, 0.0, 0.0, 2.0_f64.sqrt() / 2.0];
        let timeline =
            AttitudeTimeline::new(&[sample(100.0, quarter_turn_z), sample(0.0, identity)]);
        assert_eq!(timeline.samples().first().unwrap().timestamp_ms, 0.0);
        assert_eq!(timeline.samples().last().unwrap().timestamp_ms, 100.0);
        let midpoint = timeline.interpolate(50.0).unwrap();
        assert!((quaternion_angle_deg(identity, midpoint.quaternion()) - 45.0).abs() < 1e-10);
        assert!(timeline.interpolate(-1.0).is_none());
        assert!(timeline.interpolate(101.0).is_none());

        let duplicates = AttitudeTimeline::new(&[
            sample(0.0, identity),
            sample(0.0, quarter_turn_z),
            sample(100.0, quarter_turn_z),
        ]);
        assert_eq!(duplicates.samples().len(), 2);
        assert!(
            quaternion_angle_deg(identity, duplicates.interpolate(0.0).unwrap().quaternion())
                > 89.9
        );
    }

    #[test]
    fn relative_gyro_timeline_does_not_interpolate_across_gaps() {
        let timeline = AttitudeTimeline::new_with_max_interpolation_gap(
            &[
                sample(0.0, [1.0, 0.0, 0.0, 0.0]),
                sample(200.0, [1.0, 0.0, 0.0, 0.0]),
            ],
            Some(GYRO_INTEGRATION_GAP_WARNING_MS),
        );
        assert!(timeline.interpolate(0.0).is_some());
        assert!(timeline.interpolate(100.0).is_none());
        assert!(timeline.interpolate(200.0).is_some());
    }

    #[test]
    fn normalized_export_round_trips_and_exposes_timeline() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("telemetry.json");
        let source = NormalizedTelemetry {
            schema_version: NORMALIZED_TELEMETRY_SCHEMA_VERSION,
            parser: "test".to_owned(),
            parser_revision: PARSER_REVISION.to_owned(),
            camera_type: "test".to_owned(),
            camera_model: Some("test".to_owned()),
            source_size: 1,
            source_modified_nanos: None,
            timestamps_accurate: true,
            sensor_readout_time_ms: None,
            timebase: "milliseconds".to_owned(),
            normalized_imu_sample_count: 0,
            normalized_imu: Vec::new(),
            fused_attitude_sample_count: 2,
            fused_attitude_rate_hz: Some(10.0),
            fused_attitude: vec![
                sample(0.0, [1.0, 0.0, 0.0, 0.0]),
                sample(100.0, [1.0, 0.0, 0.0, 0.0]),
            ],
            integrated_gyro_attitude_sample_count: 0,
            integrated_gyro_attitude_rate_hz: None,
            integrated_gyro_attitude: Vec::new(),
            attitude_source: AttitudeSource::NativeFused,
            attitude_diagnostics: AttitudeDiagnostics::default(),
            coordinate_frame: "test".to_owned(),
            applied_to_colmap: false,
            warnings: Vec::new(),
        };
        fs::write(&path, serde_json::to_vec(&source).unwrap()).unwrap();
        let loaded = read_normalized_telemetry(&path).unwrap();
        assert_eq!(loaded.fused_attitude, source.fused_attitude);
        assert_eq!(
            loaded.attitude_timeline().diagnostics().valid_sample_count,
            2
        );
    }

    #[test]
    fn integrates_normalized_gyro_as_relative_attitude() {
        let imu = [
            IMUData {
                timestamp_ms: 0.0,
                gyro: Some([0.0, 0.0, 450.0]),
                ..IMUData::default()
            },
            IMUData {
                timestamp_ms: 100.0,
                gyro: Some([0.0, 0.0, 450.0]),
                ..IMUData::default()
            },
        ];
        let integrated = integrate_normalized_gyro(&imu).unwrap();
        assert_eq!(integrated.samples.len(), 2);
        assert!(
            (quaternion_angle_deg(
                integrated.samples[0].quaternion(),
                integrated.samples[1].quaternion()
            ) - 45.0)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn fuses_acceleration_into_downward_sensor_gravity() {
        let half = std::f64::consts::FRAC_PI_4;
        let attitude = [half.cos(), half.sin(), 0.0, 0.0];
        let inverse = [attitude[0], -attitude[1], -attitude[2], -attitude[3]];
        let up_sensor = rotate_vector(inverse, [0.0, 0.0, 1.0]).unwrap();
        let imu = (0..=200)
            .map(|index| IMUData {
                timestamp_ms: index as f64 * 10.0,
                gyro: Some([0.0, 0.0, 0.0]),
                accl: Some(up_sensor.map(|value| value * 9.80665)),
                ..IMUData::default()
            })
            .collect::<Vec<_>>();
        let timeline = build_gravity_direction_timeline(&imu).unwrap();
        let estimate = timeline.interpolate(1_900.0).unwrap();
        let expected_down = up_sensor.map(|value| -value);
        assert!(vector_angle_deg(estimate, expected_down).unwrap() < 1e-8);
        assert!(timeline.sample_count() >= 50);
        assert!(timeline.rms_innovation_deg < 1e-8);
    }

    #[test]
    fn gravity_fusion_lowpasses_high_frequency_acceleration_noise() {
        let imu = (0..=2_000)
            .map(|index| {
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                IMUData {
                    timestamp_ms: index as f64,
                    gyro: Some([0.0, 0.0, 0.0]),
                    accl: Some([12.0 * sign, -8.0 * sign, 9.80665 + 5.0 * sign]),
                    magn: None,
                }
            })
            .collect::<Vec<_>>();
        let timeline = build_gravity_direction_timeline(&imu).unwrap();
        let gravity = timeline.interpolate(1_900.0).unwrap();
        assert!(vector_angle_deg(gravity, [0.0, 0.0, -1.0]).unwrap() < 1.0);
    }

    #[test]
    fn gyro_integration_rejects_unbounded_gaps() {
        let imu = [
            IMUData {
                timestamp_ms: 0.0,
                gyro: Some([1.0, 0.0, 0.0]),
                ..IMUData::default()
            },
            IMUData {
                timestamp_ms: MAX_GYRO_INTEGRATION_GAP_MS + 1.0,
                gyro: Some([1.0, 0.0, 0.0]),
                ..IMUData::default()
            },
        ];
        assert!(integrate_normalized_gyro(&imu)
            .unwrap_err()
            .contains("maximum supported gap"));
    }

    #[test]
    fn normalized_cache_rejects_older_semantic_schema() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("telemetry.json");
        let payload = |schema_version| {
            serde_json::json!({
                "schemaVersion": schema_version,
                "parserRevision": PARSER_REVISION,
                "sourceSize": 42,
                "sourceModifiedNanos": "123",
                "normalizedImuSampleCount": 0,
                "fusedAttitudeSampleCount": 1,
                "integratedGyroAttitudeSampleCount": 0
            })
        };
        fs::write(&path, serde_json::to_vec(&payload(1)).unwrap()).unwrap();
        assert!(existing_export(&path, 42, Some("123")).is_none());
        fs::write(
            &path,
            serde_json::to_vec(&payload(NORMALIZED_TELEMETRY_SCHEMA_VERSION)).unwrap(),
        )
        .unwrap();
        assert!(existing_export(&path, 42, Some("123")).is_some());
    }

    #[test]
    fn insta360_descriptions_do_not_claim_dji_or_fused_attitude() {
        let timebase = describe_timebase("Insta360", "Insta360", true);
        let coordinate_frame = describe_coordinate_frame(
            "Insta360",
            "Insta360",
            10,
            AttitudeSource::GyroIntegratedRelative,
        );
        let warnings = base_telemetry_warnings(
            "Insta360",
            "Insta360",
            10,
            AttitudeSource::GyroIntegratedRelative,
        );

        assert!(timebase.contains("first frame/gyro timestamp"));
        assert!(!timebase.contains("DJI"));
        assert!(!timebase.contains("OSV"));
        assert!(coordinate_frame.contains("normalized gyro integrated"));
        assert!(coordinate_frame.contains("relative rotation only"));
        assert!(!coordinate_frame.contains("DJI"));
        assert!(!coordinate_frame.contains("OSV"));
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("relative rotation and hand-eye calibration")));
        assert!(warnings.iter().all(|warning| !warning.contains("DJI")));
        assert!(warnings.iter().all(|warning| !warning.contains("OSV")));
    }

    #[test]
    fn coordinate_frame_requires_transform_when_attitude_exists() {
        let coordinate_frame =
            describe_coordinate_frame("dvtm_oq101", "DJI", 4, AttitudeSource::NativeFused);
        let warnings = base_telemetry_warnings("dvtm_oq101", "DJI", 4, AttitudeSource::NativeFused);

        assert!(coordinate_frame.contains("fused attitude"));
        assert!(coordinate_frame.contains("sensor-to-camera transform is unverified"));
        assert!(coordinate_frame.contains("not a COLMAP camera qvec"));
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("fused attitude as a COLMAP prior")));
    }

    #[test]
    fn dji_schema_dispatch_covers_known_product_families() {
        assert_eq!(
            dji_attitude_layout("dvtm_oq101.proto"),
            DjiAttitudeLayout::MultiTag2
        );
        assert_eq!(
            dji_attitude_layout("dvtm_AVATA360.proto"),
            DjiAttitudeLayout::MultiTag2
        );
        assert_eq!(
            dji_attitude_layout("dvtm_eagle4_wa530.proto"),
            DjiAttitudeLayout::SingleTag4
        );
        assert_eq!(
            dji_attitude_layout("dvtm_wm169.proto"),
            DjiAttitudeLayout::SingleTag2
        );
        assert_eq!(
            dji_attitude_layout("dvtm_future_camera.proto"),
            DjiAttitudeLayout::Auto
        );
    }

    #[test]
    fn dji_multi_attitude_payload_decodes_without_product_generated_code() {
        let expected = DjiTelemetryQuaternion {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let payload = DjiTelemetryDeviceMultiAttitude {
            current_frame: Some(DjiTelemetryDeviceAttitude {
                timestamp: 1,
                vsync: 2,
                attitude: vec![expected],
                offset: 0.0,
            }),
        }
        .encode_to_vec();
        let imu = DjiTelemetryImuFrameMeta {
            attitude_tag2: Some(payload),
            attitude_tag4: None,
        };
        let decoded = decode_dji_device_attitude(&imu, DjiAttitudeLayout::MultiTag2).unwrap();
        assert_eq!(decoded.attitude, vec![expected]);
    }

    #[test]
    fn avata360_stabilized_camera_attitude_decodes_from_product_frame() {
        let expected = DjiTelemetryQuaternion {
            w: 0.5,
            x: -0.5,
            y: 0.5,
            z: -0.5,
        };
        let payload = DjiAvata360ProductFrameMeta {
            gps: None,
            relative_altitude: None,
            stabilization_meta: Some(DjiAvata360StabilizationMeta {
                camera_attitude: Some(expected),
            }),
        }
        .encode_to_vec();
        assert_eq!(
            decode_avata360_stabilized_camera_attitude(&payload),
            Some(expected)
        );
    }

    #[test]
    #[ignore = "requires GS360_TEST_OSV"]
    fn parses_real_osmo_optical_occlusions() {
        let source = PathBuf::from(std::env::var("GS360_TEST_OSV").expect("GS360_TEST_OSV"));
        assert!(read_dji_optical_occlusions(&source).unwrap().is_some());
    }

    #[test]
    #[ignore = "requires GS360_TEST_OSV; diagnostic for new DJI product schemas"]
    fn inspects_real_dji_proto_layout() {
        fn read_varint(data: &[u8], cursor: &mut usize) -> Option<u64> {
            let mut value = 0u64;
            for shift in (0..64).step_by(7) {
                let byte = *data.get(*cursor)?;
                *cursor += 1;
                value |= u64::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    return Some(value);
                }
            }
            None
        }

        fn collect_quaternion_paths(
            data: &[u8],
            prefix: &str,
            depth: usize,
            paths: &mut std::collections::BTreeMap<String, usize>,
        ) {
            if depth == 0 {
                return;
            }
            let mut cursor = 0usize;
            while cursor < data.len() {
                let Some(key) = read_varint(data, &mut cursor) else {
                    return;
                };
                let field = (key >> 3) as u32;
                match key & 7 {
                    0 => {
                        if read_varint(data, &mut cursor).is_none() {
                            return;
                        }
                    }
                    1 => cursor = cursor.saturating_add(8),
                    2 => {
                        let Some(length) = read_varint(data, &mut cursor) else {
                            return;
                        };
                        let end = cursor.saturating_add(length as usize);
                        let Some(payload) = data.get(cursor..end) else {
                            return;
                        };
                        let path = if prefix.is_empty() {
                            field.to_string()
                        } else {
                            format!("{prefix}.{field}")
                        };
                        if let Ok(value) = DjiTelemetryQuaternion::decode(payload) {
                            let norm = (value.w * value.w
                                + value.x * value.x
                                + value.y * value.y
                                + value.z * value.z)
                                .sqrt();
                            if norm.is_finite() && (0.9..=1.1).contains(&norm) {
                                *paths.entry(path.clone()).or_default() += 1;
                            }
                        }
                        collect_quaternion_paths(payload, &path, depth - 1, paths);
                        cursor = end;
                    }
                    5 => cursor = cursor.saturating_add(4),
                    _ => return,
                }
            }
        }

        let source = PathBuf::from(std::env::var("GS360_TEST_OSV").expect("GS360_TEST_OSV"));
        let mut stream = fs::File::open(&source).unwrap();
        let size = stream.metadata().unwrap().len() as usize;
        let mut clip_count = 0usize;
        let mut frame_count = 0usize;
        let mut callback_count = 0usize;
        let mut decode_error_count = 0usize;
        let mut imu_tag2_count = 0usize;
        let mut imu_tag4_count = 0usize;
        let mut camera_attitude_count = 0usize;
        let mut gps_count = 0usize;
        let mut relative_altitude_count = 0usize;
        let mut camera_quaternion_paths = std::collections::BTreeMap::new();
        telemetry_parser::util::get_metadata_track_samples(
            &mut stream,
            size,
            true,
            |_, data, _, _| {
                callback_count += 1;
                collect_quaternion_paths(
                    data,
                    "root",
                    6,
                    &mut camera_quaternion_paths,
                );
                let meta = match DjiTelemetryProductMeta::decode(data) {
                    Ok(meta) => meta,
                    Err(error) => {
                        decode_error_count += 1;
                        if decode_error_count == 1 {
                            println!("first decode error={error}; bytes={:02x?}", &data[..data.len().min(32)]);
                        }
                        return;
                    }
                };
                if let Some(header) = meta.clip_meta.and_then(|clip| clip.clip_meta_header) {
                    clip_count += 1;
                    println!("schema={} model={}", header.proto_file_name, header.product_name);
                }
                if let Some(frame) = meta.frame_meta {
                    frame_count += 1;
                    if let Some(camera) = frame.camera_frame_meta {
                        collect_quaternion_paths(
                            &camera,
                            "camera",
                            3,
                            &mut camera_quaternion_paths,
                        );
                        camera_attitude_count += usize::from(
                            decode_dji_camera_attitude(camera.as_slice(), "dvtm_AVATA360.proto")
                                .is_some(),
                        );
                    }
                    if let Some(imu) = frame.imu_frame_meta {
                        imu_tag2_count += usize::from(imu.attitude_tag2.is_some());
                        imu_tag4_count += usize::from(imu.attitude_tag4.is_some());
                    }
                    if let Some(product) = frame.product_frame_meta {
                        if let Ok(product) = DjiAvata360ProductFrameMeta::decode(product.as_slice())
                        {
                            if let Some(gps) = product.gps {
                                if gps_count == 0 {
                                    println!(
                                        "first gps coordinates={:?} altitudeMm={} status={} altitudeType={} hasTime={}",
                                        gps.coordinates.map(|coordinates| (
                                            coordinates.unit,
                                            coordinates.latitude,
                                            coordinates.longitude
                                        )),
                                        gps.altitude_mm,
                                        gps.status,
                                        gps.altitude_type,
                                        gps.has_gps_time
                                    );
                                }
                                gps_count += 1;
                            }
                            if let Some(relative) = product.relative_altitude {
                                if relative_altitude_count == 0 {
                                    println!(
                                        "first relative altitude={}mm valid={}",
                                        relative.altitude_mm, relative.valid
                                    );
                                }
                                relative_altitude_count += 1;
                            }
                        }
                    }
                }
            },
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        println!(
            "callbacks={callback_count} decodeErrors={decode_error_count} clip={clip_count} frame={frame_count} imu2={imu_tag2_count} imu4={imu_tag4_count} cameraQ={camera_attitude_count} gps={gps_count} relativeAltitude={relative_altitude_count}"
        );
        println!("camera quaternion paths={camera_quaternion_paths:?}");
        assert!(clip_count > 0);
        assert!(frame_count > 0);
    }

    #[test]
    #[ignore = "requires GS360_TEST_OSV to point to a real supported DJI capture"]
    fn parses_real_dji_capture_and_resumes() {
        let source = PathBuf::from(std::env::var("GS360_TEST_OSV").expect("GS360_TEST_OSV"));
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("telemetry.json");
        let first = parse_and_write(&source, &output, Arc::new(AtomicBool::new(false))).unwrap();
        assert!(first
            .camera_model
            .as_deref()
            .is_some_and(|model| !model.is_empty()));
        assert!(first.fused_attitude_sample_count > 0);
        let normalized = read_normalized_telemetry(&output).unwrap();
        if normalized
            .camera_model
            .as_deref()
            .is_some_and(|model| model.contains("Avata360"))
        {
            assert!(normalized.parser.contains("dvtm_AVATA360.proto"));
            assert!(normalized
                .parser
                .contains("stabilization-meta/camera-attitude"));
        }
        println!(
            "first attitude={:?}, last attitude={:?}",
            normalized.fused_attitude.first(),
            normalized.fused_attitude.last()
        );
        assert_eq!(
            normalized
                .attitude_diagnostics
                .non_monotonic_timestamp_count,
            0
        );
        assert_eq!(normalized.attitude_diagnostics.invalid_quaternion_count, 0);
        assert!(normalized
            .attitude_diagnostics
            .coverage_duration_ms
            .is_some_and(|duration| duration > 1_000.0));
        let second = parse_and_write(&source, &output, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(
            first.fused_attitude_sample_count,
            second.fused_attitude_sample_count
        );
        println!(
            "model: {:?}, normalized IMU: {}, fused attitude: {}, diagnostics: {:?}",
            first.camera_model,
            first.normalized_imu_sample_count,
            first.fused_attitude_sample_count,
            normalized.attitude_diagnostics
        );
    }

    #[test]
    #[ignore = "requires GS360_TEST_INSTA_DIR"]
    fn parses_real_insta360_imu_as_relative_attitude() {
        let root = PathBuf::from(
            std::env::var("GS360_TEST_INSTA_DIR").expect("GS360_TEST_INSTA_DIR is required"),
        );
        let mut sources = Vec::new();
        collect_insv(&root, &mut sources);
        sources.sort();
        let temp = tempfile::tempdir().unwrap();
        let mut parsed = Vec::new();
        for (index, source) in sources.iter().enumerate() {
            let output = temp.path().join(format!("telemetry-{index}.json"));
            let Ok(export) = parse_and_write(source, &output, Arc::new(AtomicBool::new(false)))
            else {
                continue;
            };
            let normalized = read_normalized_telemetry(&output).unwrap();
            assert_eq!(normalized.camera_type, "Insta360");
            assert!(export.normalized_imu_sample_count > 0);
            assert!(
                export.integrated_gyro_attitude_sample_count > 0,
                "model {:?} did not produce relative attitude: {:?}",
                export.camera_model,
                normalized.warnings
            );
            assert_eq!(
                normalized.attitude_source,
                AttitudeSource::GyroIntegratedRelative
            );
            assert_eq!(normalized.fused_attitude_sample_count, 0);
            assert!(!normalized.attitude_source.has_absolute_reference());
            let mut acceleration_norms = normalized
                .normalized_imu
                .iter()
                .filter_map(|sample| sample.accl)
                .map(|value| value.iter().map(|axis| axis * axis).sum::<f64>().sqrt())
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            acceleration_norms.sort_by(f64::total_cmp);
            let acceleration_median = acceleration_norms
                .get(acceleration_norms.len() / 2)
                .copied();
            assert!(
                build_gravity_direction_timeline(&normalized.normalized_imu).is_some(),
                "model {:?} did not produce a gravity estimate; accel count {}, median {:?}",
                export.camera_model,
                acceleration_norms.len(),
                acceleration_median,
            );
            assert!(!normalized.timebase.contains("DJI"));
            assert!(!normalized.coordinate_frame.contains("DJI"));
            parsed.push(export.camera_model);
        }
        assert!(parsed.len() >= 6, "parsed models: {parsed:?}");
        assert!(parsed
            .iter()
            .any(|model| model.as_deref() == Some("Insta360 X3")));
        assert!(parsed
            .iter()
            .any(|model| model.as_deref() == Some("Insta360 X4")));
        assert!(parsed
            .iter()
            .any(|model| model.as_deref() == Some("Insta360 X5")));
        assert!(parsed
            .iter()
            .any(|model| model.as_deref() == Some("Insta360 X6")));
        assert!(parsed
            .iter()
            .any(|model| model.as_deref() == Some("Insta360 OneRS")));
    }
}
