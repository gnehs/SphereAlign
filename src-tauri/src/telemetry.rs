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

const PARSER_REVISION: &str = "77a3b810a0e0f64688a90546c5aaf24c9dba00bd";
const NORMALIZED_TELEMETRY_SCHEMA_VERSION: u32 = 2;

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

/// A normalized, timestamp-ordered view of fused attitude samples.
///
/// The source order is retained in [`AttitudeDiagnostics`] so a caller can
/// detect malformed timestamps, while interpolation uses a sorted copy and
/// therefore remains safe for otherwise recoverable telemetry.
#[derive(Debug, Clone)]
pub struct AttitudeTimeline {
    samples: Vec<QuaternionSample>,
    diagnostics: AttitudeDiagnostics,
}

impl AttitudeTimeline {
    pub fn new(samples: &[QuaternionSample]) -> Self {
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
    #[serde(default)]
    pub attitude_diagnostics: AttitudeDiagnostics,
    pub coordinate_frame: String,
    pub applied_to_colmap: bool,
    pub warnings: Vec<String>,
}

impl NormalizedTelemetry {
    pub fn attitude_timeline(&self) -> AttitudeTimeline {
        AttitudeTimeline::new(&self.fused_attitude)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryExport {
    pub path: PathBuf,
    pub camera_model: Option<String>,
    pub normalized_imu_sample_count: usize,
    pub fused_attitude_sample_count: usize,
}

#[derive(Clone, PartialEq, Message)]
struct DjiProductMeta {
    #[prost(message, optional, tag = "2")]
    stream_meta: Option<DjiStreamMeta>,
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
        |_| {},
        cancel_flag.clone(),
    )
    .map_err(|error| error.to_string())?;
    if cancel_flag.load(Ordering::Acquire) {
        return Err("cancelled".to_owned());
    }

    let normalized_imu =
        telemetry_parser::util::normalized_imu(&input, None).map_err(|error| error.to_string())?;
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
    if normalized_imu.is_empty() && fused_attitude.is_empty() {
        return Err("supported container was detected, but it contained no usable IMU or fused-attitude samples".to_owned());
    }

    let attitude_diagnostics = validate_attitude_samples(&fused_attitude);
    let fused_attitude_rate_hz = attitude_diagnostics.rate_hz;
    let camera_model = input.camera_model().cloned();
    let normalized_imu_sample_count = normalized_imu.len();
    let mut warnings = vec![
        "Raw OSV data streams remain the source of truth.".to_owned(),
        "A verified sensor-to-camera transform is required before using attitude as a COLMAP prior."
            .to_owned(),
    ];
    if invalid_fused_attitude_samples > 0 {
        warnings.push("Invalid fused-attitude quaternion samples were dropped.".to_owned());
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
    let normalized = NormalizedTelemetry {
        schema_version: NORMALIZED_TELEMETRY_SCHEMA_VERSION,
        parser: input.parser_name().to_owned(),
        parser_revision: PARSER_REVISION.to_owned(),
        camera_type: input.camera_type(),
        camera_model: camera_model.clone(),
        source_size,
        source_modified_nanos,
        timestamps_accurate: input.has_accurate_timestamps(),
        sensor_readout_time_ms: input.frame_readout_time(),
        timebase:
            "milliseconds relative to the first DJI metadata frame; leading samples may be negative"
                .to_owned(),
        normalized_imu_sample_count,
        normalized_imu,
        fused_attitude_sample_count: fused_attitude.len(),
        fused_attitude_rate_hz,
        fused_attitude,
        attitude_diagnostics,
        coordinate_frame: "telemetry-parser DJI normalized attitude; not a COLMAP camera qvec"
            .to_owned(),
        applied_to_colmap: false,
        warnings,
    };
    write_json_atomic(output_path, &normalized)?;
    Ok(TelemetryExport {
        path: output_path.to_path_buf(),
        camera_model,
        normalized_imu_sample_count,
        fused_attitude_sample_count: normalized.fused_attitude_sample_count,
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
    if normalized_imu_sample_count == 0 && fused_attitude_sample_count == 0 {
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
                "fusedAttitudeSampleCount": 1
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
    #[ignore = "requires GS360_TEST_OSV"]
    fn parses_real_osmo_optical_occlusions() {
        let source = PathBuf::from(std::env::var("GS360_TEST_OSV").expect("GS360_TEST_OSV"));
        assert!(read_dji_optical_occlusions(&source).unwrap().is_some());
    }

    #[test]
    #[ignore = "requires GS360_TEST_OSV to point to a real supported capture"]
    fn parses_real_osmo_360_capture_and_resumes() {
        let source = PathBuf::from(std::env::var("GS360_TEST_OSV").expect("GS360_TEST_OSV"));
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("telemetry.json");
        let first = parse_and_write(&source, &output, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(first.camera_model.as_deref(), Some("Osmo 360"));
        assert!(first.fused_attitude_sample_count > 0);
        let second = parse_and_write(&source, &output, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(
            first.fused_attitude_sample_count,
            second.fused_attitude_sample_count
        );
        println!(
            "normalized IMU: {}, fused attitude: {}",
            first.normalized_imu_sample_count, first.fused_attitude_sample_count
        );
    }
}
