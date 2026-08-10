//! Time-offset and rotational hand-eye calibration for visual/telemetry poses.
//!
//! This module deliberately does not guess a calibration when the input does
//! not contain enough motion.  The time offset is estimated from the norm of
//! angular velocity (which is invariant to a fixed sensor-to-camera rotation),
//! then the fixed rotation is estimated from the rotational hand-eye equation
//! `A X = X B`.  The output is a serializable model, but callers must still
//! decide whether a valid model is safe to use as a COLMAP prior.
//!
//! The relative-rotation convention used here is:
//!
//! ```text
//! A = visual_i * inverse(visual_j)
//! B = telemetry_i * inverse(telemetry_j)
//! A * X = X * B
//! ```
//!
//! This is the conjugation form of a fixed change of rotation basis.  The
//! returned quaternion is therefore a `camera_from_sensor` basis transform,
//! not a COLMAP camera `qvec`.

use crate::telemetry::{
    normalize_quaternion, quaternion_angle_deg, AttitudeTimeline, Quaternion, QuaternionSample,
};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const IMU_CALIBRATION_SCHEMA_VERSION: u32 = 1;

const DEFAULT_OFFSET_MIN_MS: f64 = -500.0;
const DEFAULT_OFFSET_MAX_MS: f64 = 500.0;
const DEFAULT_OFFSET_STEP_MS: f64 = 5.0;
const DEFAULT_MIN_SAMPLES: usize = 8;
const DEFAULT_MIN_EXCITATION_DEG: f64 = 45.0;
const DEFAULT_MIN_CORRELATION: f64 = 0.35;
const DEFAULT_MIN_AXIS_DIVERSITY: f64 = 0.15;
const DEFAULT_MAX_RESIDUAL_DEG: f64 = 8.0;
const DEFAULT_MAX_OFFSET_CANDIDATES: usize = 20_001;

/// A timestamped visual absolute orientation in scalar-first `(w, x, y, z)`
/// order.  Relative visual motions can be supplied directly to
/// [`solve_rotational_hand_eye`] as [`RotationPair`] values.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VisualRotationSample {
    pub timestamp_ms: f64,
    pub rotation_wxyz: Quaternion,
}

/// A pair of relative rotations for the hand-eye equation `A X = X B`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RotationPair {
    pub visual_relative_wxyz: Quaternion,
    pub telemetry_relative_wxyz: Quaternion,
}

/// Configuration for offset search and rotational hand-eye validation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationConfig {
    pub offset_min_ms: f64,
    pub offset_max_ms: f64,
    pub offset_step_ms: f64,
    pub min_samples: usize,
    pub min_excitation_deg: f64,
    pub min_correlation: f64,
    pub min_axis_diversity: f64,
    pub max_residual_deg: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            offset_min_ms: DEFAULT_OFFSET_MIN_MS,
            offset_max_ms: DEFAULT_OFFSET_MAX_MS,
            offset_step_ms: DEFAULT_OFFSET_STEP_MS,
            min_samples: DEFAULT_MIN_SAMPLES,
            min_excitation_deg: DEFAULT_MIN_EXCITATION_DEG,
            min_correlation: DEFAULT_MIN_CORRELATION,
            min_axis_diversity: DEFAULT_MIN_AXIS_DIVERSITY,
            max_residual_deg: DEFAULT_MAX_RESIDUAL_DEG,
        }
    }
}

impl CalibrationConfig {
    fn validated(self) -> Result<Self, CalibrationError> {
        if !self.offset_min_ms.is_finite()
            || !self.offset_max_ms.is_finite()
            || !self.offset_step_ms.is_finite()
            || self.offset_step_ms <= 0.0
            || self.offset_min_ms > self.offset_max_ms
        {
            return Err(CalibrationError::InvalidConfiguration(
                "offset search bounds/step are invalid",
            ));
        }
        if self.min_samples < 2 {
            return Err(CalibrationError::InvalidConfiguration(
                "min_samples must be at least two",
            ));
        }
        if !self.min_excitation_deg.is_finite()
            || self.min_excitation_deg < 0.0
            || !self.min_correlation.is_finite()
            || !(-1.0..=1.0).contains(&self.min_correlation)
            || !self.min_axis_diversity.is_finite()
            || !(0.0..=1.0).contains(&self.min_axis_diversity)
            || !self.max_residual_deg.is_finite()
            || self.max_residual_deg < 0.0
        {
            return Err(CalibrationError::InvalidConfiguration(
                "calibration thresholds are invalid",
            ));
        }
        Ok(self)
    }
}

/// A successful angular-speed correlation result.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TimeOffsetEstimate {
    pub time_offset_ms: f64,
    pub correlation: f64,
    pub paired_samples: usize,
    pub visual_excitation_deg: f64,
    pub telemetry_excitation_deg: f64,
    pub coverage_start_ms: f64,
    pub coverage_end_ms: f64,
    pub valid: bool,
}

/// A successful rotational hand-eye result.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HandEyeEstimate {
    pub sensor_to_camera_quaternion: Quaternion,
    pub residual_deg: f64,
    pub max_residual_deg: f64,
    pub paired_samples: usize,
    pub excitation_deg: f64,
    pub axis_diversity: f64,
    pub valid: bool,
}

/// Which interpretation of the telemetry absolute quaternion produced the
/// lower hand-eye residual.  `AsProvided` is the normal DJI parser output;
/// `Inverted` is evaluated as a guard against a source that exports the
/// opposite world/sensor convention.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum TelemetryOrientationConvention {
    AsProvided,
    Inverted,
}

/// JSON-compatible calibration model written by a caller after calibration.
/// `valid` is false for an explicitly constructed invalid model; the normal
/// estimation APIs return a [`CalibrationError`] instead of silently creating
/// such a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationModel {
    pub schema_version: u32,
    pub valid: bool,
    pub time_offset_ms: Option<f64>,
    pub sensor_to_camera_quaternion: Option<Quaternion>,
    pub residual_deg: Option<f64>,
    pub max_residual_deg: Option<f64>,
    pub correlation: Option<f64>,
    pub coverage_start_ms: Option<f64>,
    pub coverage_end_ms: Option<f64>,
    pub paired_sample_count: usize,
    pub visual_excitation_deg: Option<f64>,
    pub telemetry_excitation_deg: Option<f64>,
    pub hand_eye_excitation_deg: Option<f64>,
    pub axis_diversity: Option<f64>,
    pub telemetry_orientation_convention: Option<TelemetryOrientationConvention>,
    pub reason: Option<String>,
}

impl CalibrationModel {
    /// Construct a safe, explicitly invalid model for persistence or UI.
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self {
            schema_version: IMU_CALIBRATION_SCHEMA_VERSION,
            valid: false,
            time_offset_ms: None,
            sensor_to_camera_quaternion: None,
            residual_deg: None,
            max_residual_deg: None,
            correlation: None,
            coverage_start_ms: None,
            coverage_end_ms: None,
            paired_sample_count: 0,
            visual_excitation_deg: None,
            telemetry_excitation_deg: None,
            hand_eye_excitation_deg: None,
            axis_diversity: None,
            telemetry_orientation_convention: None,
            reason: Some(reason.into()),
        }
    }
}

/// Reasons for rejecting an offset or hand-eye calibration.
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationError {
    InvalidConfiguration(&'static str),
    InvalidInput(&'static str),
    InsufficientSamples {
        required: usize,
        actual: usize,
    },
    NoTimeOverlap,
    LowExcitation {
        visual_deg: f64,
        telemetry_deg: f64,
        required_deg: f64,
    },
    LowCorrelation {
        best: f64,
        required: f64,
    },
    DegenerateMotion(&'static str),
    ResidualTooHigh {
        residual_deg: f64,
        allowed_deg: f64,
    },
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => write!(formatter, "invalid calibration configuration: {reason}"),
            Self::InvalidInput(reason) => write!(formatter, "invalid calibration input: {reason}"),
            Self::InsufficientSamples { required, actual } => {
                write!(formatter, "insufficient calibration samples: {actual} < {required}")
            }
            Self::NoTimeOverlap => write!(formatter, "visual and telemetry timelines do not overlap"),
            Self::LowExcitation { visual_deg, telemetry_deg, required_deg } => write!(
                formatter,
                "insufficient rotational excitation: visual={visual_deg:.3}°, telemetry={telemetry_deg:.3}°, required={required_deg:.3}°"
            ),
            Self::LowCorrelation { best, required } => {
                write!(formatter, "angular-speed correlation is too low: {best:.4} < {required:.4}")
            }
            Self::DegenerateMotion(reason) => write!(formatter, "degenerate hand-eye motion: {reason}"),
            Self::ResidualTooHigh { residual_deg, allowed_deg } => write!(
                formatter,
                "hand-eye residual is too high: {residual_deg:.4}° > {allowed_deg:.4}°"
            ),
        }
    }
}

impl std::error::Error for CalibrationError {}

#[derive(Debug, Clone, Copy)]
struct AngularSample {
    timestamp_ms: f64,
    speed_dps: f64,
}

#[derive(Debug, Clone, Copy)]
struct UnitAxis {
    x: f64,
    y: f64,
    z: f64,
}

fn finite_quaternion(value: Quaternion) -> Result<Quaternion, CalibrationError> {
    normalize_quaternion(value).ok_or(CalibrationError::InvalidInput(
        "rotation contains a non-finite or zero-length quaternion",
    ))
}

fn finite_timestamp(timestamp_ms: f64) -> Result<(), CalibrationError> {
    timestamp_ms
        .is_finite()
        .then_some(())
        .ok_or(CalibrationError::InvalidInput(
            "rotation timestamp is not finite",
        ))
}

fn quaternion_conjugate(value: Quaternion) -> Quaternion {
    [value[0], -value[1], -value[2], -value[3]]
}

fn quaternion_inverse(value: Quaternion) -> Quaternion {
    quaternion_conjugate(value)
}

fn quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion {
    [
        left[0] * right[0] - left[1] * right[1] - left[2] * right[2] - left[3] * right[3],
        left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2],
        left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
        left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0],
    ]
}

fn relative_rotation(first: Quaternion, second: Quaternion) -> Quaternion {
    quaternion_multiply(first, quaternion_inverse(second))
}

fn angular_samples(
    samples: &[(f64, Quaternion)],
) -> Result<(Vec<AngularSample>, f64), CalibrationError> {
    if samples.len() < 2 {
        return Err(CalibrationError::InsufficientSamples {
            required: 2,
            actual: samples.len(),
        });
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut result = Vec::with_capacity(ordered.len().saturating_sub(1));
    let mut excitation = 0.0;
    for pair in ordered.windows(2) {
        let (left_timestamp, left_quaternion) = pair[0];
        let (right_timestamp, right_quaternion) = pair[1];
        finite_timestamp(left_timestamp)?;
        finite_timestamp(right_timestamp)?;
        let dt_ms = right_timestamp - left_timestamp;
        if dt_ms <= f64::EPSILON {
            continue;
        }
        let left_quaternion = finite_quaternion(left_quaternion)?;
        let right_quaternion = finite_quaternion(right_quaternion)?;
        let angle_deg = quaternion_angle_deg(left_quaternion, right_quaternion);
        if !angle_deg.is_finite() {
            return Err(CalibrationError::InvalidInput(
                "rotation angle is not finite",
            ));
        }
        let speed_dps = angle_deg * 1000.0 / dt_ms;
        if speed_dps.is_finite() {
            result.push(AngularSample {
                timestamp_ms: (left_timestamp + right_timestamp) * 0.5,
                speed_dps,
            });
            excitation += angle_deg;
        }
    }
    if result.is_empty() {
        return Err(CalibrationError::InsufficientSamples {
            required: 1,
            actual: 0,
        });
    }
    Ok((result, excitation))
}

fn interpolate_speed(samples: &[AngularSample], timestamp_ms: f64) -> Option<f64> {
    if !timestamp_ms.is_finite() || samples.is_empty() {
        return None;
    }
    if samples.len() == 1 {
        return if (samples[0].timestamp_ms - timestamp_ms).abs() <= f64::EPSILON {
            Some(samples[0].speed_dps)
        } else {
            None
        };
    }
    if timestamp_ms < samples[0].timestamp_ms
        || timestamp_ms > samples[samples.len() - 1].timestamp_ms
    {
        return None;
    }
    let index = samples.binary_search_by(|sample| sample.timestamp_ms.total_cmp(&timestamp_ms));
    match index {
        Ok(index) => Some(samples[index].speed_dps),
        Err(index) => {
            let left = samples.get(index.checked_sub(1)?)?;
            let right = samples.get(index)?;
            let duration = right.timestamp_ms - left.timestamp_ms;
            if duration <= f64::EPSILON {
                return Some(right.speed_dps);
            }
            let t = (timestamp_ms - left.timestamp_ms) / duration;
            Some(left.speed_dps * (1.0 - t) + right.speed_dps * t)
        }
    }
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let mean_left = left.iter().sum::<f64>() / left.len() as f64;
    let mean_right = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&left_value, &right_value) in left.iter().zip(right.iter()) {
        let left_delta = left_value - mean_left;
        let right_delta = right_value - mean_right;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    let denominator = (left_variance * right_variance).sqrt();
    (denominator > f64::EPSILON && denominator.is_finite())
        .then_some((covariance / denominator).clamp(-1.0, 1.0))
}

fn offset_candidates(config: CalibrationConfig) -> Vec<f64> {
    let span = config.offset_max_ms - config.offset_min_ms;
    let requested = (span / config.offset_step_ms).floor() as usize;
    let count = requested.min(DEFAULT_MAX_OFFSET_CANDIDATES);
    let effective_step = if requested > DEFAULT_MAX_OFFSET_CANDIDATES {
        span / DEFAULT_MAX_OFFSET_CANDIDATES as f64
    } else {
        config.offset_step_ms
    };
    (0..=count)
        .map(|index| {
            (config.offset_min_ms + index as f64 * effective_step).min(config.offset_max_ms)
        })
        .collect()
}

fn prepare_absolute_visual(
    samples: &[VisualRotationSample],
) -> Result<Vec<(f64, Quaternion)>, CalibrationError> {
    let mut result = samples
        .iter()
        .map(|sample| {
            finite_timestamp(sample.timestamp_ms)?;
            Ok((
                sample.timestamp_ms,
                finite_quaternion(sample.rotation_wxyz)?,
            ))
        })
        .collect::<Result<Vec<_>, CalibrationError>>()?;
    result.sort_by(|left, right| left.0.total_cmp(&right.0));
    result.dedup_by(|left, right| {
        if (left.0 - right.0).abs() <= f64::EPSILON {
            *left = *right;
            true
        } else {
            false
        }
    });
    Ok(result)
}

fn prepare_telemetry(
    samples: &[QuaternionSample],
) -> Result<Vec<(f64, Quaternion)>, CalibrationError> {
    let mut result = samples
        .iter()
        .map(|sample| {
            finite_timestamp(sample.timestamp_ms)?;
            Ok((sample.timestamp_ms, finite_quaternion(sample.quaternion())?))
        })
        .collect::<Result<Vec<_>, CalibrationError>>()?;
    result.sort_by(|left, right| left.0.total_cmp(&right.0));
    result.dedup_by(|left, right| {
        if (left.0 - right.0).abs() <= f64::EPSILON {
            *left = *right;
            true
        } else {
            false
        }
    });
    Ok(result)
}

fn interpolate_quaternion(samples: &[(f64, Quaternion)], timestamp_ms: f64) -> Option<Quaternion> {
    if !timestamp_ms.is_finite() || samples.is_empty() {
        return None;
    }
    if timestamp_ms < samples[0].0 || timestamp_ms > samples[samples.len() - 1].0 {
        return None;
    }
    let index = samples.binary_search_by(|sample| sample.0.total_cmp(&timestamp_ms));
    match index {
        Ok(index) => Some(samples[index].1),
        Err(index) => {
            let left = samples.get(index.checked_sub(1)?)?;
            let right = samples.get(index)?;
            let duration = right.0 - left.0;
            if duration <= f64::EPSILON {
                return Some(right.1);
            }
            let factor = (timestamp_ms - left.0) / duration;
            crate::telemetry::slerp_quaternion(left.1, right.1, factor)
        }
    }
}

/// Estimate the time offset by correlating visual and telemetry angular-speed
/// norms.  A positive offset means telemetry is sampled at
/// `visual_timestamp + offset`.
pub fn estimate_time_offset(
    visual_samples: &[VisualRotationSample],
    telemetry_samples: &[QuaternionSample],
    config: CalibrationConfig,
) -> Result<TimeOffsetEstimate, CalibrationError> {
    let config = config.validated()?;
    if visual_samples.len() < config.min_samples || telemetry_samples.len() < config.min_samples {
        return Err(CalibrationError::InsufficientSamples {
            required: config.min_samples,
            actual: visual_samples.len().min(telemetry_samples.len()),
        });
    }
    let visual = prepare_absolute_visual(visual_samples)?;
    let telemetry = prepare_telemetry(telemetry_samples)?;
    let visual_input = visual.clone();
    let telemetry_input = telemetry.clone();
    let (visual_speed, visual_excitation) = angular_samples(&visual_input)?;
    let (telemetry_speed, telemetry_excitation) = angular_samples(&telemetry_input)?;
    if visual_speed.len() < config.min_samples || telemetry_speed.len() < config.min_samples {
        return Err(CalibrationError::InsufficientSamples {
            required: config.min_samples,
            actual: visual_speed.len().min(telemetry_speed.len()),
        });
    }
    if visual_excitation < config.min_excitation_deg
        || telemetry_excitation < config.min_excitation_deg
    {
        return Err(CalibrationError::LowExcitation {
            visual_deg: visual_excitation,
            telemetry_deg: telemetry_excitation,
            required_deg: config.min_excitation_deg,
        });
    }
    let visual_start = visual_speed.first().map(|sample| sample.timestamp_ms);
    let visual_end = visual_speed.last().map(|sample| sample.timestamp_ms);
    let telemetry_start = telemetry_speed.first().map(|sample| sample.timestamp_ms);
    let telemetry_end = telemetry_speed.last().map(|sample| sample.timestamp_ms);
    let (visual_start, visual_end, telemetry_start, telemetry_end) =
        match (visual_start, visual_end, telemetry_start, telemetry_end) {
            (Some(visual_start), Some(visual_end), Some(telemetry_start), Some(telemetry_end)) => {
                (visual_start, visual_end, telemetry_start, telemetry_end)
            }
            _ => return Err(CalibrationError::NoTimeOverlap),
        };
    let mut best: Option<TimeOffsetEstimate> = None;
    for offset_ms in offset_candidates(config) {
        let mut visual_values = Vec::new();
        let mut telemetry_values = Vec::new();
        for sample in &visual_speed {
            let telemetry_timestamp = sample.timestamp_ms + offset_ms;
            let Some(telemetry_value) = interpolate_speed(&telemetry_speed, telemetry_timestamp)
            else {
                continue;
            };
            visual_values.push(sample.speed_dps);
            telemetry_values.push(telemetry_value);
        }
        if visual_values.len() < config.min_samples {
            continue;
        }
        let Some(correlation) = pearson_correlation(&visual_values, &telemetry_values) else {
            continue;
        };
        let candidate = TimeOffsetEstimate {
            time_offset_ms: offset_ms,
            correlation,
            paired_samples: visual_values.len(),
            visual_excitation_deg: visual_excitation,
            telemetry_excitation_deg: telemetry_excitation,
            coverage_start_ms: visual_start.max(telemetry_start - offset_ms),
            coverage_end_ms: visual_end.min(telemetry_end - offset_ms),
            valid: correlation >= config.min_correlation,
        };
        if candidate.coverage_start_ms >= candidate.coverage_end_ms {
            continue;
        }
        let replace = best.is_none_or(|current| {
            candidate.correlation > current.correlation + 1e-12
                || ((candidate.correlation - current.correlation).abs() <= 1e-12
                    && candidate.paired_samples > current.paired_samples)
        });
        if replace {
            best = Some(candidate);
        }
    }
    let best = best.ok_or(CalibrationError::NoTimeOverlap)?;
    if best.correlation < config.min_correlation {
        return Err(CalibrationError::LowCorrelation {
            best: best.correlation,
            required: config.min_correlation,
        });
    }
    Ok(best)
}

/// Same as [`estimate_time_offset`], using an already-normalized timeline.
/// The timeline is sampled at its reported rate (capped to a practical
/// 200 Hz) so callers do not need access to its private sample storage.
#[allow(dead_code)]
pub fn estimate_time_offset_with_timeline(
    visual_samples: &[VisualRotationSample],
    timeline: &AttitudeTimeline,
    config: CalibrationConfig,
) -> Result<TimeOffsetEstimate, CalibrationError> {
    let (start, end) = timeline.coverage().ok_or(CalibrationError::NoTimeOverlap)?;
    let rate = timeline
        .diagnostics()
        .rate_hz
        .unwrap_or(100.0)
        .clamp(1.0, 200.0);
    let mut step_ms = (1000.0 / rate).max(1.0);
    let mut count = ((end - start) / step_ms).ceil() as usize + 1;
    if count > 100_001 {
        step_ms = (end - start) / 100_000.0;
        count = 100_001;
    }
    let telemetry = (0..count)
        .filter_map(|index| {
            let timestamp_ms = (start + index as f64 * step_ms).min(end);
            timeline.interpolate(timestamp_ms)
        })
        .collect::<Vec<_>>();
    estimate_time_offset(visual_samples, &telemetry, config)
}

/// Run the complete calibration pipeline from an [`AttitudeTimeline`].  The
/// timeline is sampled using its reported rate (with the same practical cap
/// as [`estimate_time_offset_with_timeline`]); this keeps the API useful to
/// callers that intentionally do not expose the timeline's source samples.
#[allow(dead_code)]
pub fn estimate_calibration_with_timeline(
    visual_samples: &[VisualRotationSample],
    timeline: &AttitudeTimeline,
    config: CalibrationConfig,
) -> Result<CalibrationModel, CalibrationError> {
    let (start, end) = timeline.coverage().ok_or(CalibrationError::NoTimeOverlap)?;
    let rate = timeline
        .diagnostics()
        .rate_hz
        .unwrap_or(100.0)
        .clamp(1.0, 200.0);
    let mut step_ms = (1000.0 / rate).max(1.0);
    let mut count = ((end - start) / step_ms).ceil() as usize + 1;
    if count > 100_001 {
        step_ms = (end - start) / 100_000.0;
        count = 100_001;
    }
    let telemetry = (0..count)
        .filter_map(|index| {
            let timestamp_ms = (start + index as f64 * step_ms).min(end);
            timeline.interpolate(timestamp_ms)
        })
        .collect::<Vec<_>>();
    estimate_calibration(visual_samples, &telemetry, config)
}

fn quat_left_matrix(value: Quaternion) -> [[f64; 4]; 4] {
    let [w, x, y, z] = value;
    [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]]
}

fn quat_right_matrix(value: Quaternion) -> [[f64; 4]; 4] {
    let [w, x, y, z] = value;
    [[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]]
}

fn axis_for_quaternion(value: Quaternion) -> Option<(UnitAxis, f64)> {
    let value = finite_quaternion(value).ok()?;
    let angle_rad = 2.0 * value[0].clamp(-1.0, 1.0).abs().acos();
    let sine = (1.0 - value[0].clamp(-1.0, 1.0).powi(2)).sqrt();
    if angle_rad <= 1e-8 || sine <= 1e-8 {
        return None;
    }
    let sign = if value[0] >= 0.0 { 1.0 } else { -1.0 };
    Some((
        UnitAxis {
            x: value[1] * sign / sine,
            y: value[2] * sign / sine,
            z: value[3] * sign / sine,
        },
        angle_rad.to_degrees(),
    ))
}

fn axis_cross_norm(left: UnitAxis, right: UnitAxis) -> f64 {
    let cross_x = left.y * right.z - left.z * right.y;
    let cross_y = left.z * right.x - left.x * right.z;
    let cross_z = left.x * right.y - left.y * right.x;
    (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z).sqrt()
}

#[allow(clippy::needless_range_loop)] // Fixed-size Jacobi rotations are clearest by matrix index.
fn jacobi_smallest_eigenvector(mut matrix: [[f64; 4]; 4]) -> Option<Quaternion> {
    let mut vectors = [[0.0; 4]; 4];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..96 {
        let mut pivot = (0, 1);
        let mut largest = matrix[0][1].abs();
        for row in 0..4 {
            for column in (row + 1)..4 {
                if matrix[row][column].abs() > largest {
                    largest = matrix[row][column].abs();
                    pivot = (row, column);
                }
            }
        }
        if largest <= 1e-13 {
            break;
        }
        let (p, q) = pivot;
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let cosine = angle.cos();
        let sine = angle.sin();
        for row in 0..4 {
            let p_value = matrix[row][p];
            let q_value = matrix[row][q];
            matrix[row][p] = cosine * p_value - sine * q_value;
            matrix[row][q] = sine * p_value + cosine * q_value;
        }
        for column in 0..4 {
            let p_value = matrix[p][column];
            let q_value = matrix[q][column];
            matrix[p][column] = cosine * p_value - sine * q_value;
            matrix[q][column] = sine * p_value + cosine * q_value;
        }
        for row in 0..4 {
            let p_value = vectors[row][p];
            let q_value = vectors[row][q];
            vectors[row][p] = cosine * p_value - sine * q_value;
            vectors[row][q] = sine * p_value + cosine * q_value;
        }
    }
    let mut smallest_index = 0;
    for index in 1..4 {
        if matrix[index][index] < matrix[smallest_index][smallest_index] {
            smallest_index = index;
        }
    }
    let vector = [
        vectors[0][smallest_index],
        vectors[1][smallest_index],
        vectors[2][smallest_index],
        vectors[3][smallest_index],
    ];
    normalize_quaternion(vector)
}

/// Solve the rotational hand-eye equation `A X = X B` for a fixed `X`.
pub fn solve_rotational_hand_eye(
    pairs: &[RotationPair],
    config: CalibrationConfig,
) -> Result<HandEyeEstimate, CalibrationError> {
    let config = config.validated()?;
    if pairs.len() < config.min_samples {
        return Err(CalibrationError::InsufficientSamples {
            required: config.min_samples,
            actual: pairs.len(),
        });
    }
    let mut normal = [[0.0_f64; 4]; 4];
    let mut axes = Vec::new();
    let mut valid_pairs = 0usize;
    for pair in pairs {
        let visual = finite_quaternion(pair.visual_relative_wxyz)?;
        let telemetry = finite_quaternion(pair.telemetry_relative_wxyz)?;
        if let Some((axis, _angle)) = axis_for_quaternion(visual) {
            axes.push(axis);
        }
        let left = quat_left_matrix(visual);
        let right = quat_right_matrix(telemetry);
        for row in 0..4 {
            let mut equation = [0.0; 4];
            for column in 0..4 {
                equation[column] = left[row][column] - right[row][column];
            }
            for first in 0..4 {
                for second in 0..4 {
                    normal[first][second] += equation[first] * equation[second];
                }
            }
        }
        valid_pairs += 1;
    }
    if valid_pairs < config.min_samples {
        return Err(CalibrationError::InsufficientSamples {
            required: config.min_samples,
            actual: valid_pairs,
        });
    }
    let pair_excitation = pairs
        .iter()
        .filter_map(|pair| axis_for_quaternion(pair.visual_relative_wxyz).map(|(_, angle)| angle))
        .sum::<f64>();
    let telemetry_excitation = pairs
        .iter()
        .filter_map(|pair| {
            axis_for_quaternion(pair.telemetry_relative_wxyz).map(|(_, angle)| angle)
        })
        .sum::<f64>();
    let excitation = pair_excitation.min(telemetry_excitation);
    if excitation < config.min_excitation_deg {
        return Err(CalibrationError::LowExcitation {
            visual_deg: pair_excitation,
            telemetry_deg: telemetry_excitation,
            required_deg: config.min_excitation_deg,
        });
    }
    let mut axis_diversity: f64 = 0.0;
    for (index, left) in axes.iter().enumerate() {
        for right in axes.iter().skip(index + 1) {
            axis_diversity = axis_diversity.max(axis_cross_norm(*left, *right));
        }
    }
    if axis_diversity < config.min_axis_diversity {
        return Err(CalibrationError::DegenerateMotion(
            "relative rotation axes are nearly parallel",
        ));
    }
    let x = jacobi_smallest_eigenvector(normal).ok_or(CalibrationError::DegenerateMotion(
        "quaternion normal matrix has no finite solution",
    ))?;
    let mut residual_squared = 0.0;
    let mut max_residual: f64 = 0.0;
    for pair in pairs {
        let visual = finite_quaternion(pair.visual_relative_wxyz)?;
        let telemetry = finite_quaternion(pair.telemetry_relative_wxyz)?;
        let left = quaternion_multiply(visual, x);
        let right = quaternion_multiply(x, telemetry);
        let residual = quaternion_angle_deg(left, right);
        if !residual.is_finite() {
            return Err(CalibrationError::DegenerateMotion(
                "non-finite hand-eye residual",
            ));
        }
        residual_squared += residual * residual;
        max_residual = max_residual.max(residual);
    }
    let residual_deg = (residual_squared / pairs.len() as f64).sqrt();
    if residual_deg > config.max_residual_deg {
        return Err(CalibrationError::ResidualTooHigh {
            residual_deg,
            allowed_deg: config.max_residual_deg,
        });
    }
    Ok(HandEyeEstimate {
        sensor_to_camera_quaternion: x,
        residual_deg,
        max_residual_deg: max_residual,
        paired_samples: pairs.len(),
        excitation_deg: excitation,
        axis_diversity,
        valid: true,
    })
}

fn build_relative_pairs(
    visual: &[(f64, Quaternion)],
    telemetry: &[(f64, Quaternion)],
    offset_ms: f64,
    max_pairs: usize,
    invert_telemetry: bool,
) -> Vec<RotationPair> {
    if visual.len() < 2 || max_pairs == 0 {
        return Vec::new();
    }
    let stride = (visual.len() / max_pairs.max(1)).max(1);
    let mut pairs = Vec::with_capacity(max_pairs.min(visual.len()));
    for index in 0..visual.len().saturating_sub(stride) {
        let other = index + stride;
        let Some(mut first_telemetry) =
            interpolate_quaternion(telemetry, visual[index].0 + offset_ms)
        else {
            continue;
        };
        let Some(mut second_telemetry) =
            interpolate_quaternion(telemetry, visual[other].0 + offset_ms)
        else {
            continue;
        };
        if invert_telemetry {
            first_telemetry = quaternion_inverse(first_telemetry);
            second_telemetry = quaternion_inverse(second_telemetry);
        }
        pairs.push(RotationPair {
            visual_relative_wxyz: relative_rotation(visual[index].1, visual[other].1),
            telemetry_relative_wxyz: relative_rotation(first_telemetry, second_telemetry),
        });
        if pairs.len() >= max_pairs {
            break;
        }
    }
    pairs
}

/// Run the complete offset + rotational hand-eye calibration pipeline.
pub fn estimate_calibration(
    visual_samples: &[VisualRotationSample],
    telemetry_samples: &[QuaternionSample],
    config: CalibrationConfig,
) -> Result<CalibrationModel, CalibrationError> {
    let config = config.validated()?;
    let visual = prepare_absolute_visual(visual_samples)?;
    let telemetry = prepare_telemetry(telemetry_samples)?;
    let offset = estimate_time_offset(
        &visual
            .iter()
            .map(|(timestamp_ms, rotation_wxyz)| VisualRotationSample {
                timestamp_ms: *timestamp_ms,
                rotation_wxyz: *rotation_wxyz,
            })
            .collect::<Vec<_>>(),
        &telemetry
            .iter()
            .map(|(timestamp_ms, quaternion)| QuaternionSample {
                timestamp_ms: *timestamp_ms,
                w: quaternion[0],
                x: quaternion[1],
                y: quaternion[2],
                z: quaternion[3],
            })
            .collect::<Vec<_>>(),
        config,
    )?;
    let as_provided_pairs =
        build_relative_pairs(&visual, &telemetry, offset.time_offset_ms, 256, false);
    let inverted_pairs =
        build_relative_pairs(&visual, &telemetry, offset.time_offset_ms, 256, true);
    let as_provided = solve_rotational_hand_eye(&as_provided_pairs, config);
    let inverted = solve_rotational_hand_eye(&inverted_pairs, config);
    let (hand_eye, telemetry_orientation_convention) = match (as_provided, inverted) {
        (Ok(as_provided), Ok(inverted)) => {
            if as_provided.residual_deg <= inverted.residual_deg {
                (as_provided, TelemetryOrientationConvention::AsProvided)
            } else {
                (inverted, TelemetryOrientationConvention::Inverted)
            }
        }
        (Ok(as_provided), Err(_)) => (as_provided, TelemetryOrientationConvention::AsProvided),
        (Err(_), Ok(inverted)) => (inverted, TelemetryOrientationConvention::Inverted),
        (Err(as_provided_error), Err(_)) => return Err(as_provided_error),
    };
    let (visual_start, visual_end) = visual
        .first()
        .zip(visual.last())
        .map(|(first, last)| (first.0, last.0))
        .ok_or(CalibrationError::InsufficientSamples {
            required: config.min_samples,
            actual: visual.len(),
        })?;
    let (telemetry_start, telemetry_end) = telemetry
        .first()
        .zip(telemetry.last())
        .map(|(first, last)| {
            (
                first.0 - offset.time_offset_ms,
                last.0 - offset.time_offset_ms,
            )
        })
        .ok_or(CalibrationError::InsufficientSamples {
            required: config.min_samples,
            actual: telemetry.len(),
        })?;
    let coverage_start_ms = visual_start.max(telemetry_start);
    let coverage_end_ms = visual_end.min(telemetry_end);
    Ok(CalibrationModel {
        schema_version: IMU_CALIBRATION_SCHEMA_VERSION,
        valid: true,
        time_offset_ms: Some(offset.time_offset_ms),
        sensor_to_camera_quaternion: Some(hand_eye.sensor_to_camera_quaternion),
        residual_deg: Some(hand_eye.residual_deg),
        max_residual_deg: Some(hand_eye.max_residual_deg),
        correlation: Some(offset.correlation),
        coverage_start_ms: (coverage_start_ms < coverage_end_ms).then_some(coverage_start_ms),
        coverage_end_ms: (coverage_start_ms < coverage_end_ms).then_some(coverage_end_ms),
        paired_sample_count: hand_eye.paired_samples,
        visual_excitation_deg: Some(offset.visual_excitation_deg),
        telemetry_excitation_deg: Some(offset.telemetry_excitation_deg),
        hand_eye_excitation_deg: Some(hand_eye.excitation_deg),
        axis_diversity: Some(hand_eye.axis_diversity),
        telemetry_orientation_convention: Some(telemetry_orientation_convention),
        reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis_angle(axis: [f64; 3], angle_deg: f64) -> Quaternion {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let half = angle_deg.to_radians() * 0.5;
        [
            half.cos(),
            axis[0] / norm * half.sin(),
            axis[1] / norm * half.sin(),
            axis[2] / norm * half.sin(),
        ]
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

    fn visual_sample(timestamp_ms: f64, quaternion: Quaternion) -> VisualRotationSample {
        VisualRotationSample {
            timestamp_ms,
            rotation_wxyz: quaternion,
        }
    }

    fn mul(left: Quaternion, right: Quaternion) -> Quaternion {
        quaternion_multiply(left, right)
    }

    #[test]
    fn offset_search_recovers_known_positive_offset() {
        let mut visual = Vec::new();
        let mut telemetry = Vec::new();
        let step_ms = 50.0;
        for index in 0..80 {
            let t = index as f64 * step_ms;
            let angle = 12.0 * (index as f64 * 0.42).sin() + 7.0 * (index as f64 * 0.17).cos();
            let q = mul(
                axis_angle([0.0, 0.0, 1.0], angle),
                axis_angle([1.0, 0.0, 0.0], angle * 0.35),
            );
            visual.push(visual_sample(t, q));
            telemetry.push(sample(t + 80.0, q));
        }
        let config = CalibrationConfig {
            offset_min_ms: -150.0,
            offset_max_ms: 150.0,
            offset_step_ms: 5.0,
            min_correlation: 0.8,
            min_excitation_deg: 20.0,
            ..CalibrationConfig::default()
        };
        let estimate = estimate_time_offset(&visual, &telemetry, config).unwrap();
        assert!(
            (estimate.time_offset_ms - 80.0).abs() <= 5.0,
            "{estimate:?}"
        );
        assert!(estimate.correlation > 0.99, "{estimate:?}");
    }

    #[test]
    fn hand_eye_solver_recovers_fixed_basis_transform() {
        let x = axis_angle([0.3, -0.8, 0.4], 67.0);
        let mut pairs = Vec::new();
        let motions = [
            ([1.0, 0.0, 0.0], 28.0),
            ([0.0, 1.0, 0.0], 41.0),
            ([0.0, 0.0, 1.0], 35.0),
            ([1.0, 1.0, 0.0], 22.0),
            ([0.0, 1.0, 1.0], 31.0),
            ([1.0, 0.0, 1.0], 27.0),
            ([1.0, -1.0, 0.5], 19.0),
            ([0.5, 0.2, -1.0], 24.0),
            ([-0.4, 1.0, 0.7], 29.0),
        ];
        for (axis, angle) in motions {
            let telemetry = axis_angle(axis, angle);
            let visual = mul(mul(x, telemetry), quaternion_inverse(x));
            pairs.push(RotationPair {
                visual_relative_wxyz: visual,
                telemetry_relative_wxyz: telemetry,
            });
        }
        let config = CalibrationConfig {
            min_excitation_deg: 20.0,
            max_residual_deg: 0.01,
            ..CalibrationConfig::default()
        };
        let estimate = solve_rotational_hand_eye(&pairs, config).unwrap();
        assert!(
            quaternion_angle_deg(estimate.sensor_to_camera_quaternion, x) < 1e-5,
            "{estimate:?}"
        );
        assert!(estimate.residual_deg < 1e-5, "{estimate:?}");
    }

    #[test]
    fn low_excitation_is_rejected_in_both_stages() {
        let mut visual = Vec::new();
        let mut telemetry = Vec::new();
        for index in 0..16 {
            let t = index as f64 * 100.0;
            let q = axis_angle([0.0, 0.0, 1.0], index as f64 * 0.05);
            visual.push(visual_sample(t, q));
            telemetry.push(sample(t, q));
        }
        let offset_error =
            estimate_time_offset(&visual, &telemetry, CalibrationConfig::default()).unwrap_err();
        assert!(matches!(
            offset_error,
            CalibrationError::LowExcitation { .. }
        ));
        let pairs = (0..8)
            .map(|index| {
                let q = axis_angle([0.0, 0.0, 1.0], 0.1);
                let _ = index;
                RotationPair {
                    visual_relative_wxyz: q,
                    telemetry_relative_wxyz: q,
                }
            })
            .collect::<Vec<_>>();
        let hand_eye_error =
            solve_rotational_hand_eye(&pairs, CalibrationConfig::default()).unwrap_err();
        assert!(matches!(
            hand_eye_error,
            CalibrationError::LowExcitation { .. } | CalibrationError::DegenerateMotion(_)
        ));
    }

    #[test]
    fn invalid_model_is_explicit_and_serializable() {
        let model = CalibrationModel::invalid("insufficient excitation");
        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("\"schemaVersion\":1"));
        assert!(json.contains("\"valid\":false"));
        assert!(json.contains("insufficient excitation"));
    }

    #[test]
    fn quaternion_matrix_conventions_match_multiplication() {
        let left = axis_angle([1.0, 2.0, 3.0], 32.0);
        let right = axis_angle([-2.0, 1.0, 0.5], 17.0);
        let expected = mul(left, right);
        let left_matrix = quat_left_matrix(left);
        let right_matrix = quat_right_matrix(right);
        let mut left_result = [0.0; 4];
        let mut right_result = [0.0; 4];
        for row in 0..4 {
            left_result[row] = left_matrix[row]
                .iter()
                .zip(right.iter())
                .map(|(coefficient, value)| coefficient * value)
                .sum();
            right_result[row] = right_matrix[row]
                .iter()
                .zip(left.iter())
                .map(|(coefficient, value)| coefficient * value)
                .sum();
        }
        assert!(quaternion_angle_deg(expected, left_result) < 1e-8);
        assert!(quaternion_angle_deg(expected, right_result) < 1e-8);
    }

    #[test]
    fn configuration_rejects_non_positive_offset_step() {
        let config = CalibrationConfig {
            offset_step_ms: 0.0,
            ..CalibrationConfig::default()
        };
        let visual = [visual_sample(0.0, [1.0, 0.0, 0.0, 0.0])];
        let telemetry = [sample(0.0, [1.0, 0.0, 0.0, 0.0])];
        assert!(matches!(
            estimate_time_offset(&visual, &telemetry, config),
            Err(CalibrationError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn complete_pipeline_reports_convention_and_fixed_rotation() {
        let x = axis_angle([0.4, -0.2, 0.8], 52.0);
        let x_inverse = quaternion_inverse(x);
        let mut visual = Vec::new();
        let mut telemetry = Vec::new();
        let mut imu_orientation = [1.0, 0.0, 0.0, 0.0];
        for index in 0..90 {
            let timestamp_ms = index as f64 * 40.0;
            let axis = match index % 3 {
                0 => [1.0, 0.1, 0.0],
                1 => [0.0, 1.0, 0.2],
                _ => [0.2, 0.0, 1.0],
            };
            let increment = axis_angle(axis, 2.0 + (index % 5) as f64 * 0.35);
            imu_orientation = mul(imu_orientation, increment);
            let camera_orientation = mul(mul(x, imu_orientation), x_inverse);
            visual.push(visual_sample(timestamp_ms, camera_orientation));
            telemetry.push(sample(timestamp_ms + 80.0, imu_orientation));
        }
        let config = CalibrationConfig {
            offset_min_ms: -120.0,
            offset_max_ms: 160.0,
            offset_step_ms: 5.0,
            min_excitation_deg: 30.0,
            min_correlation: 0.7,
            max_residual_deg: 0.1,
            ..CalibrationConfig::default()
        };
        let model = estimate_calibration(&visual, &telemetry, config).unwrap();
        assert!(model.valid);
        assert_eq!(
            model.telemetry_orientation_convention,
            Some(TelemetryOrientationConvention::AsProvided)
        );
        assert!((model.time_offset_ms.unwrap() - 80.0).abs() <= 5.0);
        assert!(
            quaternion_angle_deg(model.sensor_to_camera_quaternion.unwrap(), x) < 1e-4,
            "{model:?}"
        );
    }
}
