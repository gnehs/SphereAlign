//! Safe, normalized telemetry export for supported video containers.
//!
//! DJI Osmo 360 currently exposes fused attitude through telemetry-parser's
//! `dvtm_oq101` decoder. Raw data streams are still preserved independently;
//! these quaternions must not be treated as COLMAP camera qvec values without
//! an explicit, verified sensor-to-camera coordinate transform.

use prost::Message;
use serde::Serialize;
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuaternionSample {
    pub timestamp_ms: f64,
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedTelemetry {
    pub schema_version: u32,
    pub parser: String,
    pub parser_revision: &'static str,
    pub camera_type: String,
    pub camera_model: Option<String>,
    pub source_size: u64,
    pub source_modified_nanos: Option<String>,
    pub timestamps_accurate: bool,
    pub sensor_readout_time_ms: Option<f64>,
    pub timebase: &'static str,
    pub normalized_imu_sample_count: usize,
    pub normalized_imu: Vec<IMUData>,
    pub fused_attitude_sample_count: usize,
    pub fused_attitude_rate_hz: Option<f64>,
    pub fused_attitude: Vec<QuaternionSample>,
    pub coordinate_frame: &'static str,
    pub applied_to_colmap: bool,
    pub warnings: Vec<&'static str>,
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
            fused_attitude.extend(values.get().iter().map(|value| QuaternionSample {
                timestamp_ms: value.t,
                w: value.v.w,
                x: value.v.x,
                y: value.v.y,
                z: value.v.z,
            }));
        }
    }
    if normalized_imu.is_empty() && fused_attitude.is_empty() {
        return Err("supported container was detected, but it contained no usable IMU or fused-attitude samples".to_owned());
    }

    let fused_attitude_rate_hz = match (fused_attitude.first(), fused_attitude.last()) {
        (Some(first), Some(last)) if last.timestamp_ms > first.timestamp_ms => Some(
            (fused_attitude.len().saturating_sub(1) as f64 * 1000.0)
                / (last.timestamp_ms - first.timestamp_ms),
        ),
        _ => None,
    };
    let camera_model = input.camera_model().cloned();
    let normalized_imu_sample_count = normalized_imu.len();
    let normalized = NormalizedTelemetry {
        schema_version: 1,
        parser: input.parser_name().to_owned(),
        parser_revision: PARSER_REVISION,
        camera_type: input.camera_type(),
        camera_model: camera_model.clone(),
        source_size,
        source_modified_nanos,
        timestamps_accurate: input.has_accurate_timestamps(),
        sensor_readout_time_ms: input.frame_readout_time(),
        timebase: "milliseconds relative to the first DJI metadata frame; leading samples may be negative",
        normalized_imu_sample_count,
        normalized_imu,
        fused_attitude_sample_count: fused_attitude.len(),
        fused_attitude_rate_hz,
        fused_attitude,
        coordinate_frame: "telemetry-parser DJI normalized attitude; not a COLMAP camera qvec",
        applied_to_colmap: false,
        warnings: vec![
            "Raw OSV data streams remain the source of truth.",
            "A verified sensor-to-camera transform is required before using attitude as a COLMAP prior.",
        ],
    };
    write_json_atomic(output_path, &normalized)?;
    Ok(TelemetryExport {
        path: output_path.to_path_buf(),
        camera_model,
        normalized_imu_sample_count,
        fused_attitude_sample_count: normalized.fused_attitude_sample_count,
    })
}

fn existing_export(
    path: &Path,
    source_size: u64,
    source_modified_nanos: Option<&str>,
) -> Option<TelemetryExport> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    if value.get("parserRevision")?.as_str()? != PARSER_REVISION
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
