//! Safe COLMAP 4.1.1 pose/focal-prior database integration.
//!
//! This module deliberately writes only the fields COLMAP exposes as
//! `PosePrior`: gravity, position, position covariance, and coordinate system.
//! It never stores a DJI quaternion in the database.  The schema and byte
//! layouts below follow the official COLMAP 4.1.1 source:
//!
//! * `src/colmap/scene/database_sqlite.cc` (`CreatePosePriorTable`,
//!   `ReadStaticMatrixBlob`, and `WriteStaticMatrixBlob`)
//! * `src/colmap/geometry/pose_prior.h` (`PosePrior::kNaN` and gravity)
//! * `src/colmap/scene/database_sqlite.cc` (`Read/WriteRigid3d...` for rig
//!   extrinsics)
//!
//! Static Eigen matrix BLOBs are copied in the host's native byte order by
//! COLMAP (there is no endian marker).  We therefore reject big-endian hosts
//! instead of silently producing a database that COLMAP would decode wrongly.
//! Rig extrinsics are a separate seven-double little-endian BLOB in COLMAP
//! 4.1.1 and are validated as such.

use rusqlite::{params, Connection, Row};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Metadata schema emitted by [`write_global_mapper_prior_marker`].
pub const GLOBAL_MAPPER_PRIOR_MARKER_SCHEMA_VERSION: u32 = 1;
/// COLMAP's current database version number for release 4.1.1.
pub const COLMAP_4_1_1_DATABASE_VERSION: i64 = 4_010_100;
/// Global rotation averaging should not be enabled with a sparse gravity set.
pub const MIN_GLOBAL_GRAVITY_COVERAGE_RATIO: f64 = 0.8;

const CAMERA_SENSOR_TYPE: i64 = 0;
const UNDEFINED_COORDINATE_SYSTEM: i64 = -1;
const MAX_CAMERA_MODEL_ID: i64 = 16;
const MAX_REASONABLE_IMAGE_DIMENSION: i64 = 100_000;
const F64_BYTES: usize = std::mem::size_of::<f64>();
const GRAVITY_BLOB_BYTES: usize = 3 * F64_BYTES;
const RIG_EXTRINSIC_BLOB_BYTES: usize = 7 * F64_BYTES;

/// One sensor-frame gravity observation keyed by the exact COLMAP image name.
///
/// The vector is normalized before writing.  It must be expressed in the
/// camera/sensor coordinate system; this type intentionally has no quaternion
/// field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GravityPriorInput {
    pub image_name: String,
    pub gravity: [f64; 3],
}

/// A focal prior record that was obtained from a real calibration source.
///
/// `verified` is an explicit provenance gate.  Callers must not set it for
/// COLMAP's `default_focal_length_factor = 0.3` guess.  Only metadata copied
/// from an actual calibration or COLMAP's view-graph calibrator is accepted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FocalPriorInput {
    pub camera_id: i64,
    pub model: i64,
    pub width: i64,
    pub height: i64,
    pub params: Vec<f64>,
    pub source: String,
    pub verified: bool,
}

/// Calibration metadata recorded alongside an injected prior set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PriorCalibrationMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_offset_ms: Option<f64>,
    /// Require a valid `sensor_from_rig` for every non-reference camera.
    /// This defaults to true because gravity-aligned global mapping requires
    /// known rig extrinsics.
    #[serde(default = "default_require_complete_rig_extrinsics")]
    pub require_complete_rig_extrinsics: bool,
}

impl Default for PriorCalibrationMetadata {
    fn default() -> Self {
        Self {
            calibration_version: None,
            time_offset_ms: None,
            require_complete_rig_extrinsics: true,
        }
    }
}

fn default_require_complete_rig_extrinsics() -> bool {
    true
}

/// Summary of gravity writes and the database coverage they provide.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GravityPriorReport {
    pub input_count: usize,
    pub injected_count: usize,
    pub updated_count: usize,
    pub camera_frame_image_count: usize,
    pub gravity_coverage_ratio: f64,
    pub gravity_prior_valid: bool,
    pub normalized_gravity_count: usize,
}

/// Summary of focal-prior validation and camera coverage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FocalPriorReport {
    pub input_count: usize,
    pub marked_count: usize,
    pub perspective_camera_count: usize,
    pub cameras_with_prior_focal_length: usize,
    pub focal_coverage_ratio: f64,
    pub focal_prior_valid: bool,
    pub accepted_sources: Vec<String>,
}

/// Marker payload consumed by the guarded `global_mapper` path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GlobalMapperPriorMarker {
    pub schema_version: u32,
    pub colmap_database_version: i64,
    pub focal_prior_valid: bool,
    pub focal_coverage_ratio: f64,
    pub focal_prior_camera_count: usize,
    pub focal_prior_camera_total: usize,
    pub gravity_prior_valid: bool,
    pub gravity_coverage_ratio: f64,
    pub gravity_prior_image_count: usize,
    pub gravity_image_total: usize,
    pub database_pose_priors_injected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_version: Option<String>,
    /// Keep the name expected by the pipeline's prerequisite checker.
    #[serde(
        rename = "sensorToCameraCalibrationVersion",
        skip_serializing_if = "Option::is_none"
    )]
    pub sensor_to_camera_calibration_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_offset_ms: Option<f64>,
}

/// Combined report returned after validating and injecting both prior types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GlobalMapperPriorReport {
    pub marker: GlobalMapperPriorMarker,
    pub gravity: GravityPriorReport,
    pub focal: FocalPriorReport,
}

/// One calibrated camera entry reconstructed from COLMAP 4.1.1's rig tables.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RigCameraExtrinsic {
    pub image_prefix: String,
    pub ref_sensor: bool,
    pub cam_from_rig_rotation: [f64; 4],
    pub cam_from_rig_translation: [f64; 3],
}

/// Read the configured camera rig back from the database.  This is used to
/// persist an unknown-rig bootstrap result before the gravity/global pass.
pub fn read_rig_camera_extrinsics(database_path: &Path) -> Result<Vec<RigCameraExtrinsic>, String> {
    let connection = open_checked_database(database_path)?;
    let mut statement = connection
        .prepare(
            "SELECT i.name, i.camera_id, r.ref_sensor_id, rs.sensor_from_rig
             FROM images i
             JOIN frame_data fd
               ON fd.data_id = i.image_id AND fd.sensor_type = 0
             JOIN frames f ON f.frame_id = fd.frame_id
             JOIN rigs r ON r.rig_id = f.rig_id
             LEFT JOIN rig_sensors rs
               ON rs.rig_id = f.rig_id
              AND rs.sensor_id = i.camera_id
              AND rs.sensor_type = 0
             ORDER BY i.camera_id, i.name",
        )
        .map_err(|error| format!("無法讀取 rig camera extrinsics：{error}"))?;
    let mut rows = statement.query([]).map_err(|error| error.to_string())?;
    let mut cameras = BTreeMap::<i64, RigCameraExtrinsic>::new();
    while let Some(row) = rows.next().map_err(|error| error.to_string())? {
        let name: String = row.get(0).map_err(|error| error.to_string())?;
        let camera_id: i64 = row.get(1).map_err(|error| error.to_string())?;
        let ref_sensor_id: i64 = row.get(2).map_err(|error| error.to_string())?;
        let blob: Option<Vec<u8>> = row.get(3).map_err(|error| error.to_string())?;
        let prefix = name
            .split_once('/')
            .map(|(prefix, _)| format!("{prefix}/"))
            .ok_or_else(|| format!("COLMAP image name 缺少 lens prefix：{name}"))?;
        let ref_sensor = camera_id == ref_sensor_id;
        let values = if ref_sensor {
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        } else {
            let blob = blob
                .ok_or_else(|| format!("camera {camera_id} 缺少 non-reference sensor_from_rig"))?;
            validate_rig_extrinsic_blob(&blob, &name)?;
            let decoded = blob
                .chunks_exact(F64_BYTES)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("f64 chunk")))
                .collect::<Vec<_>>();
            decoded
                .try_into()
                .map_err(|_| format!("camera {camera_id} sensor_from_rig 長度不是 7 doubles"))?
        };
        let entry = RigCameraExtrinsic {
            image_prefix: prefix,
            ref_sensor,
            cam_from_rig_rotation: [values[0], values[1], values[2], values[3]],
            cam_from_rig_translation: [values[4], values[5], values[6]],
        };
        if let Some(existing) = cameras.get(&camera_id) {
            if existing != &entry {
                return Err(format!("camera {camera_id} 對應到不一致的 rig extrinsics"));
            }
        } else {
            cameras.insert(camera_id, entry);
        }
    }
    if cameras.len() < 2 || cameras.values().filter(|camera| camera.ref_sensor).count() != 1 {
        return Err("COLMAP database 未提供完整雙鏡頭 rig extrinsics".to_owned());
    }
    Ok(cameras.into_values().collect())
}

#[derive(Debug, Clone)]
struct ImageBinding {
    image_id: i64,
    sensor_id: i64,
    sensor_type: i64,
    is_reference_sensor: bool,
    sensor_from_rig: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct ValidatedFocalInput {
    input: FocalPriorInput,
    expected_params: Vec<f64>,
}

type ExistingPosePriorRow = (i64, Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>, i64);
type PosePriorRoundTripRow = (
    i64,
    i64,
    i64,
    Option<Vec<u8>>,
    Option<Vec<u8>>,
    i64,
    Option<Vec<u8>>,
);

/// Inject gravity priors in one atomic transaction and verify every row after
/// writing.  Missing image names, ambiguous frame mappings, malformed schema,
/// invalid vectors, or malformed existing priors reject the entire operation.
#[allow(dead_code)]
pub fn inject_gravity_priors(
    database_path: &Path,
    inputs: &[GravityPriorInput],
    calibration: &PriorCalibrationMetadata,
) -> Result<GravityPriorReport, String> {
    let mut connection = open_checked_database(database_path)?;
    run_immediate_transaction(&mut connection, |db| {
        let bindings = validate_gravity_inputs(db, inputs, calibration)?;
        write_gravity_rows_in_transaction(db, inputs, &bindings)
    })
}

/// Validate real focal calibration records and set COLMAP's
/// `prior_focal_length=1` flag atomically.  Existing valid flags are retained
/// and included in the returned coverage.
#[allow(dead_code)]
pub fn mark_focal_priors(
    database_path: &Path,
    inputs: &[FocalPriorInput],
) -> Result<FocalPriorReport, String> {
    let mut connection = open_checked_database(database_path)?;
    run_immediate_transaction(&mut connection, |db| {
        let validated = validate_focal_inputs(db, inputs)?;
        write_focal_rows_in_transaction(db, &validated)
    })
}

/// Read back all perspective cameras whose `prior_focal_length` flag is set.
///
/// This is intentionally strict: a single perspective camera without a prior
/// causes an error instead of returning a partial vector.  The function is
/// intended for the post-`view_graph_calibrator` round-trip, so the caller can
/// prove that every camera record used by global mapping has a real focal
/// prior.  Equirectangular cameras (model 17) have no focal length and are
/// excluded from this contract.
pub fn read_focal_prior_inputs(
    database_path: &Path,
    source: &str,
) -> Result<Vec<FocalPriorInput>, String> {
    if !is_accepted_focal_source(source) {
        return Err(format!(
            "focal prior source 不受信任：{source}（只接受 metadata 或 view_graph_calibrator）"
        ));
    }
    let connection = open_checked_database(database_path)?;
    let mut statement = connection
        .prepare(
            "SELECT camera_id, model, width, height, params, prior_focal_length
             FROM cameras ORDER BY camera_id",
        )
        .map_err(|error| format!("無法讀取 cameras focal priors：{error}"))?;
    let mut rows = statement
        .query([])
        .map_err(|error| format!("無法查詢 cameras focal priors：{error}"))?;
    let mut output = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|error| format!("無法讀取 camera focal prior row：{error}"))?
    {
        let camera_id: i64 = row.get(0).map_err(|error| error.to_string())?;
        let model: i64 = row.get(1).map_err(|error| error.to_string())?;
        let width: i64 = row.get(2).map_err(|error| error.to_string())?;
        let height: i64 = row.get(3).map_err(|error| error.to_string())?;
        let params_blob: Option<Vec<u8>> = row.get(4).map_err(|error| error.to_string())?;
        let prior: i64 = row.get(5).map_err(|error| error.to_string())?;
        if model == 17 {
            continue;
        }
        if model > 17 {
            return Err(format!(
                "camera {camera_id} 使用尚未驗證的 model {model}，無法建立 focal prior"
            ));
        }
        if prior != 1 {
            return Err(format!(
                "camera {camera_id} 的 prior_focal_length 不是 1；拒絕回傳不完整 focal prior set"
            ));
        }
        let expected_count = camera_model_num_params(model)
            .ok_or_else(|| format!("camera {camera_id} model {model} layout 未知"))?;
        let Some(params_blob) = params_blob else {
            return Err(format!("camera {camera_id} 的 params 是 NULL"));
        };
        let params = decode_native_f64_blob(&params_blob, expected_count, "camera params")?;
        let input = FocalPriorInput {
            camera_id,
            model,
            width,
            height,
            params,
            source: source.trim().to_owned(),
            verified: true,
        };
        validate_focal_input_shape(&input)?;
        output.push(input);
    }
    if output.is_empty() {
        return Err("database 沒有可用的 perspective focal prior".to_owned());
    }
    Ok(output)
}

/// Return the validated focal coverage summary without changing the database.
/// This is useful after COLMAP's `view_graph_calibrator`, whose successful
/// process exit alone is not sufficient evidence that every camera received a
/// usable prior.
pub fn read_focal_prior_report(
    database_path: &Path,
    source: &str,
) -> Result<FocalPriorReport, String> {
    let inputs = read_focal_prior_inputs(database_path, source)?;
    let mut sources = BTreeSet::new();
    sources.insert(source.trim().to_ascii_lowercase());
    let connection = open_checked_database(database_path)?;
    summarize_focal_priors(&connection, inputs.len(), sources)
}

/// Inject gravity priors and mark verified focal priors in one transaction.
/// The returned marker is ready to persist with
/// [`write_global_mapper_prior_marker`].
pub fn inject_global_mapper_priors(
    database_path: &Path,
    focal_inputs: &[FocalPriorInput],
    gravity_inputs: &[GravityPriorInput],
    calibration: &PriorCalibrationMetadata,
) -> Result<GlobalMapperPriorReport, String> {
    let mut connection = open_checked_database(database_path)?;
    let (gravity_report, focal_report) = run_immediate_transaction(&mut connection, |db| {
        let bindings = validate_gravity_inputs(db, gravity_inputs, calibration)?;
        let validated_focal = validate_focal_inputs(db, focal_inputs)?;
        let gravity = write_gravity_rows_in_transaction(db, gravity_inputs, &bindings)?;
        let focal = write_focal_rows_in_transaction(db, &validated_focal)?;
        Ok((gravity, focal))
    })?;

    let calibration_version = calibration
        .calibration_version
        .as_ref()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let marker = GlobalMapperPriorMarker {
        schema_version: GLOBAL_MAPPER_PRIOR_MARKER_SCHEMA_VERSION,
        colmap_database_version: COLMAP_4_1_1_DATABASE_VERSION,
        focal_prior_valid: focal_report.focal_prior_valid,
        focal_coverage_ratio: focal_report.focal_coverage_ratio,
        focal_prior_camera_count: focal_report.cameras_with_prior_focal_length,
        focal_prior_camera_total: focal_report.perspective_camera_count,
        gravity_prior_valid: gravity_report.gravity_prior_valid,
        gravity_coverage_ratio: gravity_report.gravity_coverage_ratio,
        gravity_prior_image_count: gravity_report.injected_count,
        gravity_image_total: gravity_report.camera_frame_image_count,
        database_pose_priors_injected: gravity_report.injected_count > 0,
        calibration_version: calibration_version.clone(),
        sensor_to_camera_calibration_version: calibration_version,
        time_offset_ms: calibration.time_offset_ms,
    };

    Ok(GlobalMapperPriorReport {
        marker,
        gravity: gravity_report,
        focal: focal_report,
    })
}

/// Persist a marker atomically.  The temporary file is in the same directory,
/// so a rename cannot accidentally cross filesystems.
pub fn write_global_mapper_prior_marker(
    path: &Path,
    report: &GlobalMapperPriorReport,
) -> Result<(), String> {
    write_global_mapper_prior_marker_payload(path, &report.marker)
}

/// Persist an already validated marker payload.  This is useful when focal
/// calibration and gravity injection are performed as separate stages but
/// both reports have independently passed their strict validators.
pub fn write_global_mapper_prior_marker_payload(
    path: &Path,
    marker: &GlobalMapperPriorMarker,
) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("無法建立 global mapper prior marker 目錄：{error}"))?;
    let temporary = path.with_extension("json.partial");
    let bytes = serde_json::to_vec_pretty(marker)
        .map_err(|error| format!("無法序列化 global mapper prior marker：{error}"))?;
    fs::write(&temporary, bytes)
        .map_err(|error| format!("無法寫入 global mapper prior marker：{error}"))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("無法提交 global mapper prior marker：{error}"))
}

fn open_checked_database(path: &Path) -> Result<Connection, String> {
    if cfg!(target_endian = "big") {
        return Err(
            "目前平台為 big-endian；COLMAP 4.1.1 的 static matrix BLOB 沒有 endian marker，拒絕寫入"
                .to_owned(),
        );
    }
    let connection = Connection::open(path)
        .map_err(|error| format!("無法開啟 COLMAP database {}：{error}", path.display()))?;
    connection
        .execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|error| format!("無法啟用 SQLite foreign keys：{error}"))?;
    verify_colmap_schema(&connection)?;
    Ok(connection)
}

fn verify_colmap_schema(connection: &Connection) -> Result<(), String> {
    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|error| format!("無法讀取 COLMAP database user_version：{error}"))?;
    if version > COLMAP_4_1_1_DATABASE_VERSION {
        return Err(format!(
            "COLMAP database user_version {version} 高於已驗證的 4.1.1 schema {COLMAP_4_1_1_DATABASE_VERSION}，拒絕寫入"
        ));
    }

    // COLMAP 4.1.1 calls this table `frame_data`; there is intentionally no
    // generic `sensors` table.  The camera sensor is identified by
    // frame_data.sensor_type = 0 (SensorType::CAMERA).
    let required = [
        (
            "images",
            [
                ("image_id", "INTEGER"),
                ("name", "TEXT"),
                ("camera_id", "INTEGER"),
            ]
            .as_slice(),
        ),
        (
            "cameras",
            [
                ("camera_id", "INTEGER"),
                ("model", "INTEGER"),
                ("width", "INTEGER"),
                ("height", "INTEGER"),
                ("params", "BLOB"),
                ("prior_focal_length", "INTEGER"),
            ]
            .as_slice(),
        ),
        (
            "frames",
            [("frame_id", "INTEGER"), ("rig_id", "INTEGER")].as_slice(),
        ),
        (
            "frame_data",
            [
                ("frame_id", "INTEGER"),
                ("data_id", "INTEGER"),
                ("sensor_id", "INTEGER"),
                ("sensor_type", "INTEGER"),
            ]
            .as_slice(),
        ),
        (
            "rigs",
            [
                ("rig_id", "INTEGER"),
                ("ref_sensor_id", "INTEGER"),
                ("ref_sensor_type", "INTEGER"),
            ]
            .as_slice(),
        ),
        (
            "rig_sensors",
            [
                ("rig_id", "INTEGER"),
                ("sensor_id", "INTEGER"),
                ("sensor_type", "INTEGER"),
                ("sensor_from_rig", "BLOB"),
            ]
            .as_slice(),
        ),
        (
            "pose_priors",
            [
                ("pose_prior_id", "INTEGER"),
                ("corr_data_id", "INTEGER"),
                ("corr_sensor_id", "INTEGER"),
                ("corr_sensor_type", "INTEGER"),
                ("position", "BLOB"),
                ("position_covariance", "BLOB"),
                ("gravity", "BLOB"),
                ("coordinate_system", "INTEGER"),
            ]
            .as_slice(),
        ),
    ];
    for (table, columns) in required {
        let actual = table_columns(connection, table)?;
        for (column, expected_type) in columns {
            let Some(actual_type) = actual.get(*column) else {
                return Err(format!("COLMAP schema 缺少 {table}.{column}"));
            };
            if !actual_type.eq_ignore_ascii_case(expected_type) {
                return Err(format!(
                    "COLMAP schema 欄位 {table}.{column} 型別為 {actual_type}，預期 {expected_type}"
                ));
            }
        }
    }
    Ok(())
}

fn table_columns(connection: &Connection, table: &str) -> Result<BTreeMap<String, String>, String> {
    let mut statement = connection
        .prepare(&format!("PRAGMA table_info('{}')", table))
        .map_err(|error| format!("無法檢查 COLMAP table {table}：{error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })
        .map_err(|error| format!("無法讀取 COLMAP table {table} schema：{error}"))?;
    let mut columns = BTreeMap::new();
    for row in rows {
        let (name, declared_type) =
            row.map_err(|error| format!("無法讀取 COLMAP table {table} 欄位：{error}"))?;
        columns.insert(name, declared_type);
    }
    if columns.is_empty() {
        return Err(format!("COLMAP schema 缺少 table {table}"));
    }
    Ok(columns)
}

fn validate_gravity_inputs(
    connection: &Connection,
    inputs: &[GravityPriorInput],
    calibration: &PriorCalibrationMetadata,
) -> Result<Vec<ImageBinding>, String> {
    validate_calibration_metadata(calibration)?;
    let mut seen = BTreeSet::new();
    let mut bindings = Vec::with_capacity(inputs.len());
    for input in inputs {
        if input.image_name.trim().is_empty() {
            return Err("gravity prior image_name 不得為空".to_owned());
        }
        if !seen.insert(input.image_name.clone()) {
            return Err(format!(
                "gravity prior image_name 重複：{}",
                input.image_name
            ));
        }
        let normalized = normalize_gravity(input.gravity)?;
        let binding = load_image_binding(connection, &input.image_name, calibration)?;
        // Store the normalized value only after all validation has completed;
        // this check also catches a caller that supplied a non-finite vector.
        if !normalized.iter().all(|value| value.is_finite()) {
            return Err(format!("gravity prior {} 含非有限值", input.image_name));
        }
        bindings.push(binding);
    }
    Ok(bindings)
}

fn validate_calibration_metadata(metadata: &PriorCalibrationMetadata) -> Result<(), String> {
    if metadata
        .calibration_version
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err("sensor-to-camera calibration version 不得為空白".to_owned());
    }
    if metadata
        .time_offset_ms
        .is_some_and(|value| !value.is_finite())
    {
        return Err("time_offset_ms 必須是有限數值".to_owned());
    }
    Ok(())
}

fn load_image_binding(
    connection: &Connection,
    image_name: &str,
    calibration: &PriorCalibrationMetadata,
) -> Result<ImageBinding, String> {
    let sql = "SELECT i.image_id, i.camera_id, fd.frame_id, fd.data_id,
                      fd.sensor_id, fd.sensor_type, f.rig_id,
                      r.ref_sensor_id, r.ref_sensor_type, rs.sensor_from_rig
               FROM images i
               JOIN frame_data fd
                 ON fd.data_id = i.image_id AND fd.sensor_type = ?1
               JOIN frames f ON f.frame_id = fd.frame_id
               JOIN rigs r ON r.rig_id = f.rig_id
               LEFT JOIN rig_sensors rs
                 ON rs.rig_id = f.rig_id
                AND rs.sensor_id = fd.sensor_id
                AND rs.sensor_type = fd.sensor_type
               WHERE i.name = ?2";
    let mut statement = connection
        .prepare(sql)
        .map_err(|error| format!("無法準備 image/frame/rig 對應查詢：{error}"))?;
    let mut rows = statement
        .query(params![CAMERA_SENSOR_TYPE, image_name])
        .map_err(|error| format!("無法查詢 image/frame/rig 對應：{error}"))?;
    let mut matches = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|error| format!("無法讀取 image/frame/rig 對應：{error}"))?
    {
        matches.push(parse_image_binding(row)?);
    }
    if matches.is_empty() {
        return Err(format!(
            "image {image_name} 找不到 COLMAP 4.1.1 的 camera frame_data/frames/rigs 對應"
        ));
    }
    if matches.len() != 1 {
        return Err(format!(
            "image {image_name} 對應到 {} 組 frame_data；拒絕使用不唯一 corr_data_id",
            matches.len()
        ));
    }
    let binding = matches.remove(0);
    if binding.sensor_type != CAMERA_SENSOR_TYPE {
        return Err(format!("image {image_name} 不是 COLMAP camera sensor"));
    }
    if calibration.require_complete_rig_extrinsics && !binding.is_reference_sensor {
        let Some(blob) = binding.sensor_from_rig.as_deref() else {
            return Err(format!(
                "image {image_name} 的 non-reference sensor 缺少 sensor_from_rig 外參"
            ));
        };
        validate_rig_extrinsic_blob(blob, image_name)?;
    }
    Ok(binding)
}

fn parse_image_binding(row: &Row<'_>) -> Result<ImageBinding, String> {
    let image_id: i64 = row.get(0).map_err(|error| error.to_string())?;
    let camera_id: i64 = row.get(1).map_err(|error| error.to_string())?;
    let frame_id: i64 = row.get(2).map_err(|error| error.to_string())?;
    let data_id: i64 = row.get(3).map_err(|error| error.to_string())?;
    let sensor_id: i64 = row.get(4).map_err(|error| error.to_string())?;
    let sensor_type: i64 = row.get(5).map_err(|error| error.to_string())?;
    let rig_id: i64 = row.get(6).map_err(|error| error.to_string())?;
    let ref_sensor_id: i64 = row.get(7).map_err(|error| error.to_string())?;
    let ref_sensor_type: i64 = row.get(8).map_err(|error| error.to_string())?;
    let sensor_from_rig: Option<Vec<u8>> = row.get(9).map_err(|error| error.to_string())?;
    if image_id < 0 || camera_id < 0 || frame_id < 0 || data_id < 0 || sensor_id < 0 || rig_id < 0 {
        return Err("COLMAP image/frame/rig identifier 不得為負數".to_owned());
    }
    if image_id != data_id || camera_id != sensor_id || sensor_type != CAMERA_SENSOR_TYPE {
        return Err(format!(
            "COLMAP image/frame_data camera mapping 不一致：image={image_id}, camera={camera_id}, data={data_id}, sensor={sensor_id}, sensor_type={sensor_type}"
        ));
    }
    if ref_sensor_type != CAMERA_SENSOR_TYPE || ref_sensor_id < 0 {
        return Err("COLMAP rig 的 reference sensor 不是 CAMERA 或 id 無效".to_owned());
    }
    Ok(ImageBinding {
        image_id,
        sensor_id,
        sensor_type,
        is_reference_sensor: sensor_id == ref_sensor_id,
        sensor_from_rig,
    })
}

fn validate_rig_extrinsic_blob(blob: &[u8], image_name: &str) -> Result<(), String> {
    if blob.len() != RIG_EXTRINSIC_BLOB_BYTES {
        return Err(format!(
            "image {image_name} 的 sensor_from_rig BLOB 長度為 {}，預期 {RIG_EXTRINSIC_BLOB_BYTES} bytes",
            blob.len()
        ));
    }
    let values = blob
        .chunks_exact(F64_BYTES)
        .map(|chunk| {
            let bytes: [u8; F64_BYTES] = chunk.try_into().expect("chunks_exact size");
            f64::from_le_bytes(bytes)
        })
        .collect::<Vec<_>>();
    if !values.iter().all(|value| value.is_finite()) {
        return Err(format!("image {image_name} 的 sensor_from_rig 含非有限值"));
    }
    let norm = values[..4]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON || (norm - 1.0).abs() > 1e-3 {
        return Err(format!(
            "image {image_name} 的 sensor_from_rig quaternion 未正規化"
        ));
    }
    Ok(())
}

fn normalize_gravity(gravity: [f64; 3]) -> Result<[f64; 3], String> {
    if !gravity.iter().all(|value| value.is_finite()) {
        return Err("gravity 必須全部是有限數值".to_owned());
    }
    let norm = gravity
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm <= 1e-12 {
        return Err("gravity 不得是零向量".to_owned());
    }
    Ok([gravity[0] / norm, gravity[1] / norm, gravity[2] / norm])
}

fn validate_focal_inputs(
    connection: &Connection,
    inputs: &[FocalPriorInput],
) -> Result<Vec<ValidatedFocalInput>, String> {
    let mut seen = BTreeSet::new();
    let mut validated = Vec::with_capacity(inputs.len());
    for input in inputs {
        if !seen.insert(input.camera_id) {
            return Err(format!("focal prior camera_id 重複：{}", input.camera_id));
        }
        validate_focal_input_shape(input)?;
        let (db_model, db_width, db_height, blob, prior): (i64, i64, i64, Option<Vec<u8>>, i64) =
            connection
                .query_row(
                    "SELECT model, width, height, params, prior_focal_length
                     FROM cameras WHERE camera_id = ?1",
                    params![input.camera_id],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    },
                )
                .map_err(|error| format!("無法讀取 camera {}：{error}", input.camera_id))?;
        if prior != 0 && prior != 1 {
            return Err(format!(
                "camera {} 的 prior_focal_length 不是 0/1：{prior}",
                input.camera_id
            ));
        }
        if (db_model, db_width, db_height) != (input.model, input.width, input.height) {
            return Err(format!(
                "camera {} 的 model/width/height 與校正 metadata 不一致",
                input.camera_id
            ));
        }
        let Some(blob) = blob else {
            return Err(format!("camera {} 的 params 是 NULL", input.camera_id));
        };
        let db_params = decode_native_f64_blob(&blob, input.params.len(), "camera params")?;
        for (index, (actual, expected)) in db_params.iter().zip(&input.params).enumerate() {
            let tolerance = 1e-12_f64.max(expected.abs() * 1e-10);
            if (actual - expected).abs() > tolerance {
                return Err(format!(
                    "camera {} 的 params[{index}] 與校正 metadata 不一致",
                    input.camera_id
                ));
            }
        }
        validated.push(ValidatedFocalInput {
            input: input.clone(),
            expected_params: db_params,
        });
    }
    Ok(validated)
}

fn validate_focal_input_shape(input: &FocalPriorInput) -> Result<(), String> {
    if input.camera_id < 0 {
        return Err("focal prior camera_id 不得為負數".to_owned());
    }
    if !(0..=MAX_CAMERA_MODEL_ID).contains(&input.model) {
        return Err(format!(
            "camera {} 的 model {} 不是已驗證的 perspective COLMAP model（0..16）",
            input.camera_id, input.model
        ));
    }
    if !(1..=MAX_REASONABLE_IMAGE_DIMENSION).contains(&input.width)
        || !(1..=MAX_REASONABLE_IMAGE_DIMENSION).contains(&input.height)
    {
        return Err(format!(
            "camera {} 的 width/height 無效：{}x{}",
            input.camera_id, input.width, input.height
        ));
    }
    let expected_count = camera_model_num_params(input.model).ok_or_else(|| {
        format!(
            "camera {} 的 model {} 沒有可安全驗證的 parameter layout",
            input.camera_id, input.model
        )
    })?;
    if input.params.len() != expected_count {
        return Err(format!(
            "camera {} 的 params 數量為 {}，model {} 預期 {expected_count}",
            input.camera_id,
            input.params.len(),
            input.model
        ));
    }
    if !input.params.iter().all(|value| value.is_finite()) {
        return Err(format!("camera {} 的 params 含非有限值", input.camera_id));
    }
    let focal_indices = camera_model_focal_indices(input.model);
    if focal_indices
        .iter()
        .any(|index| input.params.get(*index).is_none_or(|value| *value <= 0.0))
    {
        return Err(format!(
            "camera {} 的 focal parameter 必須大於零",
            input.camera_id
        ));
    }
    if !is_accepted_focal_source(&input.source) {
        return Err(format!(
            "camera {} 的 focal prior source 不受信任：{}（只接受 metadata 或 view_graph_calibrator）",
            input.camera_id, input.source
        ));
    }
    if !input.verified {
        return Err(format!(
            "camera {} 的 focal prior 未標記 verified；不可把 default 0.3 猜測值宣告為 prior",
            input.camera_id
        ));
    }
    if looks_like_default_focal_guess(input) {
        return Err(format!(
            "camera {} 的 focal params 看起來是 default_focal_length_factor=0.3 猜測值，不得標記為 prior",
            input.camera_id
        ));
    }
    Ok(())
}

fn looks_like_default_focal_guess(input: &FocalPriorInput) -> bool {
    let default_focal = 0.3 * input.width.max(input.height) as f64;
    let focal_indices = camera_model_focal_indices(input.model);
    let focal_matches = focal_indices.iter().all(|index| {
        input
            .params
            .get(*index)
            .is_some_and(|value| (*value - default_focal).abs() <= 1e-12)
    });
    let principal_indices = camera_model_principal_indices(input.model);
    let principal_point_matches = principal_indices.first().is_some_and(|index| {
        input
            .params
            .get(*index)
            .is_some_and(|value| (*value - input.width as f64 / 2.0).abs() <= 1e-12)
    }) && principal_indices.get(1).is_some_and(|index| {
        input
            .params
            .get(*index)
            .is_some_and(|value| (*value - input.height as f64 / 2.0).abs() <= 1e-12)
    });
    let distortion_is_zero = input
        .params
        .iter()
        .enumerate()
        .filter(|(index, _)| !focal_indices.contains(index) && !principal_indices.contains(index))
        .all(|(_, value)| value.abs() <= 1e-12);
    focal_matches && principal_point_matches && distortion_is_zero
}

fn is_accepted_focal_source(source: &str) -> bool {
    matches!(
        source.trim().to_ascii_lowercase().as_str(),
        "metadata" | "view_graph_calibrator" | "viewgraphcalibrator" | "view-graph-calibrator"
    )
}

fn camera_model_num_params(model: i64) -> Option<usize> {
    Some(match model {
        0 => 3,   // SIMPLE_PINHOLE
        1 => 4,   // PINHOLE
        2 => 4,   // SIMPLE_RADIAL
        3 => 5,   // RADIAL
        4 => 8,   // OPENCV
        5 => 8,   // OPENCV_FISHEYE
        6 => 12,  // FULL_OPENCV
        7 => 5,   // FOV
        8 => 4,   // SIMPLE_RADIAL_FISHEYE
        9 => 5,   // RADIAL_FISHEYE
        10 => 12, // THIN_PRISM_FISHEYE
        11 => 16, // RAD_TAN_THIN_PRISM_FISHEYE
        12 => 4,  // SIMPLE_DIVISION
        13 => 5,  // DIVISION
        14 => 3,  // SIMPLE_FISHEYE
        15 => 4,  // FISHEYE
        16 => 6,  // EUCM
        _ => return None,
    })
}

fn camera_model_focal_indices(model: i64) -> &'static [usize] {
    match model {
        0 | 2 | 3 | 7 | 8 | 9 | 12 | 14 => &[0],
        1 | 4 | 5 | 6 | 10 | 11 | 13 | 15 | 16 => &[0, 1],
        _ => &[],
    }
}

fn camera_model_principal_indices(model: i64) -> &'static [usize] {
    match model {
        0 | 2 | 3 | 7 | 8 | 9 | 12 | 14 => &[1, 2],
        1 | 4 | 5 | 6 | 10 | 11 | 13 | 15 | 16 => &[2, 3],
        _ => &[],
    }
}

fn write_gravity_rows_in_transaction(
    connection: &Connection,
    inputs: &[GravityPriorInput],
    bindings: &[ImageBinding],
) -> Result<GravityPriorReport, String> {
    if inputs.len() != bindings.len() {
        return Err("gravity prior inputs 與 image bindings 數量不一致".to_owned());
    }
    let position_nan = nan_blob(3);
    let covariance_nan = nan_blob(9);
    let mut injected_count = 0;
    let mut updated_count = 0;
    for (input, binding) in inputs.iter().zip(bindings) {
        let gravity = normalize_gravity(input.gravity)?;
        let gravity_blob = native_f64_blob(&gravity);
        let existing = find_existing_pose_prior(connection, binding, &input.image_name)?;
        let had_existing = existing.is_some();
        if let Some((pose_prior_id, position, covariance, old_gravity, old_coordinate_system)) =
            existing
        {
            if pose_prior_id < 0 || pose_prior_id > u32::MAX as i64 {
                return Err(format!(
                    "image {} 的既有 pose_prior_id 超出 uint32 範圍",
                    input.image_name
                ));
            }
            let position_state = validate_existing_matrix_blob(
                position.as_deref(),
                3,
                "position",
                &input.image_name,
            )?;
            let covariance_state = validate_existing_matrix_blob(
                covariance.as_deref(),
                9,
                "position_covariance",
                &input.image_name,
            )?;
            if position_state.is_nan && !covariance_state.is_nan {
                return Err(format!(
                    "image {} 的 position 未知但 covariance 已知，拒絕不一致 PosePrior",
                    input.image_name
                ));
            }
            let (position_blob, covariance_blob, coordinate_system) = if position_state.is_nan {
                (&position_nan, &covariance_nan, UNDEFINED_COORDINATE_SYSTEM)
            } else {
                let position_blob = position.as_ref().expect("validated position BLOB");
                let covariance_blob = covariance.as_ref().expect("validated covariance BLOB");
                if !(-1..=1).contains(&old_coordinate_system) {
                    return Err(format!(
                        "image {} 的 coordinate_system 無效：{}",
                        input.image_name, old_coordinate_system
                    ));
                }
                (position_blob, covariance_blob, old_coordinate_system)
            };
            if let Some(old_gravity) = old_gravity.as_deref() {
                if old_gravity.len() != GRAVITY_BLOB_BYTES {
                    return Err(format!(
                        "image {} 的既有 gravity BLOB 長度錯誤",
                        input.image_name
                    ));
                }
            }
            connection
                .execute(
                    "UPDATE pose_priors
                     SET corr_data_id = ?1, corr_sensor_id = ?2, corr_sensor_type = ?3,
                         position = ?4, position_covariance = ?5,
                         coordinate_system = ?6, gravity = ?7
                     WHERE pose_prior_id = ?8",
                    params![
                        binding.image_id,
                        binding.sensor_id,
                        binding.sensor_type,
                        position_blob,
                        covariance_blob,
                        coordinate_system,
                        &gravity_blob,
                        pose_prior_id,
                    ],
                )
                .map_err(|error| {
                    format!(
                        "無法更新 image {} 的 gravity prior：{error}",
                        input.image_name
                    )
                })?;
            updated_count += 1;
            verify_gravity_row(
                connection,
                binding,
                &gravity,
                position_blob,
                covariance_blob,
                coordinate_system,
                &input.image_name,
            )?;
        } else {
            let pose_prior_id: i64 = connection
                .query_row(
                    "SELECT COALESCE(MAX(pose_prior_id), -1) + 1 FROM pose_priors",
                    [],
                    |row| row.get(0),
                )
                .map_err(|error| format!("無法配置 pose_prior_id：{error}"))?;
            if pose_prior_id < 0 || pose_prior_id > u32::MAX as i64 {
                return Err("pose_prior_id 超出 COLMAP 4.1.1 uint32 範圍".to_owned());
            }
            connection
                .execute(
                    "INSERT INTO pose_priors
                       (pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                        position, position_covariance, coordinate_system, gravity)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        pose_prior_id,
                        binding.image_id,
                        binding.sensor_id,
                        binding.sensor_type,
                        &position_nan,
                        &covariance_nan,
                        UNDEFINED_COORDINATE_SYSTEM,
                        &gravity_blob,
                    ],
                )
                .map_err(|error| {
                    format!(
                        "無法新增 image {} 的 gravity prior：{error}",
                        input.image_name
                    )
                })?;
            injected_count += 1;
        }
        if !had_existing {
            verify_gravity_row(
                connection,
                binding,
                &gravity,
                &position_nan,
                &covariance_nan,
                UNDEFINED_COORDINATE_SYSTEM,
                &input.image_name,
            )?;
        }
    }
    let camera_frame_image_count = count_camera_frame_images(connection)?;
    let covered = injected_count + updated_count;
    let gravity_coverage_ratio = coverage_ratio(covered, camera_frame_image_count);
    Ok(GravityPriorReport {
        input_count: inputs.len(),
        injected_count: covered,
        updated_count,
        camera_frame_image_count,
        gravity_coverage_ratio,
        gravity_prior_valid: covered > 0
            && gravity_coverage_ratio >= MIN_GLOBAL_GRAVITY_COVERAGE_RATIO,
        normalized_gravity_count: inputs.len(),
    })
}

fn find_existing_pose_prior(
    connection: &Connection,
    binding: &ImageBinding,
    image_name: &str,
) -> Result<Option<ExistingPosePriorRow>, String> {
    let mut statement = connection
        .prepare(
            "SELECT pose_prior_id, position, position_covariance, gravity,
                    coordinate_system
             FROM pose_priors
             WHERE corr_data_id = ?1 AND corr_sensor_id = ?2 AND corr_sensor_type = ?3",
        )
        .map_err(|error| format!("無法準備 image {image_name} 的 pose prior 查詢：{error}"))?;
    let rows = statement
        .query_map(
            params![binding.image_id, binding.sensor_id, binding.sensor_type],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .map_err(|error| format!("無法讀取 image {image_name} 的 pose prior：{error}"))?;
    let mut matches = Vec::new();
    for row in rows {
        matches.push(
            row.map_err(|error| format!("無法讀取 image {image_name} 的 pose prior：{error}"))?,
        );
    }
    if matches.len() > 1 {
        return Err(format!(
            "image {image_name} 的 pose_priors corr_data_id mapping 不唯一（{} rows）",
            matches.len()
        ));
    }
    Ok(matches.pop())
}

fn verify_gravity_row(
    connection: &Connection,
    binding: &ImageBinding,
    expected_gravity: &[f64; 3],
    expected_position: &[u8],
    expected_covariance: &[u8],
    expected_coordinate_system: i64,
    image_name: &str,
) -> Result<(), String> {
    let row: PosePriorRoundTripRow = connection
        .query_row(
            "SELECT corr_data_id, corr_sensor_id, corr_sensor_type, position,
                    position_covariance, coordinate_system, gravity
             FROM pose_priors WHERE corr_data_id = ?1 AND corr_sensor_id = ?2
             AND corr_sensor_type = ?3",
            params![binding.image_id, binding.sensor_id, binding.sensor_type],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            },
        )
        .map_err(|error| format!("gravity prior read-after-write 失敗（{image_name}）：{error}"))?;
    if row.0 != binding.image_id || row.1 != binding.sensor_id || row.2 != binding.sensor_type {
        return Err(format!(
            "gravity prior corr_data_id mapping 錯誤（{image_name}）"
        ));
    }
    if row.5 != expected_coordinate_system {
        return Err(format!(
            "gravity prior coordinate_system read-after-write 不一致（{image_name}）"
        ));
    }
    if row.3.as_deref() != Some(expected_position) {
        return Err(format!(
            "gravity prior position read-after-write 不一致（{image_name}）"
        ));
    }
    if row.4.as_deref() != Some(expected_covariance) {
        return Err(format!(
            "gravity prior position_covariance read-after-write 不一致（{image_name}）"
        ));
    }
    let Some(gravity_blob) = row.6 else {
        return Err(format!("gravity prior BLOB 為 NULL（{image_name}）"));
    };
    let actual = decode_native_f64_blob(&gravity_blob, 3, "gravity")?;
    for (actual, expected) in actual.iter().zip(expected_gravity) {
        if (actual - expected).abs() > 1e-12 {
            return Err(format!(
                "gravity prior read-after-write 數值不一致（{image_name}）"
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ExistingMatrixState {
    is_nan: bool,
}

fn validate_existing_matrix_blob(
    blob: Option<&[u8]>,
    values: usize,
    field: &str,
    image_name: &str,
) -> Result<ExistingMatrixState, String> {
    let Some(blob) = blob else {
        return Err(format!(
            "image {image_name} 的既有 {field} 是 NULL；拒絕在未知值上附加 gravity prior"
        ));
    };
    let decoded = decode_native_f64_blob(blob, values, field)
        .map_err(|error| format!("image {image_name} 的既有 {field} BLOB 無效：{error}"))?;
    let all_nan = decoded.iter().all(|value| value.is_nan());
    let all_finite = decoded.iter().all(|value| value.is_finite());
    if !all_nan && !all_finite {
        return Err(format!(
            "image {image_name} 的既有 {field} 必須全部是 NaN 或全部是有限值"
        ));
    }
    Ok(ExistingMatrixState { is_nan: all_nan })
}

fn count_camera_frame_images(connection: &Connection) -> Result<usize, String> {
    let count: i64 = connection
        .query_row(
            "SELECT COUNT(DISTINCT i.image_id) FROM images i
             JOIN frame_data fd ON fd.data_id = i.image_id AND fd.sensor_type = ?1
             JOIN frames f ON f.frame_id = fd.frame_id
             JOIN rigs r ON r.rig_id = f.rig_id",
            params![CAMERA_SENSOR_TYPE],
            |row| row.get(0),
        )
        .map_err(|error| format!("無法計算 camera frame image coverage：{error}"))?;
    usize::try_from(count).map_err(|_| "camera frame image count 超出 usize".to_owned())
}

fn write_focal_rows_in_transaction(
    connection: &Connection,
    inputs: &[ValidatedFocalInput],
) -> Result<FocalPriorReport, String> {
    for input in inputs {
        connection
            .execute(
                "UPDATE cameras SET prior_focal_length = 1 WHERE camera_id = ?1",
                params![input.input.camera_id],
            )
            .map_err(|error| {
                format!(
                    "無法標記 camera {} prior_focal_length：{error}",
                    input.input.camera_id
                )
            })?;
        let prior: i64 = connection
            .query_row(
                "SELECT prior_focal_length FROM cameras WHERE camera_id = ?1",
                params![input.input.camera_id],
                |row| row.get(0),
            )
            .map_err(|error| {
                format!(
                    "camera {} read-after-write 失敗：{error}",
                    input.input.camera_id
                )
            })?;
        if prior != 1 {
            return Err(format!(
                "camera {} prior_focal_length read-after-write 不是 1",
                input.input.camera_id
            ));
        }
        // Keep the decoded parameter vector live in this transaction.  This
        // makes it explicit that only the flag is changed; params are never
        // rewritten by this API.
        if input.expected_params.is_empty() {
            return Err(format!("camera {} params 不得為空", input.input.camera_id));
        }
    }
    summarize_focal_priors(
        connection,
        inputs.len(),
        inputs
            .iter()
            .map(|input| input.input.source.trim().to_ascii_lowercase())
            .collect(),
    )
}

fn summarize_focal_priors(
    connection: &Connection,
    input_count: usize,
    input_sources: BTreeSet<String>,
) -> Result<FocalPriorReport, String> {
    let mut statement = connection
        .prepare("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras")
        .map_err(|error| format!("無法檢查 cameras focal coverage：{error}"))?;
    let mut rows = statement
        .query([])
        .map_err(|error| format!("無法讀取 cameras focal coverage：{error}"))?;
    let mut perspective_total = 0_usize;
    let mut prior_total = 0_usize;
    while let Some(row) = rows
        .next()
        .map_err(|error| format!("無法讀取 camera focal coverage row：{error}"))?
    {
        let camera_id: i64 = row.get(0).map_err(|error| error.to_string())?;
        let model: i64 = row.get(1).map_err(|error| error.to_string())?;
        let width: i64 = row.get(2).map_err(|error| error.to_string())?;
        let height: i64 = row.get(3).map_err(|error| error.to_string())?;
        let params_blob: Option<Vec<u8>> = row.get(4).map_err(|error| error.to_string())?;
        let prior: i64 = row.get(5).map_err(|error| error.to_string())?;
        if model > MAX_CAMERA_MODEL_ID {
            // EQUIRECTANGULAR (17) and future spherical models do not have a
            // focal prior.  They are excluded from the denominator.
            continue;
        }
        if model < 0 {
            return Err(format!("camera {camera_id} model 無效：{model}"));
        }
        perspective_total += 1;
        if !(1..=MAX_REASONABLE_IMAGE_DIMENSION).contains(&width)
            || !(1..=MAX_REASONABLE_IMAGE_DIMENSION).contains(&height)
        {
            return Err(format!("camera {camera_id} width/height 無效"));
        }
        let expected_params = camera_model_num_params(model)
            .ok_or_else(|| format!("camera {camera_id} model {model} layout 未知"))?;
        let Some(params_blob) = params_blob else {
            return Err(format!("camera {camera_id} params 是 NULL"));
        };
        let values = decode_native_f64_blob(&params_blob, expected_params, "camera params")?;
        if !values.iter().all(|value| value.is_finite()) {
            return Err(format!("camera {camera_id} params 含非有限值"));
        }
        if camera_model_focal_indices(model)
            .iter()
            .any(|index| values.get(*index).is_none_or(|value| *value <= 0.0))
        {
            return Err(format!("camera {camera_id} focal parameter 無效"));
        }
        if prior != 0 && prior != 1 {
            return Err(format!("camera {camera_id} prior_focal_length 不是 0/1"));
        }
        if prior == 1 {
            prior_total += 1;
        }
    }
    let ratio = coverage_ratio(prior_total, perspective_total);
    let sources = input_sources.into_iter().collect::<Vec<_>>();
    Ok(FocalPriorReport {
        input_count,
        marked_count: input_count,
        perspective_camera_count: perspective_total,
        cameras_with_prior_focal_length: prior_total,
        focal_coverage_ratio: ratio,
        focal_prior_valid: perspective_total > 0 && ratio >= MIN_GLOBAL_GRAVITY_COVERAGE_RATIO,
        accepted_sources: sources,
    })
}

fn coverage_ratio(covered: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        covered as f64 / total as f64
    }
}

fn run_immediate_transaction<T, F>(connection: &mut Connection, operation: F) -> Result<T, String>
where
    F: FnOnce(&Connection) -> Result<T, String>,
{
    connection
        .execute_batch("BEGIN IMMEDIATE;")
        .map_err(|error| format!("無法開始 COLMAP prior transaction：{error}"))?;
    match operation(connection) {
        Ok(value) => {
            if let Err(error) = connection.execute_batch("COMMIT;") {
                let _ = connection.execute_batch("ROLLBACK;");
                Err(format!("無法提交 COLMAP prior transaction：{error}"))
            } else {
                Ok(value)
            }
        }
        Err(error) => {
            let _ = connection.execute_batch("ROLLBACK;");
            Err(error)
        }
    }
}

fn native_f64_blob(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * F64_BYTES);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn nan_blob(values: usize) -> Vec<u8> {
    native_f64_blob(&vec![f64::NAN; values])
}

fn decode_native_f64_blob(bytes: &[u8], values: usize, field: &str) -> Result<Vec<f64>, String> {
    if bytes.len() != values * F64_BYTES {
        return Err(format!(
            "{field} BLOB 長度為 {}，預期 {}",
            bytes.len(),
            values * F64_BYTES
        ));
    }
    Ok(bytes
        .chunks_exact(F64_BYTES)
        .map(|chunk| {
            let value_bytes: [u8; F64_BYTES] = chunk.try_into().expect("chunks_exact size");
            f64::from_ne_bytes(value_bytes)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::NamedTempFile;

    fn fixture(path: &Path, add_non_reference: bool) {
        let connection = Connection::open(path).expect("open fixture");
        connection
            .execute_batch(&format!(
                "PRAGMA user_version = {COLMAP_4_1_1_DATABASE_VERSION};
                 CREATE TABLE cameras(
                   camera_id INTEGER PRIMARY KEY NOT NULL,
                   model INTEGER NOT NULL,
                   width INTEGER NOT NULL,
                   height INTEGER NOT NULL,
                   params BLOB,
                   prior_focal_length INTEGER NOT NULL
                 );
                 CREATE TABLE rigs(
                   rig_id INTEGER PRIMARY KEY NOT NULL,
                   ref_sensor_id INTEGER NOT NULL,
                   ref_sensor_type INTEGER NOT NULL
                 );
                 CREATE TABLE rig_sensors(
                   rig_id INTEGER NOT NULL,
                   sensor_id INTEGER NOT NULL,
                   sensor_type INTEGER NOT NULL,
                   sensor_from_rig BLOB
                 );
                 CREATE TABLE frames(
                   frame_id INTEGER PRIMARY KEY NOT NULL,
                   rig_id INTEGER NOT NULL
                 );
                 CREATE TABLE frame_data(
                   frame_id INTEGER NOT NULL,
                   data_id INTEGER NOT NULL,
                   sensor_id INTEGER NOT NULL,
                   sensor_type INTEGER NOT NULL
                 );
                 CREATE TABLE images(
                   image_id INTEGER PRIMARY KEY NOT NULL,
                   name TEXT NOT NULL UNIQUE,
                   camera_id INTEGER NOT NULL
                 );
                 CREATE TABLE pose_priors(
                   pose_prior_id INTEGER PRIMARY KEY NOT NULL,
                   corr_data_id INTEGER NOT NULL,
                   corr_sensor_id INTEGER NOT NULL,
                   corr_sensor_type INTEGER NOT NULL,
                   position BLOB,
                   position_covariance BLOB,
                   gravity BLOB,
                   coordinate_system INTEGER NOT NULL
                 );"
            ))
            .expect("create fixture schema");
        let params = native_f64_blob(&[100.0, 100.0, 64.0, 64.0, 0.0, 0.0, 0.0, 0.0]);
        connection
            .execute(
                "INSERT INTO cameras VALUES(1,5,128,128,?1,0)",
                params![params],
            )
            .expect("camera");
        connection
            .execute("INSERT INTO rigs VALUES(1,1,0)", [])
            .expect("rig");
        connection
            .execute("INSERT INTO frames VALUES(1,1)", [])
            .expect("frame");
        connection
            .execute("INSERT INTO frame_data VALUES(1,1,1,0)", [])
            .expect("frame data");
        connection
            .execute("INSERT INTO images VALUES(1,'lens0/frame.jpg',1)", [])
            .expect("image");
        if add_non_reference {
            let extrinsic = native_f64_blob(&[1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]);
            // Real COLMAP rig blobs are little-endian, which is also native on
            // the supported test hosts.
            connection
                .execute(
                    "INSERT INTO cameras VALUES(2,5,128,128,?1,0)",
                    params![params],
                )
                .expect("camera 2");
            connection
                .execute(
                    "INSERT INTO rig_sensors VALUES(1,2,0,?1)",
                    params![extrinsic],
                )
                .expect("rig sensor");
            connection
                .execute("INSERT INTO frame_data VALUES(1,2,2,0)", [])
                .expect("frame data 2");
            connection
                .execute("INSERT INTO images VALUES(2,'lens1/frame.jpg',2)", [])
                .expect("image 2");
        }
    }

    fn focal_input(camera_id: i64) -> FocalPriorInput {
        FocalPriorInput {
            camera_id,
            model: 5,
            width: 128,
            height: 128,
            params: vec![100.0, 100.0, 64.0, 64.0, 0.0, 0.0, 0.0, 0.0],
            source: "metadata".to_owned(),
            verified: true,
        }
    }

    #[test]
    fn writes_gravity_with_nan_position_and_round_trips() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), false);
        let report = inject_gravity_priors(
            file.path(),
            &[GravityPriorInput {
                image_name: "lens0/frame.jpg".to_owned(),
                gravity: [0.0, 0.0, 2.0],
            }],
            &PriorCalibrationMetadata {
                calibration_version: Some("test-calibration".to_owned()),
                time_offset_ms: Some(0.0),
                require_complete_rig_extrinsics: true,
            },
        )
        .expect("gravity write");
        assert_eq!(report.injected_count, 1);
        assert_eq!(report.gravity_coverage_ratio, 1.0);
        let connection = Connection::open(file.path()).expect("reopen");
        let (position, covariance, gravity): (Vec<u8>, Vec<u8>, Vec<u8>) = connection
            .query_row(
                "SELECT position, position_covariance, gravity FROM pose_priors",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read prior");
        assert!(decode_native_f64_blob(&position, 3, "position")
            .unwrap()
            .iter()
            .all(|value| value.is_nan()));
        assert!(decode_native_f64_blob(&covariance, 9, "covariance")
            .unwrap()
            .iter()
            .all(|value| value.is_nan()));
        assert_eq!(
            decode_native_f64_blob(&gravity, 3, "gravity").unwrap(),
            vec![0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn reads_reference_and_non_reference_rig_extrinsics() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), true);
        let cameras = read_rig_camera_extrinsics(file.path()).expect("read rig");
        assert_eq!(cameras.len(), 2);
        let reference = cameras
            .iter()
            .find(|camera| camera.image_prefix == "lens0/")
            .unwrap();
        assert!(reference.ref_sensor);
        assert_eq!(reference.cam_from_rig_rotation, [1.0, 0.0, 0.0, 0.0]);
        let secondary = cameras
            .iter()
            .find(|camera| camera.image_prefix == "lens1/")
            .unwrap();
        assert!(!secondary.ref_sensor);
        assert_eq!(secondary.cam_from_rig_translation, [0.1, 0.0, 0.0]);
    }

    #[test]
    fn rejects_missing_non_reference_extrinsic_without_writing() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), true);
        let connection = Connection::open(file.path()).expect("open");
        connection
            .execute("DELETE FROM rig_sensors", [])
            .expect("delete extrinsic");
        let error = inject_gravity_priors(
            file.path(),
            &[GravityPriorInput {
                image_name: "lens1/frame.jpg".to_owned(),
                gravity: [0.0, 0.0, 1.0],
            }],
            &PriorCalibrationMetadata::default(),
        )
        .expect_err("missing extrinsic should reject");
        assert!(error.contains("sensor_from_rig"));
        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM pose_priors", [], |row| row.get(0))
            .expect("count");
        assert_eq!(count, 0);
    }

    #[test]
    fn marks_only_verified_real_focal_prior() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), false);
        let report = mark_focal_priors(file.path(), &[focal_input(1)]).expect("focal write");
        assert_eq!(report.marked_count, 1);
        assert_eq!(report.cameras_with_prior_focal_length, 1);
        assert!(report.focal_prior_valid);
        let default = FocalPriorInput {
            source: "default".to_owned(),
            ..focal_input(1)
        };
        let error = mark_focal_priors(file.path(), &[default]).expect_err("default rejected");
        assert!(error.contains("source"));
        let default_guess = FocalPriorInput {
            params: vec![38.4, 38.4, 64.0, 64.0, 0.0, 0.0, 0.0, 0.0],
            ..focal_input(1)
        };
        let error = mark_focal_priors(file.path(), &[default_guess])
            .expect_err("default focal factor guess rejected");
        assert!(error.contains("default_focal_length_factor"));
    }

    #[test]
    fn reads_complete_focal_prior_set_after_round_trip() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), true);
        let error = read_focal_prior_inputs(file.path(), "view_graph_calibrator")
            .expect_err("unmarked camera must reject");
        assert!(error.contains("prior_focal_length"));
        mark_focal_priors(file.path(), &[focal_input(1), focal_input(2)])
            .expect("mark both focal priors");
        let inputs = read_focal_prior_inputs(file.path(), "view_graph_calibrator")
            .expect("read complete focal priors");
        assert_eq!(inputs.len(), 2);
        assert!(inputs.iter().all(|input| input.verified));
        assert!(inputs
            .iter()
            .all(|input| input.source == "view_graph_calibrator"));
        let report = read_focal_prior_report(file.path(), "view_graph_calibrator")
            .expect("read focal coverage report");
        assert_eq!(report.focal_coverage_ratio, 1.0);
        assert_eq!(report.accepted_sources, vec!["view_graph_calibrator"]);
    }

    #[test]
    fn rejects_bad_schema_before_any_write() {
        let file = NamedTempFile::new().expect("temp");
        Connection::open(file.path())
            .expect("open")
            .execute_batch(
                "CREATE TABLE images(image_id INTEGER); CREATE TABLE cameras(camera_id INTEGER);",
            )
            .expect("bad schema");
        let error = inject_gravity_priors(file.path(), &[], &PriorCalibrationMetadata::default())
            .expect_err("schema mismatch should reject");
        assert!(error.contains("COLMAP schema"));
    }

    #[test]
    fn rejects_ambiguous_existing_pose_prior_mapping() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), false);
        let connection = Connection::open(file.path()).expect("open");
        let nan_position = nan_blob(3);
        let nan_covariance = nan_blob(9);
        let gravity = native_f64_blob(&[0.0, 0.0, 1.0]);
        for pose_prior_id in [10_i64, 11_i64] {
            connection
                .execute(
                    "INSERT INTO pose_priors
                     VALUES(?1,1,1,0,?2,?3,?4,-1)",
                    params![pose_prior_id, &nan_position, &nan_covariance, &gravity],
                )
                .expect("duplicate pose prior");
        }
        let error = inject_gravity_priors(
            file.path(),
            &[GravityPriorInput {
                image_name: "lens0/frame.jpg".to_owned(),
                gravity: [0.0, 0.0, 1.0],
            }],
            &PriorCalibrationMetadata::default(),
        )
        .expect_err("ambiguous prior mapping should reject");
        assert!(error.contains("不唯一"));
    }

    #[test]
    fn preserves_existing_finite_position_when_adding_gravity() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), false);
        let connection = Connection::open(file.path()).expect("open");
        let position = native_f64_blob(&[1.0, 2.0, 3.0]);
        let covariance = native_f64_blob(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        connection
            .execute(
                "INSERT INTO pose_priors VALUES(10,1,1,0,?1,?2,NULL,1)",
                params![position, covariance],
            )
            .expect("existing position prior");
        inject_gravity_priors(
            file.path(),
            &[GravityPriorInput {
                image_name: "lens0/frame.jpg".to_owned(),
                gravity: [0.0, 1.0, 0.0],
            }],
            &PriorCalibrationMetadata::default(),
        )
        .expect("gravity update");
        let (position_after, covariance_after, coordinate_system): (Vec<u8>, Vec<u8>, i64) =
            connection
                .query_row(
                    "SELECT position, position_covariance, coordinate_system
                     FROM pose_priors WHERE pose_prior_id=10",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .expect("read preserved position");
        assert_eq!(
            decode_native_f64_blob(&position_after, 3, "position").unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            decode_native_f64_blob(&covariance_after, 9, "covariance").unwrap(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(coordinate_system, 1);
    }

    #[test]
    fn marker_contains_pipeline_compatible_fields() {
        let file = NamedTempFile::new().expect("temp");
        fixture(file.path(), false);
        let report = inject_global_mapper_priors(
            file.path(),
            &[focal_input(1)],
            &[GravityPriorInput {
                image_name: "lens0/frame.jpg".to_owned(),
                gravity: [0.0, 0.0, 1.0],
            }],
            &PriorCalibrationMetadata {
                calibration_version: Some("hand-eye-v1".to_owned()),
                time_offset_ms: Some(1.25),
                require_complete_rig_extrinsics: true,
            },
        )
        .expect("combined write");
        assert!(report.marker.focal_prior_valid);
        assert!(report.marker.gravity_prior_valid);
        assert_eq!(
            report
                .marker
                .sensor_to_camera_calibration_version
                .as_deref(),
            Some("hand-eye-v1")
        );
        let marker_path = file.path().with_extension("marker.json");
        write_global_mapper_prior_marker(&marker_path, &report).expect("marker");
        let marker: serde_json::Value =
            serde_json::from_slice(&fs::read(marker_path).expect("read marker"))
                .expect("json marker");
        assert_eq!(marker["databasePosePriorsInjected"], true);
        assert_eq!(marker["sensorToCameraCalibrationVersion"], "hand-eye-v1");
    }
}
