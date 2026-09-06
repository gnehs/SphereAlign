//! Product-facing quality audit. Successful SfM is a computation state, never
//! visual acceptance. Keep coverage, native geometry and data identity separate.

use crate::masking::{CancelToken, FisheyeCamera};
use crate::reconstruction_benchmark::{self as model_io, ColmapImageRecord};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const REPORT_PATH: &str = "metadata/reconstruction_quality.json";
const COVERAGE_TARGET: f64 = 0.85;
const MAX_NATIVE_ERROR_PX: f64 = 4.0;
const MIN_RIG_OBSERVATIONS: usize = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceQuality {
    pub source: String,
    pub expected_frames: usize,
    pub complete_frames: usize,
    pub coverage_ratio: f64,
    pub longest_missing_run: usize,
    pub weak_frames: Vec<String>,
    pub path_jump_frames: Vec<String>,
    pub median_step_model_units: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationAudit {
    pub feature_coordinates_checked: u64,
    pub track_observations_checked: u64,
    pub invalid_feature_coordinates: u64,
    pub invalid_track_references: u64,
    pub invalid_depth_or_projection: u64,
    pub masked_observations: u64,
    pub reprojection_outliers: u64,
    pub median_reprojection_error_px: Option<f64>,
    pub p95_reprojection_error_px: Option<f64>,
    pub max_reprojection_error_px: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactIdentity { path: String, size: u64, modified_ns: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityReport {
    pub schema_version: u32,
    pub status: String,
    pub checked_at_ms: u64,
    pub coverage_target: f64,
    pub native_error_threshold_px: f64,
    pub minimum_rig_observations: usize,
    pub expected_frames: usize,
    pub complete_frames: usize,
    pub sources: Vec<SourceQuality>,
    pub observations: ObservationAudit,
    pub issues: Vec<String>,
    pub examples: Vec<String>,
    pub review_checklist: Vec<String>,
    pub artifact_identity: Vec<ArtifactIdentity>,
}

fn cancelled(cancel: &CancelToken) -> Result<(), String> {
    if cancel.is_cancelled() { Err("cancelled".into()) } else { Ok(()) }
}

fn image_names(folder: &Path) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(folder).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_file() { continue; }
        if entry.path().extension().and_then(|v| v.to_str()).is_some_and(|v|
            matches!(v.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")) {
            names.insert(entry.file_name().to_str().ok_or("invalid image filename")?.to_owned());
        }
    }
    Ok(names)
}

/// Stop before feature extraction if there is a missing lens, wrong image
/// size, corrupt/non-binary mask, or a mismatched selection timestamp/name.
pub fn validate_capture(root: &Path, use_masks: bool, cancel: &CancelToken) -> Result<Vec<String>, String> {
    let left = image_names(&root.join("images/lens0"))?;
    let right = image_names(&root.join("images/lens1"))?;
    if left.is_empty() || left != right {
        return Err(format!("雙鏡頭影像必須同名成對：lens0={}、lens1={}，不成對={}；請重新抽幀", left.len(), right.len(), left.symmetric_difference(&right).count()));
    }
    let mut dimensions = BTreeMap::new();
    for lens in ["lens0", "lens1"] { for name in &left {
        cancelled(cancel)?;
        let relative = format!("{lens}/{name}");
        let path = root.join("images").join(&relative);
        let dims = image::image_dimensions(&path).map_err(|e| format!("{relative}: {e}"))?;
        let group = format!("{lens}/{}", source(name));
        if dimensions.insert(group, dims).is_some_and(|previous| previous != dims) {
            return Err(format!("{relative} 的尺寸與同一來源鏡頭不一致"));
        }
        if use_masks {
            let mask_path = root.join("masks").join(Path::new(&relative).with_extension("png"));
            let mask = image::open(&mask_path).map_err(|e| format!("遮罩缺失或損壞 {}: {e}", mask_path.display()))?;
            if (mask.width(), mask.height()) != dims || mask.color() != image::ColorType::L8
                || mask.as_bytes().iter().any(|v| !matches!(v, 0 | 255)) {
                return Err(format!("{relative} 的遮罩必須是原始尺寸的二值灰階 PNG"));
            }
        }
    }}
    // Source selection metadata is optional for imported COLMAP datasets.
    for entry in fs::read_dir(root.join("metadata")).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("source") || !name.ends_with("_selection.json") { continue; }
        let value: serde_json::Value = serde_json::from_slice(&fs::read(entry.path()).map_err(|e| e.to_string())?)
            .map_err(|e| format!("{name}: {e}"))?;
        let Some(rows) = value.get("selections").and_then(|v| v.as_array()) else { return Err(format!("{name} 缺少 selections")); };
        let mut last_time = None;
        let mut selected_names = BTreeSet::new();
        for row in rows.iter().filter(|row| row.get("selected").and_then(|v| v.as_bool()) == Some(true)) {
            let file = |key| row.get(key).and_then(|v| v.as_str()).and_then(|v| v.rsplit(['/', '\\']).next());
            let (Some(a), Some(b)) = (file("output_lens0"), file("output_lens1")) else { return Err(format!("{name} 缺少同步影格輸出名稱")); };
            let time = row.get("timestamp_ms").and_then(|v| v.as_f64()).filter(|v| v.is_finite() && *v >= 0.0)
                .ok_or_else(|| format!("{name}: {a} 時間戳無效"))?;
            if a != b || !left.contains(a) || !selected_names.insert(a.to_owned()) || last_time.is_some_and(|last| time <= last) {
                return Err(format!("{name}: {a} 雙鏡頭 identity 或時間順序不一致"));
            }
            last_time = Some(time);
        }
    }
    Ok(left.into_iter().collect())
}

fn source(name: &str) -> &str { name.split_once('_').map_or("unknown", |(source, _)| source) }
fn example(report: &mut QualityReport, value: String) { if report.examples.len() < 40 { report.examples.push(value); } }

fn rotation(q: [f64; 4], p: [f64; 3]) -> [f64; 3] {
    let [w, x, y, z] = q;
    [ (1.0-2.0*(y*y+z*z))*p[0] + 2.0*(x*y-z*w)*p[1] + 2.0*(x*z+y*w)*p[2],
      2.0*(x*y+z*w)*p[0] + (1.0-2.0*(x*x+z*z))*p[1] + 2.0*(y*z-x*w)*p[2],
      2.0*(x*z-y*w)*p[0] + 2.0*(y*z+x*w)*p[1] + (1.0-2.0*(x*x+y*y))*p[2] ]
}

fn center(image: &ColmapImageRecord) -> [f64; 3] {
    let q = image.qvec_camera_from_world;
    rotation([q[0], -q[1], -q[2], -q[3]], image.tvec_camera_from_world.map(|v| -v))
}
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 { (0..3).map(|i| (a[i]-b[i]).powi(2)).sum::<f64>().sqrt() }
fn quantile(values: &mut [f64], fraction: f64) -> Option<f64> {
    if values.is_empty() { return None; }
    let index = ((values.len()-1) as f64 * fraction).round() as usize;
    Some(*values.select_nth_unstable_by(index, f64::total_cmp).1)
}

fn artifact_identity(root: &Path) -> Result<Vec<ArtifactIdentity>, String> {
    let mut paths = vec![PathBuf::from("database.db")];
    for path in ["database.db-wal", "metadata/mask_run.json"] {
        if root.join(path).is_file() { paths.push(PathBuf::from(path)); }
    }
    for folder in ["images/lens0", "images/lens1", "masks/lens0", "masks/lens1"] {
        if root.join(folder).is_dir() {
            for name in image_names(&root.join(folder))? { paths.push(Path::new(folder).join(name)); }
        }
    }
    for folder in ["sparse/0", "metadata/final-model-text"] {
        for name in ["cameras", "images", "points3D", "rigs", "frames"] {
            for extension in ["txt", "bin"] {
                let path = Path::new(folder).join(format!("{name}.{extension}"));
                if root.join(&path).is_file() { paths.push(path); }
            }
        }
    }
    paths.into_iter().map(|path| {
        let metadata = fs::metadata(root.join(&path)).map_err(|e| e.to_string())?;
        Ok(ArtifactIdentity { path: path.to_string_lossy().replace('\\', "/"), size: metadata.len(),
            modified_ns: metadata.modified().map_err(|e| e.to_string())?.duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos().to_string() })
    }).collect()
}

pub fn read_report(root: &Path) -> Result<Option<QualityReport>, String> {
    let path = root.join(REPORT_PATH);
    if !path.is_file() { return Ok(None); }
    let mut report: QualityReport = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if artifact_identity(root).ok().as_ref() != Some(&report.artifact_identity) {
        report.status = "stale".into();
    }
    Ok(Some(report))
}

pub fn audit(root: &Path, use_masks: bool, cancel: &CancelToken) -> Result<QualityReport, String> {
    let starting_identity = artifact_identity(root)?;
    let names = image_names(&root.join("images/lens0"))?;
    let model_dir = root.join("metadata/final-model-text");
    let model = model_io::read_colmap_text_model(&model_dir)?;
    let mut report = QualityReport {
        schema_version: 1, status: "needs_visual_review".into(),
        checked_at_ms: SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_millis() as u64,
        coverage_target: COVERAGE_TARGET, native_error_threshold_px: MAX_NATIVE_ERROR_PX,
        minimum_rig_observations: MIN_RIG_OBSERVATIONS, expected_frames: names.len(), complete_frames: 0,
        sources: Vec::new(), observations: ObservationAudit::default(), issues: model.warnings.clone(), examples: Vec::new(),
        review_checklist: vec![
            "Check doors, walls and corridor topology against source images from several positions.".into(),
            "Check cross-capture scale and static anchors; low reprojection error does not validate global scale.".into(),
            "Inspect people, clothing, shoes, screens and changed furniture in masks and renders.".into(),
            "Keep one global COLMAP coordinate frame and complete rigs for tile training; never realign tiles independently.".into(),
            "Use overlapping training halos, crop final Gaussians to disjoint cores, and inspect both sides of every occupied seam.".into(),
            "Keep train_frame=points and evaluation images. Training completion and PSNR do not approve visual quality.".into(),
            "Treat gravity alignment as orientation evidence only; it does not establish metric scale or translation.".into(),
        ], artifact_identity: Vec::new(),
    };
    let images = model.images.iter().map(|v| (v.image_id, v)).collect::<HashMap<_, _>>();
    let images_by_name = model.images.iter().map(|v| (v.name.as_str(), v)).collect::<HashMap<_, _>>();
    let points = model.points3d.iter().map(|v| (v.point3d_id, v)).collect::<HashMap<_, _>>();
    if images.len() != model.images.len() || points.len() != model.points3d.len() || images_by_name.len() != images.len() {
        report.issues.push("Duplicate model image/point identity".into());
    }
    if !model.missing_files.is_empty() { report.issues.push(format!("Missing model files: {}", model.missing_files.join(", "))); }
    let mut cameras = HashMap::new();
    for record in &model.cameras {
        let camera = if record.model == "OPENCV_FISHEYE" && record.params.len() == 8 && record.width <= u32::MAX as u64 && record.height <= u32::MAX as u64 {
            let camera = FisheyeCamera { width: record.width as u32, height: record.height as u32, params: record.params.clone().try_into().unwrap() };
            camera.validate().ok().map(|_| camera)
        } else { None };
        if let Some(camera) = camera { cameras.insert(record.camera_id, camera); }
        else { report.issues.push(format!("Unsupported or invalid native calibration: camera {}", record.camera_id)); }
    }
    let total_track_refs: u64 = model.points3d.iter().map(|p| p.track.len() as u64).sum();
    for point in &model.points3d {
        cancelled(cancel)?;
        let mut seen = HashSet::new();
        for observation in &point.track {
            if !images.contains_key(&observation.image_id) || !seen.insert((observation.image_id, observation.point2d_index)) {
                report.observations.invalid_track_references += 1;
            }
        }
    }
    let db = Connection::open_with_flags(root.join("database.db"), OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| e.to_string())?;
    let mut query = db.prepare("SELECT i.name,i.camera_id,k.rows,k.cols,k.data FROM images i LEFT JOIN keypoints k ON k.image_id=i.image_id WHERE i.image_id=?1").map_err(|e| e.to_string())?;
    let mut lines = BufReader::new(File::open(model_dir.join("images.txt")).map_err(|e| e.to_string())?).lines();
    let mut seen_images = HashSet::new();
    let mut supported = HashMap::<u64, usize>::new();
    let mut errors = Vec::new();
    while let Some(line) = lines.next() {
        cancelled(cancel)?;
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() || line.trim_start().starts_with('#') { continue; }
        let fields: Vec<_> = line.split_whitespace().collect();
        let image_id: u64 = fields.first().ok_or("empty image header")?.parse().map_err(|_| "invalid image id")?;
        let image = images.get(&image_id).ok_or("image header missing from parsed model")?;
        if !seen_images.insert(image_id) { report.issues.push(format!("Repeated image header: {}", image.name)); }
        let observations = lines.next().ok_or("missing image observations line")?.map_err(|e| e.to_string())?;
        let fields: Vec<_> = observations.split_whitespace().collect();
        if fields.len() % 3 != 0 { return Err(format!("{}: invalid observation triple", image.name)); }
        let row = query.query_row([i64::try_from(image_id).map_err(|_| "image id overflow")?], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?,
            row.get::<_, Option<i64>>(2)?, row.get::<_, Option<i64>>(3)?, row.get::<_, Option<Vec<u8>>>(4)?)))
            .optional().map_err(|e| e.to_string())?;
        let mut keypoints = None;
        if let Some((db_name, camera_id, Some(rows), Some(cols), Some(data))) = row {
            if db_name == image.name && camera_id >= 0 && camera_id as u64 == image.camera_id && rows >= 0 && matches!(cols, 2 | 4 | 6)
                && rows.checked_mul(cols).and_then(|n| n.checked_mul(4)) == Some(data.len() as i64) && rows as usize == fields.len()/3 {
                keypoints = Some((cols as usize, data));
            }
        }
        if keypoints.is_none() { report.issues.push(format!("Model/database feature namespace mismatch: {}", image.name)); }
        let pose_valid = image.qvec_camera_from_world.iter().chain(image.tvec_camera_from_world.iter()).all(|v| v.is_finite())
            && (image.qvec_camera_from_world.iter().map(|v| v*v).sum::<f64>() - 1.0).abs() < 1e-5;
        if !pose_valid { report.issues.push(format!("Invalid camera pose: {}", image.name)); }
        let mask = if use_masks { Some(image::open(root.join("masks").join(Path::new(&image.name).with_extension("png")))
            .map_err(|e| format!("{}: {e}", image.name))?.into_luma8()) } else { None };
        let mut support = 0;
        for (index, observation) in fields.chunks_exact(3).enumerate() {
            let parse = |value: &str| value.parse::<f64>().ok().filter(|v| v.is_finite());
            let xy = parse(observation[0]).zip(parse(observation[1]));
            report.observations.feature_coordinates_checked += 1;
            let aligned = xy.zip(keypoints.as_ref()).is_some_and(|((x,y), (cols,data))| {
                let offset = index*cols*4;
                let fx = f32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as f64;
                let fy = f32::from_le_bytes(data[offset+4..offset+8].try_into().unwrap()) as f64;
                fx.is_finite() && fy.is_finite() && (x-fx).abs() <= 0.001 && (y-fy).abs() <= 0.001
            });
            if !aligned {
                report.observations.invalid_feature_coordinates += 1;
                example(&mut report, format!("{}: feature {index} differs from database", image.name));
            }
            if observation[2] == "-1" { continue; }
            report.observations.track_observations_checked += 1;
            let point = observation[2].parse::<u64>().ok().and_then(|id| points.get(&id));
            let Some(point) = point else { report.observations.invalid_track_references += 1; continue; };
            if !point.track.iter().any(|v| v.image_id == image_id && v.point2d_index == index as u64) {
                report.observations.invalid_track_references += 1;
            }
            let camera = cameras.get(&image.camera_id);
            let p = rotation(image.qvec_camera_from_world, point.xyz);
            let p = std::array::from_fn(|i| p[i] + image.tvec_camera_from_world[i]);
            let projection = if pose_valid { camera.and_then(|camera| camera.project(p)) } else { None };
            let Some(((x,y), projected)) = xy.zip(projection) else {
                report.observations.invalid_depth_or_projection += 1; continue;
            };
            let inside = camera.is_some_and(|c| x >= 0.0 && y >= 0.0 && x < c.width as f64 && y < c.height as f64);
            if !inside { report.observations.invalid_depth_or_projection += 1; continue; }
            let masked = mask.as_ref().is_some_and(|mask| x >= mask.width() as f64 || y >= mask.height() as f64
                || mask.get_pixel(x.floor() as u32, y.floor() as u32)[0] == 0);
            if masked { report.observations.masked_observations += 1; }
            let error = (x-projected[0]).hypot(y-projected[1]);
            if !error.is_finite() { report.observations.invalid_depth_or_projection += 1; continue; }
            errors.push(error);
            if error > MAX_NATIVE_ERROR_PX { report.observations.reprojection_outliers += 1; }
            else if aligned && !masked { support += 1; }
        }
        supported.insert(image_id, support);
    }
    if seen_images.len() != images.len() { report.issues.push("Not all model images were audited".into()); }
    if total_track_refs != report.observations.track_observations_checked {
        report.observations.invalid_track_references += total_track_refs.abs_diff(report.observations.track_observations_checked);
    }
    report.observations.max_reprojection_error_px = errors.iter().copied().max_by(f64::total_cmp);
    report.observations.median_reprojection_error_px = quantile(&mut errors, 0.5);
    report.observations.p95_reprojection_error_px = quantile(&mut errors, 0.95);
    let mut groups = BTreeMap::<&str, Vec<&String>>::new();
    for name in &names { groups.entry(source(name)).or_default().push(name); }
    for (source, names) in groups {
        let mut quality = SourceQuality { source: source.into(), expected_frames: names.len(), complete_frames: 0,
            coverage_ratio: 0.0, longest_missing_run: 0, weak_frames: Vec::new(), path_jump_frames: Vec::new(), median_step_model_units: None };
        let mut missing_run = 0;
        let mut path = Vec::new();
        for (index, name) in names.iter().enumerate() {
            let left = images_by_name.get(format!("lens0/{name}").as_str());
            let right = images_by_name.get(format!("lens1/{name}").as_str());
            if let Some((left, right)) = left.zip(right).filter(|(a,b)| a.frame_id.is_some() && a.frame_id == b.frame_id) {
                quality.complete_frames += 1;
                missing_run = 0;
                let count = supported.get(&left.image_id).copied().unwrap_or(0) + supported.get(&right.image_id).copied().unwrap_or(0);
                if count < MIN_RIG_OBSERVATIONS { quality.weak_frames.push((*name).clone()); }
                let position = center(left);
                if position.iter().all(|v| v.is_finite()) { path.push((index, name.as_str(), position)); }
            } else { missing_run += 1; quality.longest_missing_run = quality.longest_missing_run.max(missing_run); }
        }
        quality.coverage_ratio = quality.complete_frames as f64 / quality.expected_frames.max(1) as f64;
        let mut steps: Vec<f64> = path.windows(2).map(|p| distance(p[0].2, p[1].2)/(p[1].0-p[0].0) as f64).filter(|v| *v > 1e-12).collect();
        quality.median_step_model_units = quantile(&mut steps, 0.5);
        if let Some(median) = quality.median_step_model_units {
            for trio in path.windows(3) {
                let a = distance(trio[0].2, trio[1].2); let b = distance(trio[1].2, trio[2].2);
                if a/(trio[1].0-trio[0].0) as f64 > median*10.0 && b/(trio[2].0-trio[1].0) as f64 > median*10.0
                    && distance(trio[0].2, trio[2].2) < (a+b)*0.25 {
                    quality.path_jump_frames.push(trio[1].1.into());
                }
            }
        }
        if quality.coverage_ratio < COVERAGE_TARGET { report.issues.push(format!("{}: coverage {:.1}% is below the {:.0}% screening target", quality.source, quality.coverage_ratio*100.0, COVERAGE_TARGET*100.0)); }
        if !quality.weak_frames.is_empty() { report.issues.push(format!("{}: {} rig frames have fewer than {MIN_RIG_OBSERVATIONS} valid observations", quality.source, quality.weak_frames.len())); }
        if !quality.path_jump_frames.is_empty() { report.issues.push(format!("{}: {} possible isolated path jumps", quality.source, quality.path_jump_frames.len())); }
        report.complete_frames += quality.complete_frames;
        report.sources.push(quality);
    }
    let obs = &report.observations;
    if obs.invalid_feature_coordinates > 0 || obs.invalid_track_references > 0 { report.issues.push("Model and feature/track identities are inconsistent; rebuild from the matching feature database".into()); }
    if obs.invalid_depth_or_projection > 0 { report.issues.push("Non-finite or non-positive-depth native observations detected".into()); }
    if obs.masked_observations > 0 { report.issues.push("Old observations remain inside excluded mask regions; rerun feature extraction and matching".into()); }
    if obs.reprojection_outliers > 0 { report.issues.push("Native reprojection outliers exceed the 4 px screening gate".into()); }
    if obs.track_observations_checked == 0 { report.issues.push("Model has no 3D observations".into()); }
    if !report.issues.is_empty() { report.status = "issues_found".into(); }
    report.artifact_identity = artifact_identity(root)?;
    if report.artifact_identity != starting_identity {
        return Err("重建資料在品質檢查期間發生變更，請停止其他寫入後重新檢查".into());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb};
    use tempfile::TempDir;

    fn fixture() -> TempDir {
        let temp = TempDir::new().unwrap();
        let root = temp.path();
        let text = root.join("metadata/final-model-text"); fs::create_dir_all(&text).unwrap();
        fs::create_dir_all(root.join("sparse/0")).unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(root.join("images").join(lens)).unwrap();
            fs::create_dir_all(root.join("masks").join(lens)).unwrap();
        }
        let db = Connection::open(root.join("database.db")).unwrap();
        db.execute_batch("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER); CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);").unwrap();
        fs::write(text.join("cameras.txt"), "1 OPENCV_FISHEYE 8 8 3 3 4 4 0 0 0 0\n2 OPENCV_FISHEYE 8 8 3 3 4 4 0 0 0 0\n").unwrap();
        fs::write(text.join("rigs.txt"), "1 2 CAMERA 1 CAMERA 2 1 1 0 0 0 0 0 0\n").unwrap();
        let camera = FisheyeCamera { width: 8, height: 8, params: [3.,3.,4.,4.,0.,0.,0.,0.] };
        let points: Vec<_> = (0..16).map(|i| [(i%4) as f64*0.1-0.15,(i/4) as f64*0.1-0.15,4.0]).collect();
        let mut image_text = String::new(); let mut frames = String::new();
        for frame in 0..4 {
            let name = format!("source000_{:08}.png", frame+1);
            frames.push_str(&format!("{} 1 1 0 0 0 0 0 0 2 CAMERA 1 {} CAMERA 2 {}\n", frame+1,frame*2+1,frame*2+2));
            for lens in 0..2 {
                let id = frame*2+lens+1;
                let relative = format!("lens{lens}/{name}");
                ImageBuffer::<Rgb<u8>, _>::from_pixel(8,8,Rgb([100,100,100])).save(root.join("images").join(&relative)).unwrap();
                ImageBuffer::<Luma<u8>, _>::from_pixel(8,8,Luma([255])).save(root.join("masks").join(&relative)).unwrap();
                image_text.push_str(&format!("{id} 1 0 0 0 0 0 0 {} {relative}\n",lens+1));
                let mut blob = Vec::new();
                for (index, xyz) in points.iter().enumerate() {
                    let [x,y] = camera.project(*xyz).unwrap();
                    image_text.push_str(&format!("{x} {y} {} ",index+1));
                    blob.extend_from_slice(&(x as f32).to_le_bytes()); blob.extend_from_slice(&(y as f32).to_le_bytes());
                }
                image_text.push('\n');
                db.execute("INSERT INTO images VALUES (?1,?2,?3)",rusqlite::params![id,relative,lens+1]).unwrap();
                db.execute("INSERT INTO keypoints VALUES (?1,16,2,?2)",rusqlite::params![id,blob]).unwrap();
            }
        }
        fs::write(text.join("images.txt"), image_text).unwrap();
        fs::write(text.join("frames.txt"), frames).unwrap();
        let mut point_text = String::new();
        for (index, xyz) in points.iter().enumerate() {
            point_text.push_str(&format!("{} {} {} {} 100 100 100 0",index+1,xyz[0],xyz[1],xyz[2]));
            for id in 1..=8 { point_text.push_str(&format!(" {id} {index}")); }
            point_text.push('\n');
        }
        fs::write(text.join("points3D.txt"), point_text).unwrap();
        temp
    }

    #[test]
    fn consistent_geometry_requires_visual_review_and_checks_all_observations() {
        let temp = fixture();
        assert_eq!(validate_capture(temp.path(),true,&CancelToken::new()).unwrap().len(),4);
        let report = audit(temp.path(),true,&CancelToken::new()).unwrap();
        assert_eq!(report.status,"needs_visual_review", "{:?}",report.issues);
        assert_eq!(report.complete_frames,4);
        assert_eq!(report.observations.track_observations_checked,128);
        assert_eq!(report.observations.invalid_feature_coordinates,0);
        assert_eq!(report.observations.invalid_track_references,0);
    }

    #[test]
    fn catches_feature_namespace_mismatch_and_masked_old_tracks() {
        let temp = fixture(); let root = temp.path();
        let db = Connection::open(root.join("database.db")).unwrap();
        let mut blob: Vec<u8> = db.query_row("SELECT data FROM keypoints WHERE image_id=1",[],|row|row.get(0)).unwrap();
        blob[0..4].copy_from_slice(&1.0_f32.to_le_bytes());
        db.execute("UPDATE keypoints SET data=?1 WHERE image_id=1",[blob]).unwrap();
        ImageBuffer::<Luma<u8>, _>::from_pixel(8,8,Luma([0])).save(root.join("masks/lens0/source000_00000001.png")).unwrap();
        let report = audit(root,true,&CancelToken::new()).unwrap();
        assert_eq!(report.status,"issues_found");
        assert_eq!(report.observations.invalid_feature_coordinates,1);
        assert_eq!(report.observations.masked_observations,16);
        assert_eq!(report.sources[0].weak_frames.len(),1);
    }

    #[test]
    fn audits_each_source_even_when_other_sources_are_complete_and_marks_stale_reports() {
        let temp = fixture(); let root = temp.path();
        for lens in ["lens0","lens1"] {
            ImageBuffer::<Rgb<u8>, _>::from_pixel(8,8,Rgb([100,100,100])).save(root.join(format!("images/{lens}/source001_00000001.png"))).unwrap();
        }
        let report = audit(root,false,&CancelToken::new()).unwrap();
        assert_eq!(report.sources[1].coverage_ratio,0.0);
        assert_eq!(report.sources[1].longest_missing_run,1);
        assert_eq!(report.status,"issues_found");
        fs::write(root.join(REPORT_PATH),serde_json::to_vec(&report).unwrap()).unwrap();
        assert_eq!(read_report(root).unwrap().unwrap().status,"issues_found");
        fs::write(root.join("sparse/0/images.bin"),b"changed model").unwrap();
        assert_eq!(read_report(root).unwrap().unwrap().status,"stale");
    }

    #[test]
    fn preflight_rejects_unpaired_frames_and_nonbinary_masks() {
        let temp = fixture(); let root = temp.path();
        ImageBuffer::<Luma<u8>, _>::from_pixel(8,8,Luma([128])).save(root.join("masks/lens0/source000_00000001.png")).unwrap();
        assert!(validate_capture(root,true,&CancelToken::new()).is_err());
        fs::remove_file(root.join("images/lens1/source000_00000001.png")).unwrap();
        assert!(validate_capture(root,false,&CancelToken::new()).is_err());
    }
}
