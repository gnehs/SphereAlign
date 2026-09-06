//! Resumable, isolated native-grid drafts. No training activation or accepted pixels.
use super::{
    camera::{self, Camera, SIZE},
    dataset::{self, Frame},
    model::{self, GeometryModel, Heads},
    specialize,
};
use crate::masking::CancelToken;
use image::{GrayImage, Luma, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
const ENGINE: &str = "native-draft-v1-100deg-known-focal-nearest-seam-reject";
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    #[serde(default)]
    pub model_path: String,
    #[serde(default = "default_limit")]
    pub frame_limit: usize,
    #[serde(default = "default_threshold")]
    pub validity_threshold: f32,
}
fn default_limit() -> usize {
    8
}
fn default_threshold() -> f32 {
    0.5
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            frame_limit: 8,
            validity_threshold: 0.5,
        }
    }
}
impl Settings {
    fn validate(&self) -> Result<(), String> {
        if self.frame_limit > 1_000_000
            || !self.validity_threshold.is_finite()
            || !(0.1..=0.95).contains(&self.validity_threshold)
        {
            Err("Invalid frame limit or validity threshold".into())
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Preflight {
    pub registered: usize,
    pub selected: usize,
    pub native_pixels: u64,
    pub estimated_output_bytes: u64,
    pub model_download_bytes: u64,
    pub cameras: Vec<Camera>,
    pub model_identity: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameReport {
    pub id: u32,
    pub name: String,
    pub camera_id: u32,
    pub width: u32,
    pub height: u32,
    pub input_sha256: String,
    pub mask_sha256: Option<String>,
    pub valid_pixels: u64,
    pub total_pixels: u64,
    pub rejection_counts: [u64; 5],
    pub faces: Vec<FaceReport>,
    pub files: std::collections::BTreeMap<String, String>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceReport {
    pub face: usize,
    pub shift: Option<f64>,
    pub scale: f32,
    pub support_pixels: usize,
    pub error: Option<String>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub schema: u32,
    pub id: String,
    pub status: String,
    pub job_id: Option<String>,
    pub message: String,
    pub updated_ms: u64,
    pub settings: Settings,
    pub model_sha256: String,
    pub dataset_sha256: String,
    pub provider: String,
    pub runtime: String,
    pub quality_status: String,
    pub active_for_training: bool,
    pub total: usize,
    pub completed: usize,
    pub current: Option<String>,
    pub frames: Vec<FrameReport>,
    pub output_path: String,
    #[serde(default)]
    pub conventions: serde_json::Value,
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn io<T>(r: std::io::Result<T>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}
fn check(cancel: &CancelToken) -> Result<(), String> {
    if cancel.is_cancelled() {
        Err("Geometry cancelled; completed frames can be resumed".into())
    } else {
        Ok(())
    }
}
fn guard_output(root: &Path, path: &Path) -> Result<(), String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| "Output escapes dataset")?;
    let base = io(root.canonicalize())?;
    let mut current = root.to_path_buf();
    let mut expected = base;
    for part in relative.components() {
        if !matches!(part, std::path::Component::Normal(_)) {
            return Err("Invalid output component".into());
        }
        current.push(part);
        expected.push(part);
        if current.exists() && io(current.canonicalize())? != expected {
            return Err("Geometry output contains a symlink or junction; choose a regular dataset directory".into());
        }
    }
    Ok(())
}
fn save<T: Serialize>(path: &Path, data: &T) -> Result<(), String> {
    let tmp = path.with_extension("json.partial");
    let mut f = io(File::create(&tmp))?;
    io(f.write_all(&serde_json::to_vec_pretty(data).map_err(|e| e.to_string())?))?;
    io(f.sync_all())?;
    drop(f);
    io(fs::rename(tmp, path))
}
pub fn preflight(root: &Path, settings: &Settings) -> Result<Preflight, String> {
    settings.validate()?;
    guard_output(root, &root.join("geometry"))?;
    let d = dataset::read(root)?;
    let selected = if settings.frame_limit == 0 {
        d.frames.len()
    } else {
        settings.frame_limit.min(d.frames.len())
    };
    for frame in d.frames.iter().take(selected) {
        let camera = &d.cameras[&frame.camera_id];
        let path = dataset::safe_join(&root.join("images"), &frame.name)?;
        if image::image_dimensions(&path).map_err(|e| format!("{}: {e}", frame.name))?
            != (camera.width, camera.height)
        {
            return Err(format!(
                "{}: source dimensions disagree with final camera",
                frame.name
            ));
        }
        if let Some(path) = mask_path(root, &frame.name)? {
            if image::image_dimensions(path).map_err(|e| e.to_string())?
                != (camera.width, camera.height)
            {
                return Err(format!("{}: source mask dimensions mismatch", frame.name));
            }
        }
    }
    let pixels = d
        .frames
        .iter()
        .take(selected)
        .map(|f| {
            let c = &d.cameras[&f.camera_id];
            u64::from(c.width) * u64::from(c.height)
        })
        .sum::<u64>();
    Ok(Preflight {
        registered: d.frames.len(),
        selected,
        native_pixels: pixels,
        estimated_output_bytes: pixels * 32 + selected as u64 * 20_000_000,
        model_download_bytes: if settings.model_path.is_empty() {
            model::BYTES
        } else {
            0
        },
        cameras: d.cameras.into_values().collect(),
        model_identity: d.identity,
    })
}
pub fn list(root: &Path) -> Result<Vec<Report>, String> {
    let dir = root.join("geometry/runs");
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut result = vec![];
    for entry in io(fs::read_dir(dir))? {
        let p = io(entry)?.path().join("run.json");
        if !p.is_file() {
            continue;
        }
        let mut r: Report = serde_json::from_slice(&io(fs::read(p))?).map_err(|e| e.to_string())?;
        // OS locks are released on process death; a stale 'running' is resumable.
        if r.status == "running" {
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .open(root.join("geometry/writer.lock"));
            if let Ok(f) = lock {
                if f.try_lock().is_ok() {
                    r.status = "interrupted".into();
                    r.job_id = None;
                    r.message =
                        "Previous process stopped; resume checks every completed frame".into();
                }
            }
        }
        result.push(r);
    }
    result.sort_by_key(|r| std::cmp::Reverse(r.updated_ms));
    Ok(result)
}
pub fn preview(root: &Path, run: &str, frame: u32, kind: &str) -> Result<Vec<u8>, String> {
    if run.len() != 32 || !run.bytes().all(|v| v.is_ascii_hexdigit()) {
        return Err("Invalid run ID".into());
    }
    let allowed = [
        "rgb", "normal", "range", "validity", "reasons", "face0", "face1", "face2", "face3",
        "face4", "face5",
    ];
    if !allowed.contains(&kind) {
        return Err("Unknown preview kind".into());
    }
    let relative = format!("geometry/runs/{run}/frames/{frame}/{kind}.png");
    let p = dataset::safe_join(root, &relative)?;
    if io(fs::metadata(&p))?.len() > 100_000_000 {
        return Err("Preview exceeds size limit".into());
    }
    io(fs::read(p))
}
fn resolve_model(
    root: &Path,
    settings: &Settings,
    cancel: &CancelToken,
    progress: &mut impl FnMut(&str),
) -> Result<PathBuf, String> {
    if !settings.model_path.is_empty() {
        let p = PathBuf::from(&settings.model_path);
        if io(fs::metadata(&p))?.len() == specialize::DERIVED_BYTES {
            model::verify_hash(&p, specialize::DERIVED_HASH, specialize::DERIVED_BYTES)?;
            return Ok(p);
        }
        model::verify(&p)?;
    }
    let dir = root.join("geometry/models");
    guard_output(root, &dir)?;
    io(fs::create_dir_all(&dir))?;
    let derived = dir.join(format!("{}.onnx", specialize::DERIVED_HASH));
    guard_output(root, &derived)?;
    if derived.exists() {
        model::verify_hash(
            &derived,
            specialize::DERIVED_HASH,
            specialize::DERIVED_BYTES,
        )?;
        return Ok(derived);
    }
    let source = if !settings.model_path.is_empty() {
        PathBuf::from(&settings.model_path)
    } else {
        let p = dir.join(format!("{}.onnx", model::SHA256));
        if !p.exists() {
            progress("Downloading pinned MoGe model (420 MB)");
            let tmp = p.with_extension("download.partial");
            let url = format!(
                "https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/{}/model.onnx",
                model::REVISION
            );
            let result = (|| -> Result<(), String> {
                let mut response = ureq::get(&url).call().map_err(|e| e.to_string())?;
                let mut r = response.body_mut().as_reader();
                let mut f = io(File::create(&tmp))?;
                let mut b = [0u8; 65536];
                let mut total = 0u64;
                loop {
                    check(cancel)?;
                    let n = io(r.read(&mut b))?;
                    if n == 0 {
                        break;
                    }
                    total += n as u64;
                    if total > model::BYTES {
                        return Err("Model download exceeded pinned size".into());
                    }
                    io(f.write_all(&b[..n]))?;
                }
                io(f.sync_all())?;
                drop(f);
                model::verify(&tmp)?;
                io(fs::rename(&tmp, &p))
            })();
            if result.is_err() {
                let _ = fs::remove_file(&tmp);
            }
            result?;
        }
        model::verify(&p)?;
        p
    };
    check(cancel)?;
    progress("Preparing verified DirectML graph");
    specialize::prepare(&source, &derived)?;
    check(cancel)?;
    Ok(derived)
}
fn mask_path(root: &Path, name: &str) -> Result<Option<PathBuf>, String> {
    let native = Path::new(name)
        .with_extension("png")
        .to_string_lossy()
        .replace('\\', "/");
    for rel in [
        format!("masks/{native}"),
        format!("masks/{name}.png"),
        format!("masks/{name}"),
    ] {
        let p = root.join(&rel);
        if p.is_file() {
            return dataset::safe_join(root, &rel).map(Some);
        }
    }
    Ok(None)
}
/// Median linear estimate followed by robust reprojection refinement with known K.
/// Points contain an unknown additive Z shift. Raw point Z is never exported.
pub fn recover_shift(heads: &Heads, support: &[bool], threshold: f32) -> Result<f64, String> {
    let mut samples = vec![];
    let mut shifts = vec![];
    for y in (6..SIZE).step_by(12) {
        for x in (6..SIZE).step_by(12) {
            let i = y * SIZE + x;
            if !support[i] || heads.mask[i] < threshold || !heads.mask[i].is_finite() {
                continue;
            }
            let p = [
                heads.points[3 * i] as f64,
                heads.points[3 * i + 1] as f64,
                heads.points[3 * i + 2] as f64,
            ];
            if p.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let u = (x as f64 + 0.5 - SIZE as f64 / 2.) / camera::focal();
            let v = (y as f64 + 0.5 - SIZE as f64 / 2.) / camera::focal();
            let rr = u * u + v * v;
            if rr > 0.04 {
                let s = (u * p[0] + v * p[1]) / rr - p[2];
                if s.is_finite() {
                    shifts.push(s);
                    samples.push((p, u, v));
                }
            }
        }
    }
    if samples.len() < 24 {
        return Err("Insufficient valid support to recover known-focal depth".into());
    }
    shifts.sort_by(f64::total_cmp);
    let mut shift = shifts[shifts.len() / 2];
    for _ in 0..20 {
        let mut a = 0.;
        let mut b = 0.;
        for &(p, u, v) in &samples {
            let z = p[2] + shift;
            if z <= 1e-6 {
                continue;
            }
            let ex = p[0] / z - u;
            let ey = p[1] / z - v;
            let w = 1. / (1. + (ex * ex + ey * ey) / 0.01);
            let jx = -p[0] / (z * z);
            let jy = -p[1] / (z * z);
            a += w * (jx * jx + jy * jy);
            b += w * (jx * ex + jy * ey);
        }
        if a < 1e-12 {
            break;
        }
        let step = (b / a).clamp(-1., 1.);
        shift -= step;
        if step.abs() < 1e-7 {
            break;
        }
    }
    if !shift.is_finite() {
        return Err("Non-finite depth recovery".into());
    }
    Ok(shift)
}
struct Face {
    heads: Heads,
    support: Vec<bool>,
    shift: f64,
}
fn sample_rgb(src: &RgbImage, p: [f64; 2]) -> Option<Rgb<u8>> {
    let x = p[0] - 0.5;
    let y = p[1] - 0.5;
    if x < 0. || y < 0. || x > (src.width() - 1) as f64 || y > (src.height() - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(src.width() - 1);
    let y1 = (y0 + 1).min(src.height() - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    Some(Rgb(std::array::from_fn(|c| {
        let a = src.get_pixel(x0, y0)[c] as f64 * (1. - tx) + src.get_pixel(x1, y0)[c] as f64 * tx;
        let b = src.get_pixel(x0, y1)[c] as f64 * (1. - tx) + src.get_pixel(x1, y1)[c] as f64 * tx;
        (a * (1. - ty) + b * ty).round() as u8
    })))
}
fn predict_frame(
    root: &Path,
    out: &Path,
    camera: &Camera,
    frame: &Frame,
    input_hash: String,
    mask_hash: Option<String>,
    settings: &Settings,
    model: &mut GeometryModel,
    cancel: &CancelToken,
    progress: &mut impl FnMut(&str),
) -> Result<FrameReport, String> {
    let source = dataset::safe_join(&root.join("images"), &frame.name)?;
    let src = image::open(&source).map_err(|e| e.to_string())?.to_rgb8();
    if src.dimensions() != (camera.width, camera.height) {
        return Err(format!(
            "{}: source dimensions disagree with final camera",
            frame.name
        ));
    }
    let mask = mask_path(root, &frame.name)?
        .map(|p| {
            image::open(p)
                .map(|i| i.to_luma8())
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    if mask
        .as_ref()
        .is_some_and(|m| m.dimensions() != src.dimensions())
    {
        return Err("Source mask dimensions mismatch".into());
    }
    let mut faces: Vec<Option<Face>> = (0..6).map(|_| None).collect();
    let mut reports = vec![];
    for face in 0..6 {
        check(cancel)?;
        progress(&format!("{} · perspective {}/6", frame.name, face + 1));
        let mut rgb = RgbImage::new(SIZE as u32, SIZE as u32);
        let mut tensor = vec![0.; 3 * SIZE * SIZE];
        let mut support = vec![false; SIZE * SIZE];
        let mut n = 0;
        for y in 0..SIZE {
            if y % 32 == 0 {
                check(cancel)?;
            }
            for x in 0..SIZE {
                let i = y * SIZE + x;
                if let Some(p) = camera.pixel(camera::rotate(face, camera::face_ray(x, y))) {
                    if let Some(color) = sample_rgb(&src, p) {
                        rgb.put_pixel(x as u32, y as u32, color);
                        for c in 0..3 {
                            tensor[c * SIZE * SIZE + i] = color[c] as f32 / 255.;
                        }
                        let valid = mask.as_ref().is_none_or(|m| {
                            let xx = p[0].floor() as u32;
                            let yy = p[1].floor() as u32;
                            m.get_pixel(xx.min(m.width() - 1), yy.min(m.height() - 1))[0] >= 128
                        });
                        support[i] = valid;
                        if valid {
                            n += 1;
                        }
                    }
                }
            }
        }
        rgb.save(out.join(format!("face{face}.png")))
            .map_err(|e| e.to_string())?;
        if n < SIZE * SIZE / 100 {
            reports.push(FaceReport {
                face,
                shift: None,
                scale: 0.,
                support_pixels: n,
                error: Some("Insufficient source support".into()),
            });
            continue;
        }
        let heads = model.infer(tensor)?;
        check(cancel)?;
        match recover_shift(&heads, &support, settings.validity_threshold) {
            Ok(shift) => {
                reports.push(FaceReport {
                    face,
                    shift: Some(shift),
                    scale: heads.metric_scale,
                    support_pixels: n,
                    error: None,
                });
                faces[face] = Some(Face {
                    heads,
                    support,
                    shift,
                });
            }
            Err(e) => reports.push(FaceReport {
                face,
                shift: None,
                scale: heads.metric_scale,
                support_pixels: n,
                error: Some(e),
            }),
        }
    }
    check(cancel)?;
    progress(&format!("{} · gathering native pixels", frame.name));
    let mut normal = RgbImage::new(camera.width, camera.height);
    let mut depth = RgbImage::new(camera.width, camera.height);
    let mut validity = GrayImage::new(camera.width, camera.height);
    let mut reasons = RgbImage::new(camera.width, camera.height);
    let mut map = BufWriter::new(io(File::create(out.join("native.f32")))?);
    let mut reason_file = BufWriter::new(io(File::create(out.join("reasons.u8")))?);
    let mut counts = [0u64; 5];
    let mut scale_samples: Vec<f64> = faces
        .iter()
        .flatten()
        .flat_map(|f| {
            f.heads
                .points
                .chunks_exact(3)
                .step_by(1024)
                .map(move |p| (p[2] as f64 + f.shift) * f.heads.metric_scale as f64)
        })
        .filter(|z| z.is_finite() && *z > 0.)
        .collect();
    scale_samples.sort_by(f64::total_cmp);
    let display_max = scale_samples
        .get(scale_samples.len() * 9 / 10)
        .copied()
        .unwrap_or(1.)
        * 2.;
    for y in 0..camera.height {
        if y % 16 == 0 {
            check(cancel)?;
        }
        for x in 0..camera.width {
            let mut reason = 1u8;
            let mut value = [0f32; 4];
            if let Some(ray) = camera.ray(x as f64 + 0.5, y as f64 + 0.5) {
                reason = 2;
                if mask.as_ref().is_none_or(|m| m.get_pixel(x, y)[0] >= 128) {
                    reason = 3;
                    let mut candidates = [(0f64, [0f64; 3], 0f64); 6];
                    let mut candidate_count = 0;
                    for (f, face) in faces.iter().enumerate() {
                        let Some(face) = face else {
                            continue;
                        };
                        let Some((i, zray)) = camera::face_pixel(f, ray) else {
                            continue;
                        };
                        let p = &face.heads;
                        let z = (p.points[3 * i + 2] as f64 + face.shift) * p.metric_scale as f64;
                        if !face.support[i]
                            || !p.mask[i].is_finite()
                            || p.mask[i] < settings.validity_threshold
                            || !z.is_finite()
                            || z <= 0.
                            || z / zray > f32::MAX as f64
                        {
                            continue;
                        }
                        let Some(n) = camera::unit([
                            p.normal[3 * i] as f64,
                            p.normal[3 * i + 1] as f64,
                            p.normal[3 * i + 2] as f64,
                        ]) else {
                            continue;
                        };
                        candidates[candidate_count] = (zray, camera::rotate(f, n), z / zray);
                        candidate_count += 1;
                    }
                    let candidates = &mut candidates[..candidate_count];
                    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
                    if let Some(&(_, n, r)) = candidates.first() {
                        // Never blend depths/normals across disagreement at cube overlaps.
                        if candidates.iter().skip(1).any(|&(_, nn, rr)| {
                            camera::dot(n, nn) < 30f64.to_radians().cos()
                                || (r - rr).abs() / r.max(rr) > 0.20
                        }) {
                            reason = 4;
                        } else {
                            reason = 0;
                            value = [n[0] as f32, n[1] as f32, n[2] as f32, r as f32];
                            normal.put_pixel(
                                x,
                                y,
                                Rgb(n.map(|a| ((a * 0.5 + 0.5) * 255.).round() as u8)),
                            );
                            let t = (r / display_max).clamp(0., 1.);
                            depth.put_pixel(
                                x,
                                y,
                                Rgb([
                                    (255. * t) as u8,
                                    (255. * (1. - (2. * t - 1.).abs())) as u8,
                                    (255. * (1. - t)) as u8,
                                ]),
                            );
                            validity.put_pixel(x, y, Luma([255]));
                        }
                    }
                }
            }
            counts[reason as usize] += 1;
            reasons.put_pixel(
                x,
                y,
                Rgb([
                    [30, 160, 90],
                    [0, 0, 0],
                    [210, 140, 20],
                    [100, 100, 100],
                    [210, 50, 170],
                ][reason as usize]),
            );
            for v in value {
                io(map.write_all(&v.to_le_bytes()))?;
            }
            io(reason_file.write_all(&[reason]))?;
        }
    }
    io(map.flush())?;
    io(map.get_ref().sync_all())?;
    io(reason_file.flush())?;
    io(reason_file.get_ref().sync_all())?;
    for (name, img) in [
        ("rgb", src),
        ("normal", normal),
        ("range", depth),
        ("reasons", reasons),
    ] {
        img.save(out.join(format!("{name}.png")))
            .map_err(|e| e.to_string())?;
    }
    validity
        .save(out.join("validity.png"))
        .map_err(|e| e.to_string())?;
    let mut files = std::collections::BTreeMap::new();
    for entry in io(fs::read_dir(out))? {
        let entry = io(entry)?;
        files.insert(
            entry.file_name().to_string_lossy().into_owned(),
            dataset::hash(&entry.path())?,
        );
    }
    let report = FrameReport {
        id: frame.id,
        name: frame.name.clone(),
        camera_id: camera.id,
        width: camera.width,
        height: camera.height,
        input_sha256: input_hash,
        mask_sha256: mask_hash,
        valid_pixels: counts[0],
        total_pixels: u64::from(camera.width) * u64::from(camera.height),
        rejection_counts: counts,
        faces: reports,
        files,
    };
    save(&out.join("frame.json"), &report)?;
    Ok(report)
}
fn reusable(dir: &Path, input: &str, mask: &Option<String>) -> Option<FrameReport> {
    let r: FrameReport = serde_json::from_slice(&fs::read(dir.join("frame.json")).ok()?).ok()?;
    if r.input_sha256 != input || &r.mask_sha256 != mask || r.files.len() != 13 {
        return None;
    }
    for (name, hash) in &r.files {
        let p = dataset::safe_join(dir, name).ok()?;
        if dataset::hash(&p).ok().as_ref() != Some(hash) {
            return None;
        }
    }
    Some(r)
}
pub fn run(
    root: &Path,
    settings: Settings,
    cancel: &CancelToken,
    job_id: Option<String>,
    mut notify: impl FnMut(&Report),
) -> Result<Report, String> {
    settings.validate()?;
    let d = dataset::read(root)?;
    let base = root.join("geometry");
    guard_output(root, &base)?;
    guard_output(root, &base.join("writer.lock"))?;
    io(fs::create_dir_all(&base))?;
    let lock = io(OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(base.join("writer.lock")))?;
    lock.try_lock()
        .map_err(|_| "Geometry is already running for this dataset")?;
    let total = if settings.frame_limit == 0 {
        d.frames.len()
    } else {
        settings.frame_limit.min(d.frames.len())
    };
    let mut hashes = vec![];
    let mut h = Sha256::new();
    h.update(ENGINE);
    h.update(ort::info());
    h.update("DirectML");
    h.update(specialize::DERIVED_HASH);
    h.update(&d.identity);
    h.update(settings.validity_threshold.to_le_bytes());
    h.update((total as u64).to_le_bytes());
    for f in d.frames.iter().take(total) {
        check(cancel)?;
        let ih = dataset::hash(&dataset::safe_join(&root.join("images"), &f.name)?)?;
        let mh = mask_path(root, &f.name)?
            .map(|p| dataset::hash(&p))
            .transpose()?;
        h.update(f.name.as_bytes());
        h.update(&ih);
        h.update(mh.as_deref().unwrap_or("no-mask"));
        hashes.push((ih, mh));
    }
    let id = format!("{:x}", h.finalize())[..32].to_owned();
    let dir = base.join("runs").join(&id);
    guard_output(root, &dir.join("frames"))?;
    guard_output(root, &dir.join("run.json"))?;
    io(fs::create_dir_all(dir.join("frames")))?;
    let mut report = Report {
        schema: 1,
        id,
        status: "running".into(),
        job_id,
        message: "Preparing pinned model".into(),
        updated_ms: now(),
        settings: settings.clone(),
        model_sha256: specialize::DERIVED_HASH.into(),
        dataset_sha256: d.identity.clone(),
        provider: "DirectML".into(),
        runtime: ort::info().into(),
        quality_status: "single_view_unverified".into(),
        active_for_training: false,
        total,
        completed: 0,
        current: None,
        frames: vec![],
        output_path: dir.to_string_lossy().into_owned(),
        conventions: serde_json::json!({"native.f32":"row-major little-endian float32 [height,width,4]: nx,ny,nz,ray_range; invalid=all zero","normalFrame":"source camera OpenCV axes; model sign not trainer-certified","rangeScale":"model estimate, not aligned to COLMAP, not certified metric","reasons.u8":{"0":"valid support, unverified quality","1":"outside source lens","2":"source mask","3":"model or recovery invalid","4":"perspective disagreement"},"faceSize":SIZE,"faceFovDegrees":100,"sourceFromFace":camera::FACES,"nativeSampling":"pixel-centre nearest valid face; no depth blending","overlapRangeTolerance":0.20,"overlapNormalDegrees":30,"engine":ENGINE}),
    };
    save(&dir.join("run.json"), &report)?;
    notify(&report);
    let result = (|| -> Result<(), String> {
        let path = resolve_model(root, &settings, cancel, &mut |message| {
            report.message = message.into();
            report.updated_ms = now();
            let _ = save(&dir.join("run.json"), &report);
            notify(&report);
        })?;
        let mut model: Option<GeometryModel> = None;
        for (index, frame) in d.frames.iter().take(total).enumerate() {
            check(cancel)?;
            report.current = Some(frame.name.clone());
            let target = dir.join("frames").join(frame.id.to_string());
            let (input_hash, mask_hash) = &hashes[index];
            guard_output(root, &target)?;
            let frame_report = if let Some(r) = reusable(&target, input_hash, mask_hash) {
                r
            } else {
                if model.is_none() {
                    report.message = "Loading DirectML session".into();
                    save(&dir.join("run.json"), &report)?;
                    notify(&report);
                    model = Some(GeometryModel::load(&path, "DirectML")?);
                }
                let tmp = dir.join("frames").join(format!("{}.partial", frame.id));
                guard_output(root, &tmp)?;
                if tmp.exists() {
                    io(fs::remove_dir_all(&tmp))?;
                }
                io(fs::create_dir(&tmp))?;
                let r = predict_frame(
                    root,
                    &tmp,
                    &d.cameras[&frame.camera_id],
                    frame,
                    input_hash.clone(),
                    mask_hash.clone(),
                    &settings,
                    model.as_mut().unwrap(),
                    cancel,
                    &mut |message| {
                        report.message = message.into();
                        report.updated_ms = now();
                        let _ = save(&dir.join("run.json"), &report);
                        notify(&report);
                    },
                )?;
                // A complete frame becomes visible only after all outputs and hashes exist.
                check(cancel)?;
                if target.exists() {
                    io(fs::remove_dir_all(&target))?;
                }
                io(fs::rename(&tmp, &target))?;
                r
            };
            // Refuse source changes made by an external writer during inference.
            if dataset::hash(&dataset::safe_join(&root.join("images"), &frame.name)?)?
                != *input_hash
                || mask_path(root, &frame.name)?
                    .map(|p| dataset::hash(&p))
                    .transpose()?
                    != *mask_hash
            {
                return Err(
                    "Source image or mask changed during inference; rerun with fresh inputs".into(),
                );
            }
            report.frames.push(frame_report);
            report.completed = index + 1;
            report.updated_ms = now();
            save(&dir.join("run.json"), &report)?;
            notify(&report);
        }
        if dataset::read(root)?.identity != d.identity {
            return Err("Final COLMAP model changed during inference".into());
        }
        Ok(())
    })();
    report.status = if cancel.is_cancelled() {
        "cancelled"
    } else if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .into();
    report.job_id = None;
    report.current = None;
    report.updated_ms = now();
    report.message = result
        .as_ref()
        .err()
        .cloned()
        .unwrap_or_else(|| "Draft ready for quality review; training remains disabled".into());
    save(&dir.join("run.json"), &report)?;
    notify(&report);
    result.map(|_| report)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(root: &Path) {
        fs::create_dir_all(root.join("sparse/0")).unwrap();
        fs::write(root.join("sparse/0/cameras.txt"),"3 OPENCV_FISHEYE 32 32 8.5 8.5 16 16 0 0 0 0\n91 OPENCV_FISHEYE 32 32 8.5 8.5 16 16 0 0 0 0\n").unwrap();
        fs::write(
            root.join("sparse/0/images.txt"),
            "101 1 0 0 0 0 0 0 3 lens0/frame.png\n\n901 0 0 1 0 0 0 0 91 lens1/frame.png\n\n",
        )
        .unwrap();
        fs::write(
            root.join("sparse/0/points3D.txt"),
            "# no anchors; explicitly unverified\n",
        )
        .unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(root.join("images").join(lens)).unwrap();
            RgbImage::from_fn(32, 32, |x, y| {
                Rgb([(x * 7) as u8, (y * 7) as u8, ((x + y) * 3) as u8])
            })
            .save(root.join("images").join(lens).join("frame.png"))
            .unwrap();
        }
    }
    #[test]
    fn masks_follow_spherealign_png_naming() {
        let d = tempfile::tempdir().unwrap();
        fs::create_dir_all(d.path().join("masks/lens0")).unwrap();
        fs::write(d.path().join("masks/lens0/frame.png"), b"mask").unwrap();
        assert_eq!(
            mask_path(d.path(), "lens0/frame.jpg").unwrap().unwrap(),
            d.path()
                .join("masks/lens0/frame.png")
                .canonicalize()
                .unwrap()
        );
    }
    #[test]
    fn input_change_and_writer_lock_are_detected_before_model_load() {
        let d = tempfile::tempdir().unwrap();
        fixture(d.path());
        let before = dataset::read(d.path()).unwrap().identity;
        fs::write(
            d.path().join("sparse/0/points3D.txt"),
            b"# changed final model\n",
        )
        .unwrap();
        assert_ne!(dataset::read(d.path()).unwrap().identity, before);
        fs::create_dir(d.path().join("geometry")).unwrap();
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .read(true)
            .open(d.path().join("geometry/writer.lock"))
            .unwrap();
        lock.try_lock().unwrap();
        let result = run(
            d.path(),
            Settings::default(),
            &CancelToken::new(),
            None,
            |_| {},
        );
        assert!(result.err().unwrap().contains("already running"));
        assert!(!d.path().join("geometry/models").exists());
    }
    #[test]
    fn corrupted_cache_is_not_reused() {
        let d = tempfile::tempdir().unwrap();
        let mut files = std::collections::BTreeMap::new();
        for i in 0..13 {
            let n = format!("part{i}");
            let p = d.path().join(&n);
            fs::write(&p, [i as u8]).unwrap();
            files.insert(n, dataset::hash(&p).unwrap());
        }
        let report = FrameReport {
            id: 101,
            name: "lens0/frame.png".into(),
            camera_id: 3,
            width: 32,
            height: 32,
            input_sha256: "input".into(),
            mask_sha256: None,
            valid_pixels: 0,
            total_pixels: 1024,
            rejection_counts: [0, 1024, 0, 0, 0],
            faces: vec![],
            files,
        };
        save(&d.path().join("frame.json"), &report).unwrap();
        assert!(reusable(d.path(), "input", &None).is_some());
        assert!(reusable(d.path(), "changed source", &None).is_none());
        fs::write(d.path().join("part3"), b"tampered").unwrap();
        assert!(reusable(d.path(), "input", &None).is_none());
    }
    #[test]
    #[ignore = "Requires explicit GEOMETRY_TEST_MODEL and Windows DirectML GPU"]
    fn directml_cancel_resume_native_end_to_end() {
        let model_path = std::env::var("GEOMETRY_TEST_MODEL")
            .expect("Set GEOMETRY_TEST_MODEL to verified native static graph");
        let d = tempfile::tempdir().unwrap();
        fixture(d.path());
        let settings = Settings {
            model_path,
            frame_limit: 2,
            validity_threshold: 0.5,
        };
        let source_before = dataset::hash(&d.path().join("images/lens0/frame.png")).unwrap();
        let cancel = CancelToken::new();
        let first = run(
            d.path(),
            settings.clone(),
            &cancel,
            Some("test-job".into()),
            |r| {
                if r.completed == 1 {
                    cancel.cancel();
                }
            },
        );
        assert!(first.is_err());
        let cancelled = list(d.path()).unwrap().remove(0);
        assert_eq!(cancelled.status, "cancelled");
        assert_eq!(cancelled.completed, 1);
        assert!(!cancelled.active_for_training);
        let first_file = PathBuf::from(&cancelled.output_path).join("frames/101/native.f32");
        let modified = fs::metadata(&first_file).unwrap().modified().unwrap();
        let done = run(
            d.path(),
            settings.clone(),
            &CancelToken::new(),
            None,
            |_| {},
        )
        .unwrap();
        assert_eq!(done.id, cancelled.id);
        assert_eq!(done.completed, 2);
        assert_eq!(
            fs::metadata(&first_file).unwrap().modified().unwrap(),
            modified
        );
        let mut loaded_gpu = false;
        run(d.path(), settings, &CancelToken::new(), None, |r| {
            if r.message == "Loading DirectML session" {
                loaded_gpu = true;
            }
        })
        .unwrap();
        assert!(
            !loaded_gpu,
            "Fully verified cache must not start another GPU session"
        );
        for frame in done.frames {
            let bytes = fs::read(
                PathBuf::from(&done.output_path).join(format!("frames/{}/native.f32", frame.id)),
            )
            .unwrap();
            assert_eq!(bytes.len(), 32 * 32 * 4 * 4);
            assert!(bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
            assert_eq!(frame.rejection_counts.iter().sum::<u64>(), 1024);
        }
        assert_eq!(
            dataset::hash(&d.path().join("images/lens0/frame.png")).unwrap(),
            source_before
        );
        assert!(!d.path().join("normals").exists());
        assert!(!d.path().join("depths").exists());
    }
    #[test]
    fn known_focal_shift_recovers_plane() {
        let mut h = Heads {
            points: vec![0.; SIZE * SIZE * 3],
            normal: vec![],
            mask: vec![1.; SIZE * SIZE],
            metric_scale: 1.,
        };
        let shift = 2.3;
        for y in 0..SIZE {
            for x in 0..SIZE {
                let i = y * SIZE + x;
                let r = camera::face_ray(x, y);
                let z = 4. / (1. + 0.2 * r[0] / r[2]);
                h.points[3 * i] = (z * r[0] / r[2]) as f32;
                h.points[3 * i + 1] = (z * r[1] / r[2]) as f32;
                h.points[3 * i + 2] = (z - shift) as f32;
            }
        }
        let got = recover_shift(&h, &vec![true; SIZE * SIZE], 0.5).unwrap();
        assert!((got - shift).abs() < 1e-5);
        assert!(recover_shift(&h, &vec![false; SIZE * SIZE], 0.5).is_err());
    }
}
