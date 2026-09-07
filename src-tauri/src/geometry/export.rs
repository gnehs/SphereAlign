//! Publish complete, source-matched camera-space normal PNGs for Spirula.
use super::{
    dataset,
    draft::{self, Report},
    CancelToken,
};
use image::ImageDecoder;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path},
};

const MANIFEST: &str = ".spherealign.json";
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Export {
    schema: u32,
    run_id: String,
    dataset_sha256: String,
    files: BTreeMap<String, String>,
    convention: String,
}
fn io<T>(result: std::io::Result<T>) -> Result<T, String> {
    result.map_err(|e| e.to_string())
}

fn normal_name(name: &str) -> Result<String, String> {
    if name.contains(['\\', ':'])
        || name.is_empty()
        || Path::new(name)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err("Unsafe normal-map filename".into());
    }
    Ok(Path::new(name)
        .with_extension("png")
        .to_string_lossy()
        .replace('\\', "/"))
}

/// Called with the dataset writer lock held. Old maps are archived, never deleted.
pub(super) fn publish(root: &Path, report: &Report, cancel: &CancelToken) -> Result<(), String> {
    let data = dataset::read(root)?;
    if data.identity != report.dataset_sha256
        || data.frames.len() != report.total
        || report.completed != report.total
        || report.frames.len() != report.total
    {
        return Err(
            "Training normals require a complete run matching the final reconstruction".into(),
        );
    }
    let run = root.join("geometry/runs").join(&report.id);
    let target = root.join("normals");
    draft::guard_output(root, &target.join(MANIFEST))?;
    let expected = report
        .frames
        .iter()
        .map(|r| {
            Ok((
                normal_name(&r.name)?,
                r.files
                    .get("normal.png")
                    .ok_or("Normal PNG hash missing")?
                    .clone(),
            ))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let reuse = target.is_dir() && matches_files(root, &target, &expected, cancel)?;
    let staging = root
        .join("geometry/exports")
        .join(&report.id)
        .join("normals.partial");
    draft::guard_output(root, &staging)?;
    io(fs::create_dir_all(&staging))?;
    let mut files = BTreeMap::new();
    for frame in &data.frames {
        draft::check(cancel)?;
        let r = report
            .frames
            .iter()
            .find(|r| r.id == frame.id)
            .ok_or("Missing registered normal map")?;
        let camera = &data.cameras[&frame.camera_id];
        if r.name != frame.name
            || r.camera_id != frame.camera_id
            || (r.width, r.height) != (camera.width, camera.height)
            || dataset::hash(&dataset::safe_join(&root.join("images"), &frame.name)?)?
                != r.input_sha256
            || draft::mask_path(root, &frame.name)?
                .map(|p| dataset::hash(&p))
                .transpose()?
                != r.mask_sha256
        {
            return Err("Normal-map source or calibration changed; regenerate normals".into());
        }
        let name = normal_name(&frame.name)?;
        let sha = r.files.get("normal.png").ok_or("Normal PNG hash missing")?;
        if files.insert(name.to_lowercase(), sha.clone()).is_some() {
            return Err("Normal-map filenames collide after replacing image extensions".into());
        }
        let source = run
            .join("frames")
            .join(frame.id.to_string())
            .join("normal.png");
        draft::guard_output(root, &source)?;
        if dataset::hash(&source)? != *sha {
            return Err("Normal PNG hash mismatch".into());
        }
        let image = image::ImageReader::open(&source)
            .map_err(|e| e.to_string())?
            .into_decoder()
            .map_err(|e| e.to_string())?;
        if image.color_type() != image::ColorType::Rgb8 || image.dimensions() != (r.width, r.height)
        {
            return Err("Normal PNG must be RGB8 at the original camera resolution".into());
        }
        if !reuse {
            let target = staging.join(&name);
            draft::guard_output(root, &target)?;
            io(fs::create_dir_all(target.parent().unwrap()))?;
            io(fs::copy(&source, &target))?;
            if dataset::hash(&target)? != *sha {
                return Err("Copied normal PNG hash mismatch".into());
            }
        }
    }
    // Preserve original case in the manifest, after checking case-insensitive collisions.
    let files = report
        .frames
        .iter()
        .map(|r| Ok((normal_name(&r.name)?, r.files["normal.png"].clone())))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let export = Export { schema: 1, run_id: report.id.clone(), dataset_sha256: data.identity.clone(), files,
        convention: "RGB8 camera-space xyz = 2*rgb/255-1; invalid RGB=(0,0,0); source camera axes; no sign flip".into() };
    let manifest_dir = if reuse { &target } else { &staging };
    draft::guard_output(root, &manifest_dir.join(MANIFEST))?;
    draft::guard_output(root, &manifest_dir.join(".spherealign.json.partial"))?;
    draft::check(cancel)?;
    if dataset::read(root)?.identity != data.identity {
        return Err("COLMAP changed before normal publication".into());
    }
    draft::save(&manifest_dir.join(MANIFEST), &export)?;
    draft::guard_output(root, &target)?;
    if !reuse && !matches_files(root, &staging, &export.files, cancel)? {
        return Err("Unexpected files in normal export staging directory".into());
    }
    if !reuse && target.exists() {
        // An unowned folder is accepted only when it contains exactly these maps.
        if !target.join(MANIFEST).is_file() && !matches_files(root, &target, &export.files, cancel)?
        {
            return Err("Existing normals folder is not managed by SphereAlign and differs from this run; move it aside before exporting".into());
        }
        archive_locked(root)?;
    }
    if !reuse {
        io(fs::rename(&staging, &target))?;
    }
    let config = root.join("geometry/spirula-normals.json");
    draft::guard_output(root, &config)?;
    draft::guard_output(root, &config.with_extension("json.partial"))?;
    draft::save(
        &config,
        &serde_json::json!({
            "preset": "360-camera", "data": root.to_string_lossy(),
            "normal_dir": "normals", "load_normals": true, "load_depths": false,
            "load_masks": true, "mask_boundary_offset": -0.025, "warp_to_pinhole": false,
            "primitive": "3dgs", "train_frame": "points",
            "normal_supervision_weight": 0.01, "num_iterations": 20000,
            "train_resolution_divisor": 0, "sh_degree": 1, "cap_max": 3000000,
            "cache_images": "cpu", "disable_viewer": true, "keep_viewer_alive": false
        }),
    )?;
    Ok(())
}

fn matches_files(
    root: &Path,
    dir: &Path,
    expected: &BTreeMap<String, String>,
    cancel: &CancelToken,
) -> Result<bool, String> {
    fn collect(
        root: &Path,
        base: &Path,
        dir: &Path,
        files: &mut BTreeMap<String, String>,
        cancel: &CancelToken,
    ) -> Result<(), String> {
        draft::guard_output(root, dir)?;
        for entry in io(fs::read_dir(dir))? {
            draft::check(cancel)?;
            let path = io(entry)?.path();
            draft::guard_output(root, &path)?;
            if path == base.join(MANIFEST) {
                continue;
            }
            if path.is_dir() {
                collect(root, base, &path, files, cancel)?;
            } else {
                files.insert(
                    path.strip_prefix(base)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                    dataset::hash(&path)?,
                );
            }
        }
        Ok(())
    }
    let mut actual = BTreeMap::new();
    collect(root, dir, dir, &mut actual, cancel)?;
    Ok(actual == *expected)
}

fn archive_locked(root: &Path) -> Result<(), String> {
    let target = root.join("normals");
    draft::guard_output(root, &target)?;
    let archive_root = root.join("geometry/exports/previous");
    draft::guard_output(root, &archive_root)?;
    io(fs::create_dir_all(&archive_root))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let archive = archive_root.join(format!("{}-{stamp}", std::process::id()));
    draft::guard_output(root, &archive)?;
    io(fs::rename(target, archive))
}

/// Invalidate managed priors before an upstream stage changes their source data.
pub fn invalidate(root: &Path) -> Result<(), String> {
    if !root.exists() {
        return Ok(());
    }
    let manifest = root.join("normals").join(MANIFEST);
    draft::guard_output(root, &manifest)?;
    if !manifest.is_file() {
        return Ok(());
    }
    let _lock = draft::writer_lock(root)?;
    archive_locked(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_names_keep_lens_identity_and_reject_traversal() {
        assert_eq!(normal_name("lens1/房間.jpg").unwrap(), "lens1/房間.png");
        for name in ["../a.png", "C:/a.jpg", "lens\\a.jpg", "/a.png", ""] {
            assert!(normal_name(name).is_err());
        }
    }
}
