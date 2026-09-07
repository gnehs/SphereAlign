//! Only fixed, reproducible intermediates may be removed. PNG products remain hash-verified.
use super::{
    dataset,
    draft::{check, guard_output, save, FrameReport},
    CancelToken,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub(super) const OUTPUTS: [&str; 13] = [
    "native.f32",
    "reasons.u8",
    "rgb.png",
    "normal.png",
    "range.png",
    "validity.png",
    "reasons.png",
    "face0.png",
    "face1.png",
    "face2.png",
    "face3.png",
    "face4.png",
    "face5.png",
];
const INTERMEDIATES: [&str; 2] = ["native.f32", "reasons.u8"];

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IntermediateState {
    #[default]
    Retained,
    CleanupPending,
    Removed,
}

fn verify_manifest(frame: &FrameReport) -> Result<(), String> {
    if frame.files.len() != OUTPUTS.len()
        || OUTPUTS.iter().any(|name| !frame.files.contains_key(*name))
    {
        return Err("Incomplete Geometry output manifest".into());
    }
    if frame.width == 0
        || frame.height == 0
        || u64::from(frame.width) * u64::from(frame.height) > 64_000_000
    {
        return Err("Invalid Geometry frame dimensions".into());
    }
    Ok(())
}

fn verify_file(dir: &Path, frame: &FrameReport, name: &str) -> Result<(), String> {
    let path = dataset::safe_join(dir, name)?;
    if dataset::hash(&path)? != frame.files[name] {
        return Err(format!("Geometry output changed: {name}"));
    }
    Ok(())
}

pub(super) fn verify_cache(
    dir: &Path,
    frame: &FrameReport,
    require_intermediates: bool,
) -> Result<(), String> {
    verify_manifest(frame)?;
    if require_intermediates && frame.intermediate_state != IntermediateState::Retained {
        return Err("Intermediate files were cleaned; regenerate to retain them".into());
    }
    for name in OUTPUTS {
        if INTERMEDIATES.contains(&name)
            && frame.intermediate_state != IntermediateState::Retained
            && !dir.join(name).exists()
        {
            continue;
        }
        verify_file(dir, frame, name)?;
    }
    Ok(())
}

pub(super) fn clean_frame(
    root: &Path,
    dir: &Path,
    frame: &mut FrameReport,
    cancel: &CancelToken,
) -> Result<(), String> {
    check(cancel)?;
    guard_output(root, dir)?;
    guard_output(root, &dir.join("frame.json"))?;
    guard_output(root, &dir.join("frame.json.partial"))?;
    verify_manifest(frame)?;
    // Check every retained product before authorizing any deletion. The source/native hashes
    // were already checked by inference or cache reuse; no source file is ever a cleanup target.
    for name in OUTPUTS.into_iter().filter(|n| !INTERMEDIATES.contains(n)) {
        check(cancel)?;
        verify_file(dir, frame, name)?;
    }
    let pixels = u64::from(frame.width) * u64::from(frame.height);
    let mut targets = Vec::new();
    for (name, expected_bytes) in [("native.f32", pixels * 16), ("reasons.u8", pixels)] {
        let path = dir.join(name);
        guard_output(root, &path)?;
        match fs::metadata(&path) {
            Ok(meta) if meta.is_file() && meta.len() == expected_bytes => targets.push(path),
            Err(e)
                if e.kind() == std::io::ErrorKind::NotFound
                    && frame.intermediate_state != IntermediateState::Retained => {}
            _ => return Err(format!("Missing or unexpected intermediate file: {name}")),
        }
    }
    if frame.intermediate_state == IntermediateState::Removed && targets.is_empty() {
        return Ok(());
    }
    // Persist intent before unlinking. A crash/cancel between files is a recognized cache
    // state, and the next normal-only run finishes cleanup instead of starting GPU inference.
    frame.intermediate_state = IntermediateState::CleanupPending;
    save(&dir.join("frame.json"), frame)?;
    for path in targets {
        check(cancel)?;
        guard_output(root, &path)?;
        fs::remove_file(path).map_err(|e| e.to_string())?;
    }
    frame.intermediate_state = IntermediateState::Removed;
    frame.removed_intermediate_bytes = pixels * 17;
    save(&dir.join("frame.json"), frame)
}
