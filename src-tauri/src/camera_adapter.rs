//! Camera-specific source discovery normalized into capture bundles.
//!
//! The pipeline consumes [`CaptureBundle`] values and never infers lens
//! geometry from a filename extension.  Container-specific pairing remains at
//! this boundary so older Insta360 `_00_`/`_10_` recordings and newer
//! dual-track INSV recordings expose the same two-lens contract.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const SUPPORTED_SOURCE_EXTENSIONS: &[&str] = &[
    "osv", "insv", "mp4", "mov", "mkv", "avi", "webm", "m4v", "mts", "m2ts", "ts",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstaPairRole {
    Lens0,
    Lens1,
}

#[derive(Debug, Clone)]
pub struct ProbedSource {
    pub path: PathBuf,
    pub probe: Value,
    pub camera_model: Option<String>,
    pub color_profile: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LensStream {
    pub lens_id: &'static str,
    pub source_path: PathBuf,
    pub ffmpeg_stream_index: usize,
}

#[derive(Debug, Clone)]
pub struct CaptureBundle {
    pub adapter: &'static str,
    pub vendor: &'static str,
    pub model: Option<String>,
    pub source_paths: Vec<PathBuf>,
    pub source_probes: Vec<ProbedSource>,
    pub lenses: [LensStream; 2],
    pub telemetry_path: PathBuf,
    pub probe: Value,
    pub native_fisheye: bool,
    pub factory_intrinsics: bool,
    pub rig_extrinsics: bool,
}

/// A camera-family initialization pose used only when the two physical lenses
/// cannot bootstrap independently. This is deliberately separate from
/// `rig_extrinsics`: it is a nominal bootstrap prior rather than factory
/// calibration. Its zero baseline is only an initialization; mapper bundle
/// adjustment must remain free to recover the real CMOS-center offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigBootstrapPoseHint {
    pub cam_from_rig_rotation: [f64; 4],
    pub cam_from_rig_translation: [f64; 3],
    pub provenance: &'static str,
    pub refine_sensor_from_rig: bool,
}

/// Return the nominal back-to-back layout defined by an adapter. Native
/// dual-fisheye tracks from the supported 360 camera families use opposite
/// optical axes with the reference-lens convention represented by a
/// 180-degree Y rotation. The physical lens baseline is intentionally
/// initialized at zero because monocular SfM has no metric scale at this
/// point. Keeping a refinable rig from the first mapper pass is substantially
/// safer than deriving it after two independently drifting reconstructions.
pub fn rig_bootstrap_pose_hint(adapter: &str) -> Option<RigBootstrapPoseHint> {
    let (provenance, refine_sensor_from_rig) = match adapter {
        "Insta360DualTrackAdapter" | "Insta360PairedInsvAdapter" => {
            ("insta360-adapter-nominal-back-to-back-v1", false)
        }
        "DjiOsmo360Adapter" => ("dji-osmo-360-adapter-nominal-back-to-back-v1", true),
        _ => return None,
    };
    Some(RigBootstrapPoseHint {
        cam_from_rig_rotation: [0.0, 0.0, 1.0, 0.0],
        cam_from_rig_translation: [0.0, 0.0, 0.0],
        provenance,
        refine_sensor_from_rig,
    })
}

impl CaptureBundle {
    pub fn primary_path(&self) -> &Path {
        &self.lenses[0].source_path
    }

    pub fn display_name(&self) -> String {
        self.source_paths
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

pub fn is_supported_source(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .is_some_and(|extension| SUPPORTED_SOURCE_EXTENSIONS.contains(&extension.as_str()))
}

fn is_insv(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("insv"))
}

fn insta_pair_marker(path: &Path) -> Option<(String, InstaPairRole, usize)> {
    if !is_insv(path) {
        return None;
    }
    let file_name = path.file_name()?.to_str()?;
    let lower = file_name.to_ascii_lowercase();
    let lens0 = lower.rfind("_00_");
    let lens1 = lower.rfind("_10_");
    match (lens0, lens1) {
        (Some(index), None) => Some((file_name.to_owned(), InstaPairRole::Lens0, index)),
        (None, Some(index)) => Some((file_name.to_owned(), InstaPairRole::Lens1, index)),
        (Some(left), Some(right)) if left > right => {
            Some((file_name.to_owned(), InstaPairRole::Lens0, left))
        }
        (Some(_), Some(right)) => Some((file_name.to_owned(), InstaPairRole::Lens1, right)),
        (None, None) => None,
    }
}

fn insta_pair_key(path: &Path) -> Option<(PathBuf, InstaPairRole)> {
    let (mut file_name, role, marker_index) = insta_pair_marker(path)?;
    file_name.replace_range(marker_index..marker_index + 4, "_xx_");
    Some((
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(file_name),
        role,
    ))
}

pub fn insta_pair_sibling(path: &Path) -> Option<PathBuf> {
    let (mut file_name, role, marker_index) = insta_pair_marker(path)?;
    let replacement = match role {
        InstaPairRole::Lens0 => "_10_",
        InstaPairRole::Lens1 => "_00_",
    };
    file_name.replace_range(marker_index..marker_index + 4, replacement);
    Some(
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(file_name),
    )
}

/// Add an existing Insta360 counterpart when the user selected only one half.
/// Paths are sorted and deduplicated so manifests remain deterministic.
pub fn expand_related_sources(paths: &mut Vec<PathBuf>) {
    let mut expanded = paths.clone();
    for path in paths.iter() {
        if let Some(sibling) = insta_pair_sibling(path).filter(|candidate| candidate.is_file()) {
            expanded.push(sibling);
        }
    }
    expanded.sort();
    expanded.dedup();
    *paths = expanded;
}

fn video_streams(probe: &Value) -> Vec<&Value> {
    probe
        .get("streams")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"))
        .collect()
}

fn stream_index(stream: &Value) -> Option<usize> {
    stream
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
}

fn parse_fraction(value: Option<&str>) -> Option<f64> {
    let value = value?.trim();
    if value.is_empty() || value == "0/0" || value.eq_ignore_ascii_case("N/A") {
        return None;
    }
    if let Some((numerator, denominator)) = value.split_once('/') {
        let numerator = numerator.parse::<f64>().ok()?;
        let denominator = denominator.parse::<f64>().ok()?;
        return (denominator.abs() > f64::EPSILON).then_some(numerator / denominator);
    }
    value.parse::<f64>().ok()
}

fn stream_fps(stream: &Value) -> Option<f64> {
    parse_fraction(stream.get("avg_frame_rate").and_then(Value::as_str))
        .or_else(|| parse_fraction(stream.get("r_frame_rate").and_then(Value::as_str)))
}

fn format_duration(probe: &Value) -> Option<f64> {
    probe
        .pointer("/format/duration")
        .and_then(Value::as_str)
        .and_then(|duration| duration.parse::<f64>().ok())
}

fn validate_lens_stream_pair(left: &Value, right: &Value, context: &str) -> Result<(), String> {
    let left_size = (
        left.get("width").and_then(Value::as_u64),
        left.get("height").and_then(Value::as_u64),
    );
    let right_size = (
        right.get("width").and_then(Value::as_u64),
        right.get("height").and_then(Value::as_u64),
    );
    if left_size != right_size {
        return Err(format!(
            "{context} 的鏡頭解析度不同：lens0 {left_size:?}、lens1 {right_size:?}"
        ));
    }
    if let (Some(left_fps), Some(right_fps)) = (stream_fps(left), stream_fps(right)) {
        if (left_fps - right_fps).abs() > 0.01 {
            return Err(format!(
                "{context} 的 FPS 不同：lens0 {left_fps:.6}、lens1 {right_fps:.6}"
            ));
        }
    }
    Ok(())
}

fn validate_paired_streams(left: &ProbedSource, right: &ProbedSource) -> Result<(), String> {
    let left_video = video_streams(&left.probe);
    let right_video = video_streams(&right.probe);
    let left_stream = left_video
        .first()
        .ok_or_else(|| "Insta360 lens0 配對檔沒有可辨識的 video stream".to_owned())?;
    let right_stream = right_video
        .first()
        .ok_or_else(|| "Insta360 lens1 配對檔沒有可辨識的 video stream".to_owned())?;
    if left_video.len() != 1 || right_video.len() != 1 {
        return Err("Insta360 雙檔配對的每個檔案都必須各自包含一路 video stream".to_owned());
    }
    validate_lens_stream_pair(left_stream, right_stream, "Insta360 配對檔")?;
    if let (Some(left_duration), Some(right_duration)) =
        (format_duration(&left.probe), format_duration(&right.probe))
    {
        if (left_duration - right_duration).abs() > 1.0 {
            return Err(format!(
                "Insta360 配對檔的長度相差超過一秒：lens0 {left_duration:.3}s、lens1 {right_duration:.3}s"
            ));
        }
    }
    Ok(())
}

fn dual_track_bundle(source: ProbedSource, indices: [usize; 2]) -> CaptureBundle {
    let vendor = if is_insv(&source.path) {
        "Insta360"
    } else if source
        .path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("osv"))
    {
        "DJI"
    } else {
        "Unknown"
    };
    let adapter = match vendor {
        "Insta360" => "Insta360DualTrackAdapter",
        "DJI" => "DjiOsmo360Adapter",
        _ => "GenericDualFisheyeAdapter",
    };
    let model = source
        .camera_model
        .clone()
        .or_else(|| (vendor == "Insta360").then(|| "dual-track INSV".to_owned()));
    let path = source.path.clone();
    CaptureBundle {
        adapter,
        vendor,
        model,
        source_paths: vec![path.clone()],
        source_probes: vec![source.clone()],
        lenses: [
            LensStream {
                lens_id: "lens0",
                source_path: path.clone(),
                ffmpeg_stream_index: indices[0],
            },
            LensStream {
                lens_id: "lens1",
                source_path: path.clone(),
                ffmpeg_stream_index: indices[1],
            },
        ],
        telemetry_path: path,
        probe: source.probe,
        native_fisheye: true,
        factory_intrinsics: false,
        rig_extrinsics: false,
    }
}

/// Normalize probed files into two-lens captures.
pub fn resolve_capture_bundles(sources: Vec<ProbedSource>) -> Result<Vec<CaptureBundle>, String> {
    let mut by_path = BTreeMap::new();
    for source in sources {
        by_path.insert(source.path.clone(), source);
    }
    let mut paired = BTreeMap::<PathBuf, (Option<PathBuf>, Option<PathBuf>)>::new();
    for path in by_path.keys() {
        if video_streams(&by_path[path].probe).len() == 1 {
            if let Some((key, role)) = insta_pair_key(path) {
                let entry = paired.entry(key).or_default();
                match role {
                    InstaPairRole::Lens0 => entry.0 = Some(path.clone()),
                    InstaPairRole::Lens1 => entry.1 = Some(path.clone()),
                }
            }
        }
    }

    let mut consumed = BTreeSet::new();
    let mut captures = Vec::new();
    for path in by_path.keys().cloned().collect::<Vec<_>>() {
        if consumed.contains(&path) {
            continue;
        }
        let streams = video_streams(&by_path[&path].probe);
        if streams.len() >= 2 {
            validate_lens_stream_pair(streams[0], streams[1], "雙 track 來源")?;
            let indices = [
                stream_index(streams[0]).ok_or_else(|| {
                    format!("{} 的第一路 video stream 沒有有效 index", path.display())
                })?,
                stream_index(streams[1]).ok_or_else(|| {
                    format!("{} 的第二路 video stream 沒有有效 index", path.display())
                })?,
            ];
            consumed.insert(path.clone());
            captures.push(dual_track_bundle(
                by_path.get(&path).cloned().expect("source exists"),
                indices,
            ));
            continue;
        }

        let Some((pair_key, _)) = insta_pair_key(&path) else {
            return Err(format!(
                "{} 未包含兩路可辨識的雙魚眼 video stream",
                path.display()
            ));
        };
        let (lens0_path, lens1_path) = paired.get(&pair_key).cloned().unwrap_or_default();
        let lens0_path = lens0_path
            .ok_or_else(|| format!("Insta360 素材缺少與 {} 配對的 _00_ INSV 檔", path.display()))?;
        let lens1_path = lens1_path
            .ok_or_else(|| format!("Insta360 素材缺少與 {} 配對的 _10_ INSV 檔", path.display()))?;
        let lens0 = by_path.get(&lens0_path).expect("paired source exists");
        let lens1 = by_path.get(&lens1_path).expect("paired source exists");
        validate_paired_streams(lens0, lens1)?;
        let lens0_index = stream_index(video_streams(&lens0.probe)[0])
            .ok_or_else(|| "Insta360 lens0 stream index 無效".to_owned())?;
        let lens1_index = stream_index(video_streams(&lens1.probe)[0])
            .ok_or_else(|| "Insta360 lens1 stream index 無效".to_owned())?;
        consumed.insert(lens0_path.clone());
        consumed.insert(lens1_path.clone());
        captures.push(CaptureBundle {
            adapter: "Insta360PairedInsvAdapter",
            vendor: "Insta360",
            model: lens0
                .camera_model
                .clone()
                .or_else(|| lens1.camera_model.clone())
                .or_else(|| Some("paired INSV".to_owned())),
            source_paths: vec![lens0_path.clone(), lens1_path.clone()],
            source_probes: vec![lens0.clone(), lens1.clone()],
            lenses: [
                LensStream {
                    lens_id: "lens0",
                    source_path: lens0_path.clone(),
                    ffmpeg_stream_index: lens0_index,
                },
                LensStream {
                    lens_id: "lens1",
                    source_path: lens1_path,
                    ffmpeg_stream_index: lens1_index,
                },
            ],
            telemetry_path: lens0_path,
            probe: lens0.probe.clone(),
            native_fisheye: true,
            factory_intrinsics: false,
            rig_extrinsics: false,
        });
    }
    Ok(captures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn collect_insv(path: &Path, output: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(path) else {
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

    fn probe(streams: usize, width: u64, height: u64, fps: &str, duration: &str) -> Value {
        json!({
            "streams": (0..streams).map(|index| json!({
                "index": index,
                "codec_type": "video",
                "width": width,
                "height": height,
                "avg_frame_rate": fps
            })).collect::<Vec<_>>(),
            "format": {"duration": duration}
        })
    }

    #[test]
    fn supported_sources_include_insv_case_insensitively() {
        assert!(is_supported_source(Path::new("capture.INSV")));
        assert!(!is_supported_source(Path::new("capture.insp")));
    }

    #[test]
    fn expands_existing_insta360_pair() {
        let temp = TempDir::new().unwrap();
        let lens0 = temp.path().join("VID_20240414_135506_00_088.insv");
        let lens1 = temp.path().join("VID_20240414_135506_10_088.insv");
        std::fs::write(&lens0, []).unwrap();
        std::fs::write(&lens1, []).unwrap();
        let mut paths = vec![lens1.clone()];
        expand_related_sources(&mut paths);
        assert_eq!(paths, vec![lens0, lens1]);
    }

    #[test]
    fn resolves_single_file_dual_track_insv() {
        let path = PathBuf::from("X4.insv");
        let captures = resolve_capture_bundles(vec![ProbedSource {
            path: path.clone(),
            probe: probe(2, 3840, 3840, "30000/1001", "10.0"),
            camera_model: Some("Insta360 X4".to_owned()),
            color_profile: None,
        }])
        .unwrap();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].adapter, "Insta360DualTrackAdapter");
        assert_eq!(captures[0].lenses[0].source_path, path);
        assert_eq!(captures[0].lenses[1].ffmpeg_stream_index, 1);
    }

    #[test]
    fn insta360_adapters_expose_only_a_nominal_bootstrap_pose() {
        let expected = RigBootstrapPoseHint {
            cam_from_rig_rotation: [0.0, 0.0, 1.0, 0.0],
            cam_from_rig_translation: [0.0, 0.0, 0.0],
            provenance: "insta360-adapter-nominal-back-to-back-v1",
            refine_sensor_from_rig: false,
        };
        assert_eq!(
            rig_bootstrap_pose_hint("Insta360DualTrackAdapter"),
            Some(expected)
        );
        assert_eq!(
            rig_bootstrap_pose_hint("Insta360PairedInsvAdapter"),
            Some(expected)
        );
    }

    #[test]
    fn dji_osmo_360_exposes_a_rigid_back_to_back_bootstrap_pose() {
        assert_eq!(
            rig_bootstrap_pose_hint("DjiOsmo360Adapter"),
            Some(RigBootstrapPoseHint {
                cam_from_rig_rotation: [0.0, 0.0, 1.0, 0.0],
                cam_from_rig_translation: [0.0, 0.0, 0.0],
                provenance: "dji-osmo-360-adapter-nominal-back-to-back-v1",
                refine_sensor_from_rig: true,
            })
        );
    }

    #[test]
    fn rejects_mismatched_dual_track_lenses() {
        let path = PathBuf::from("mismatched.insv");
        let error = resolve_capture_bundles(vec![ProbedSource {
            path,
            probe: json!({
                "streams": [
                    {"index": 0, "codec_type": "video", "width": 3840, "height": 3840, "avg_frame_rate": "30/1"},
                    {"index": 1, "codec_type": "video", "width": 2880, "height": 2880, "avg_frame_rate": "30/1"}
                ],
                "format": {"duration": "10.0"}
            }),
            camera_model: None,
            color_profile: None,
        }])
        .unwrap_err();
        assert!(error.contains("解析度不同"));
    }

    #[test]
    fn resolves_paired_single_track_insv_in_lens_order() {
        let lens1 = PathBuf::from("VID_20240414_135506_10_088.insv");
        let lens0 = PathBuf::from("VID_20240414_135506_00_088.insv");
        let captures = resolve_capture_bundles(vec![
            ProbedSource {
                path: lens1.clone(),
                probe: probe(1, 2880, 2880, "30000/1001", "85.185"),
                camera_model: Some("Insta360 X3".to_owned()),
                color_profile: None,
            },
            ProbedSource {
                path: lens0.clone(),
                probe: probe(1, 2880, 2880, "30000/1001", "85.218"),
                camera_model: Some("Insta360 X3".to_owned()),
                color_profile: None,
            },
        ])
        .unwrap();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].adapter, "Insta360PairedInsvAdapter");
        assert_eq!(captures[0].lenses[0].source_path, lens0);
        assert_eq!(captures[0].lenses[1].source_path, lens1);
    }

    #[test]
    fn rejects_incomplete_insta360_pair() {
        let path = PathBuf::from("VID_20240414_135506_00_088.insv");
        let error = resolve_capture_bundles(vec![ProbedSource {
            path,
            probe: probe(1, 2880, 2880, "30000/1001", "85.0"),
            camera_model: None,
            color_profile: None,
        }])
        .unwrap_err();
        assert!(error.contains("_10_"));
    }

    #[test]
    #[ignore = "requires GS360_TEST_INSTA_DIR and ffprobe"]
    fn resolves_real_insta360_directory_without_losing_physical_sources() {
        let root = PathBuf::from(
            std::env::var("GS360_TEST_INSTA_DIR").expect("GS360_TEST_INSTA_DIR is required"),
        );
        let mut paths = Vec::new();
        collect_insv(&root, &mut paths);
        paths.sort();
        assert!(!paths.is_empty());
        let sources = paths
            .iter()
            .map(|path| {
                let output = std::process::Command::new("ffprobe")
                    .args([
                        "-v",
                        "error",
                        "-show_streams",
                        "-show_format",
                        "-of",
                        "json",
                    ])
                    .arg(path)
                    .output()
                    .unwrap();
                assert!(output.status.success());
                ProbedSource {
                    path: path.clone(),
                    probe: serde_json::from_slice(&output.stdout).unwrap(),
                    camera_model: None,
                    color_profile: None,
                }
            })
            .collect::<Vec<_>>();
        let captures = resolve_capture_bundles(sources).unwrap();
        assert_eq!(
            captures
                .iter()
                .map(|capture| capture.source_paths.len())
                .sum::<usize>(),
            paths.len()
        );
        assert!(captures
            .iter()
            .any(|capture| capture.adapter == "Insta360PairedInsvAdapter"));
        assert!(captures
            .iter()
            .any(|capture| capture.adapter == "Insta360DualTrackAdapter"));
    }
}
