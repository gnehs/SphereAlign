//! Benchmark reports for the three reconstruction variants.
//!
//! The collector intentionally treats every input as optional.  A benchmark
//! run that has not completed (or that only has COLMAP's binary model files)
//! still produces a useful, explicit partial report instead of replacing
//! unknown measurements with zero.  The text-model parser is public because
//! the same camera/frame poses are useful to the later sensor-to-camera
//! calibration stage.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

pub const RECONSTRUCTION_BENCHMARK_SCHEMA_VERSION: u32 = 1;

/// The benchmark groups requested in the IMU reconstruction plan.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkVariant {
    #[default]
    ACurrent,
    BImuPruning,
    CGlobal,
}

impl BenchmarkVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ACurrent => "a_current",
            Self::BImuPruning => "b_imu_pruning",
            Self::CGlobal => "c_global",
        }
    }
}

/// Optional report configuration.  `model_dir` is useful when a benchmark
/// compares several sparse models under one project root; otherwise the
/// collector searches `sparse/0`, then `sparse`.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkRequest {
    pub variant: BenchmarkVariant,
    pub model_dir: Option<PathBuf>,
    pub timing_path: Option<PathBuf>,
}

impl BenchmarkRequest {
    pub fn new(variant: BenchmarkVariant) -> Self {
        Self {
            variant,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InputAvailability {
    #[serde(default)]
    pub files: BTreeMap<String, bool>,
    #[serde(default)]
    pub missing_inputs: Vec<String>,
    #[serde(default)]
    pub unreadable_inputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CaptureMetrics {
    pub source_count: Option<u64>,
    pub candidate_frame_count: Option<u64>,
    pub selected_rig_frame_count: Option<u64>,
    pub expected_image_count: Option<u64>,
    pub extracted_image_count: Option<u64>,
    pub base_fps: Option<f64>,
    pub candidate_fps: Option<f64>,
    pub dense_fps: Option<f64>,
    pub pruning_enabled: Option<bool>,
    pub frame_motion_record_count: Option<u64>,
    pub telemetry_covered_frame_count: Option<u64>,
    pub telemetry_uncovered_frame_count: Option<u64>,
    pub pair_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AlignCheckpointSummary {
    pub schema_version: Option<u32>,
    pub completed: Option<bool>,
    pub fingerprint_present: bool,
    pub feature_fingerprint_present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TimingMetrics {
    /// Stage durations come from `project.json` and are wall-clock elapsed
    /// durations recorded by the pipeline.
    #[serde(default)]
    pub stage_durations_ms: BTreeMap<String, f64>,
    /// Phase durations are accepted from future timing metadata without
    /// requiring a schema migration for the benchmark report.
    #[serde(default)]
    pub phase_durations_ms: BTreeMap<String, f64>,
    pub align_checkpoint: Option<AlignCheckpointSummary>,
}

/// A free-form checklist is deliberate: visual quality cannot be inferred
/// reliably from reprojection error alone and needs a human or 3DGS review.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QualityChecklist {
    pub same_3dgs_settings: Option<bool>,
    pub distant_detail: Option<String>,
    pub floaters: Option<String>,
    pub seam_alignment: Option<String>,
    pub wall_distortion: Option<String>,
    pub path_jitter: Option<String>,
    pub coverage_holes: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapMetrics {
    pub model_format: Option<String>,
    pub model_directory: Option<String>,
    pub model_converter_required: bool,
    pub camera_count: Option<u64>,
    pub registered_image_count: Option<u64>,
    pub registered_rig_frame_count: Option<u64>,
    pub complete_registered_rig_frame_count: Option<u64>,
    pub points3d_count: Option<u64>,
    pub median_track_length: Option<f64>,
    pub median_reprojection_error_px: Option<f64>,
    pub connected_component_count: Option<u64>,
    pub largest_connected_component_image_count: Option<u64>,
    pub unregistered_frame_names: Option<Vec<String>>,
    pub unregistered_image_names: Option<Vec<String>>,
    pub parser_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReconstructionBenchmarkReport {
    pub schema_version: u32,
    pub variant: BenchmarkVariant,
    pub partial: bool,
    pub inputs: InputAvailability,
    pub capture: CaptureMetrics,
    pub colmap: ColmapMetrics,
    pub timing: TimingMetrics,
    pub quality_checklist: QualityChecklist,
    pub warnings: Vec<String>,
}

impl ReconstructionBenchmarkReport {
    /// Serialize without embedding an absolute project path, which keeps a
    /// shareable benchmark artifact from leaking a user's local directory.
    #[allow(dead_code)]
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Public image record parsed from COLMAP `images.txt`.
///
/// COLMAP documents the qvec as a Hamilton quaternion describing the
/// world-to-camera projection.  The name is kept explicit here because it is
/// easy to accidentally treat it as camera-from-world or rig-from-world in a
/// hand-eye calibration implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapImageRecord {
    pub image_id: u64,
    pub name: String,
    pub camera_id: u64,
    pub qvec_camera_from_world: [f64; 4],
    pub tvec_camera_from_world: [f64; 3],
    pub observed_point_count: u64,
    pub frame_id: Option<u64>,
    /// Images present in `images.txt` are registered by COLMAP.  The field is
    /// explicit so callers can combine these records with an expected frame
    /// list and mark missing images without changing the parser's API.
    pub registered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapTrackObservation {
    pub image_id: u64,
    pub point2d_index: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapPoint3dRecord {
    pub point3d_id: u64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub reprojection_error_px: f64,
    pub track: Vec<ColmapTrackObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapFrameDataRef {
    pub sensor_type: String,
    pub sensor_id: u64,
    pub data_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapFrameRecord {
    pub frame_id: u64,
    pub rig_id: u64,
    pub qvec_rig_from_world: [f64; 4],
    pub tvec_rig_from_world: [f64; 3],
    pub data: Vec<ColmapFrameDataRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapCameraRecord {
    pub camera_id: u64,
    pub model: String,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

/// Parsed text model.  Missing files are represented explicitly so this
/// object can be used for diagnostics even when a model is incomplete.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ColmapTextModel {
    pub cameras: Vec<ColmapCameraRecord>,
    pub images: Vec<ColmapImageRecord>,
    pub points3d: Vec<ColmapPoint3dRecord>,
    pub frames: Vec<ColmapFrameRecord>,
    pub missing_files: Vec<String>,
    pub warnings: Vec<String>,
}

/// Parse all available COLMAP text files in a model directory.
///
/// This partial parser never invents an empty model for an unreadable file;
/// it records the missing file and keeps successfully parsed siblings.  The
/// report collector uses the same function, while calibration code can use
/// the public image/frame records directly.
pub fn parse_colmap_text_model(model_dir: &Path) -> ColmapTextModel {
    let frames = read_frames_file(&model_dir.join("frames.txt"));
    let frame_links = frames
        .iter()
        .flat_map(|frame| {
            frame.data.iter().filter_map(|item| {
                item.sensor_type
                    .eq_ignore_ascii_case("camera")
                    .then_some((item.data_id, frame.frame_id))
            })
        })
        .collect::<HashMap<_, _>>();

    let mut model = ColmapTextModel {
        frames,
        ..ColmapTextModel::default()
    };
    let cameras = read_cameras_file(&model_dir.join("cameras.txt"), &mut model);
    model.cameras = cameras;
    let images = read_images_file(&model_dir.join("images.txt"), &frame_links, &mut model);
    model.images = images;
    let points3d = read_points3d_file(&model_dir.join("points3D.txt"), &mut model);
    model.points3d = points3d;
    model
}

/// Strict convenience wrapper for callers that need all three core text
/// files.  `frames.txt` and `rigs.txt` are optional for older COLMAP models.
pub fn read_colmap_text_model(model_dir: &Path) -> Result<ColmapTextModel, String> {
    let model = parse_colmap_text_model(model_dir);
    let core = ["cameras.txt", "images.txt", "points3D.txt"];
    let missing = core
        .iter()
        .filter(|name| model.missing_files.iter().any(|value| value == **name))
        .copied()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(model)
    } else {
        Err(format!(
            "COLMAP text model is missing required files: {}",
            missing.join(", ")
        ))
    }
}

/// Parse only `images.txt`, preserving camera-from-world qvec and frame links
/// when a caller has already parsed `frames.txt` separately.
#[allow(dead_code)]
pub fn read_colmap_images_txt(path: &Path) -> Result<Vec<ColmapImageRecord>, String> {
    let mut model = ColmapTextModel::default();
    let images = read_images_file(path, &HashMap::new(), &mut model);
    if model.missing_files.is_empty() {
        Ok(images)
    } else {
        Err(model.missing_files.join(", "))
    }
}

/// Parse only `frames.txt` for hand-eye and rig calibration consumers.
#[allow(dead_code)]
pub fn read_colmap_frames_txt(path: &Path) -> Result<Vec<ColmapFrameRecord>, String> {
    let mut warnings = Vec::new();
    let frames = parse_frames_path(path, &mut warnings)?;
    Ok(frames)
}

/// Collect one A/B/C report.  All I/O and parse failures are converted into
/// `partial` fields and warnings; this function intentionally does not fail
/// just because an earlier pipeline stage is incomplete.
pub fn collect_benchmark_report(
    root: &Path,
    request: &BenchmarkRequest,
) -> ReconstructionBenchmarkReport {
    let mut context = CollectorContext::new(request.variant);
    collect_capture(root, &mut context);
    collect_pairs(root, &mut context);
    collect_timing(root, request.timing_path.as_deref(), &mut context);
    collect_colmap(root, request.model_dir.as_deref(), &mut context);

    let mut report = ReconstructionBenchmarkReport {
        schema_version: RECONSTRUCTION_BENCHMARK_SCHEMA_VERSION,
        variant: request.variant,
        partial: true,
        inputs: context.inputs,
        capture: context.capture,
        colmap: context.colmap,
        timing: context.timing,
        quality_checklist: QualityChecklist::default(),
        warnings: context.warnings,
    };
    report.partial = is_partial_report(&report);
    report
}

/// Write a report as JSON.  The parent directory is created when necessary.
pub fn write_benchmark_report(
    root: &Path,
    request: &BenchmarkRequest,
    output: &Path,
) -> Result<ReconstructionBenchmarkReport, String> {
    let report = collect_benchmark_report(root, request);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    fs::write(output, bytes).map_err(|error| error.to_string())?;
    Ok(report)
}

struct CollectorContext {
    variant: BenchmarkVariant,
    inputs: InputAvailability,
    capture: CaptureMetrics,
    timing: TimingMetrics,
    colmap: ColmapMetrics,
    warnings: Vec<String>,
    expected_names_by_lens: [BTreeSet<String>; 2],
    registered_names_by_lens: [BTreeSet<String>; 2],
    registered_names: BTreeSet<String>,
}

impl CollectorContext {
    fn new(variant: BenchmarkVariant) -> Self {
        Self {
            variant,
            inputs: InputAvailability::default(),
            capture: CaptureMetrics::default(),
            timing: TimingMetrics::default(),
            colmap: ColmapMetrics::default(),
            warnings: Vec::new(),
            expected_names_by_lens: [BTreeSet::new(), BTreeSet::new()],
            registered_names_by_lens: [BTreeSet::new(), BTreeSet::new()],
            registered_names: BTreeSet::new(),
        }
    }

    fn mark(&mut self, label: impl Into<String>, present: bool) {
        self.inputs.files.insert(label.into(), present);
    }

    fn missing(&mut self, label: impl Into<String>) {
        let label = label.into();
        if !self.inputs.missing_inputs.contains(&label) {
            self.inputs.missing_inputs.push(label);
        }
    }

    fn unreadable(&mut self, label: impl Into<String>, error: impl Into<String>) {
        let label = label.into();
        if !self.inputs.unreadable_inputs.contains(&label) {
            self.inputs.unreadable_inputs.push(label.clone());
        }
        self.warnings
            .push(format!("unable to read {label}: {}", error.into()));
    }
}

fn is_partial_report(report: &ReconstructionBenchmarkReport) -> bool {
    if !report.inputs.missing_inputs.is_empty() || !report.inputs.unreadable_inputs.is_empty() {
        return true;
    }
    if report.colmap.model_converter_required || report.colmap.model_format.is_none() {
        return true;
    }
    let core_metrics_missing = [
        report.capture.selected_rig_frame_count.is_none(),
        report.capture.expected_image_count.is_none(),
        report.capture.pair_count.is_none(),
        report.colmap.camera_count.is_none(),
        report.colmap.registered_image_count.is_none(),
        report.colmap.points3d_count.is_none(),
    ];
    core_metrics_missing.into_iter().any(|missing| missing)
}

fn collect_capture(root: &Path, context: &mut CollectorContext) {
    let metadata = root.join("metadata");
    let capture_path = metadata.join("capture.json");
    let capture = match read_json(&capture_path) {
        Ok(value) => {
            context.mark("metadata/capture.json", true);
            value
        }
        Err(error) => {
            context.mark("metadata/capture.json", false);
            context.missing("metadata/capture.json");
            if capture_path.exists() {
                context.unreadable("metadata/capture.json", error);
            }
            Value::Null
        }
    };
    if capture.is_object() {
        context.capture.source_count = capture
            .get("sources")
            .and_then(Value::as_array)
            .map(|values| values.len() as u64);
        context.capture.base_fps = number_field(&capture, &["baseFps", "base_fps"]);
        context.capture.candidate_fps = number_field(&capture, &["candidateFps", "candidate_fps"]);
        context.capture.dense_fps = number_field(
            &capture,
            &["requestedDenseFps", "denseFps", "requested_dense_fps"],
        );
        context.capture.pruning_enabled = bool_field(
            &capture,
            &["motionAdaptiveCadence", "motion_adaptive_cadence"],
        );
    }

    let selection_files = discover_metadata_files(&metadata, |name| {
        name == "selection.json" || name.ends_with("_selection.json")
    });
    let motion_files =
        discover_metadata_files(&metadata, |name| name.ends_with("_frame_motion.json"));
    if selection_files.is_empty() {
        context.mark("metadata/selection.json", false);
        context.missing("metadata/*_selection.json");
    } else {
        context.mark("metadata/*_selection.json", true);
    }
    if motion_files.is_empty() {
        context.mark("metadata/*_frame_motion.json", false);
        if !matches!(context.variant, BenchmarkVariant::ACurrent) {
            context.missing("metadata/*_frame_motion.json");
        } else {
            context.warnings.push(
                "frame-motion metadata is absent for baseline A; IMU coverage is unknown".into(),
            );
        }
    } else {
        context.mark("metadata/*_frame_motion.json", true);
    }

    let mut candidate_count = 0u64;
    let mut selected_count = 0u64;
    let mut selected_names_from_metadata = [BTreeSet::new(), BTreeSet::new()];
    let has_selection_files = !selection_files.is_empty();
    for path in &selection_files {
        match read_selection_file(path) {
            Ok(file) => {
                candidate_count = candidate_count.saturating_add(file.candidate_count);
                selected_count = selected_count.saturating_add(file.selected_count);
                for (lens, names) in file.selected_names.iter().enumerate() {
                    selected_names_from_metadata[lens].extend(names.iter().cloned());
                }
            }
            Err(error) => context.unreadable(path_label(root, path), error),
        }
    }
    if has_selection_files {
        if context.capture.source_count.is_none() {
            context.capture.source_count = Some(selection_files.len() as u64);
        }
        context.capture.candidate_frame_count = Some(candidate_count);
        context.capture.selected_rig_frame_count = Some(selected_count);
        let selected_names = selected_names_from_metadata[0]
            .union(&selected_names_from_metadata[1])
            .count() as u64;
        if selected_names > 0 {
            context.capture.selected_rig_frame_count = Some(selected_names);
        }
    }

    let mut motion_count = 0u64;
    let mut covered = 0u64;
    let mut uncovered = 0u64;
    let has_motion_files = !motion_files.is_empty();
    for path in &motion_files {
        match read_motion_file(path) {
            Ok(file) => {
                motion_count = motion_count.saturating_add(file.frame_count);
                covered = covered.saturating_add(file.covered_frame_count);
                uncovered = uncovered.saturating_add(file.uncovered_frame_count);
            }
            Err(error) => context.unreadable(path_label(root, path), error),
        }
    }
    if motion_count > 0 || has_motion_files {
        context.capture.frame_motion_record_count = Some(motion_count);
        context.capture.telemetry_covered_frame_count = Some(covered);
        context.capture.telemetry_uncovered_frame_count = Some(uncovered);
    }

    let mut observed_image_names_by_lens = [BTreeSet::new(), BTreeSet::new()];
    let mut observed_image_dirs = [false, false];
    for (lens, directory) in [root.join("images/lens0"), root.join("images/lens1")]
        .into_iter()
        .enumerate()
    {
        if !directory.is_dir() {
            context.mark(format!("images/lens{lens}"), false);
            context.missing(format!("images/lens{lens}"));
            continue;
        }
        context.mark(format!("images/lens{lens}"), true);
        observed_image_dirs[lens] = true;
        if let Ok(entries) = fs::read_dir(directory) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && is_image_file(&path) {
                    if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
                        observed_image_names_by_lens[lens].insert(name.to_owned());
                        context.expected_names_by_lens[lens].insert(name.to_owned());
                    }
                }
            }
        }
    }
    for (lens, names) in selected_names_from_metadata.iter().enumerate() {
        if context.expected_names_by_lens[lens].is_empty() {
            context.expected_names_by_lens[lens].extend(names.iter().cloned());
        }
    }
    let expected_frame_count = context.expected_names_by_lens[0]
        .union(&context.expected_names_by_lens[1])
        .count() as u64;
    if observed_image_dirs.iter().any(|observed| *observed) {
        context.capture.extracted_image_count = Some(
            observed_image_names_by_lens[0].len() as u64
                + observed_image_names_by_lens[1].len() as u64,
        );
    }
    if expected_frame_count > 0 {
        context.capture.expected_image_count = Some(expected_frame_count.saturating_mul(2));
    } else if let Some(selected) = context.capture.selected_rig_frame_count {
        context.capture.expected_image_count = selected.checked_mul(2);
    }
}

fn collect_pairs(root: &Path, context: &mut CollectorContext) {
    let path = root.join("metadata/pairs.txt");
    let Ok(file) = fs::File::open(&path) else {
        context.mark("metadata/pairs.txt", false);
        context.missing("metadata/pairs.txt");
        return;
    };
    context.mark("metadata/pairs.txt", true);
    let mut count = 0u64;
    let mut invalid = 0u64;
    for line in BufReader::new(file).lines() {
        match line {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if line.split_whitespace().count() >= 2 {
                    count = count.saturating_add(1);
                } else {
                    invalid = invalid.saturating_add(1);
                }
            }
            Err(error) => {
                context.unreadable("metadata/pairs.txt", error.to_string());
                break;
            }
        }
    }
    context.capture.pair_count = Some(count);
    if invalid > 0 {
        context
            .warnings
            .push(format!("ignored {invalid} malformed pairs.txt lines"));
    }
}

fn collect_timing(root: &Path, explicit: Option<&Path>, context: &mut CollectorContext) {
    let project_path = root.join("project.json");
    if let Ok(project) = read_json(&project_path) {
        context.mark("project.json", true);
        if let Some(stages) = project.get("stages").and_then(Value::as_object) {
            for (stage, payload) in stages {
                if let Some(duration) = number_field(payload, &["durationMs", "duration_ms"]) {
                    context
                        .timing
                        .stage_durations_ms
                        .insert(stage.clone(), duration);
                }
            }
        }
        collect_phase_durations(&project, &mut context.timing.phase_durations_ms);
    } else {
        context.mark("project.json", false);
        context
            .warnings
            .push("project.json timing metadata is unavailable".into());
    }

    let checkpoint_path = root.join("metadata/align.checkpoint.json");
    match read_json(&checkpoint_path) {
        Ok(value) => {
            context.mark("metadata/align.checkpoint.json", true);
            context.timing.align_checkpoint = Some(AlignCheckpointSummary {
                schema_version: value
                    .get("schemaVersion")
                    .or_else(|| value.get("schema_version"))
                    .and_then(Value::as_u64)
                    .map(|value| value as u32),
                completed: bool_field(&value, &["completed"]),
                fingerprint_present: value
                    .get("fingerprint")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.trim().is_empty()),
                feature_fingerprint_present: value
                    .get("featureFingerprint")
                    .or_else(|| value.get("feature_fingerprint"))
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.trim().is_empty()),
            });
            collect_phase_durations(&value, &mut context.timing.phase_durations_ms);
        }
        Err(error) => {
            context.mark("metadata/align.checkpoint.json", false);
            context.missing("metadata/align.checkpoint.json");
            if checkpoint_path.exists() {
                context.unreadable("metadata/align.checkpoint.json", error);
            }
        }
    }

    let timing_candidates = explicit
        .map(|path| vec![path.to_owned()])
        .unwrap_or_else(|| {
            discover_metadata_files(&root.join("metadata"), |name| {
                let lower = name.to_ascii_lowercase();
                (lower.contains("timing") || lower.contains("phase"))
                    && lower.ends_with(".json")
                    && lower != "capture.json"
            })
        });
    for path in timing_candidates {
        if !path.is_file() {
            context.warnings.push(format!(
                "timing metadata path does not exist: {}",
                path.display()
            ));
            continue;
        }
        if let Ok(value) = read_json(&path) {
            collect_phase_durations(&value, &mut context.timing.phase_durations_ms);
        } else {
            context.unreadable(path_label(root, &path), "invalid JSON");
        }
    }
}

fn collect_colmap(root: &Path, requested: Option<&Path>, context: &mut CollectorContext) {
    let model_dir = requested
        .map(PathBuf::from)
        .or_else(|| find_model_directory(root));
    let Some(model_dir) = model_dir else {
        context.missing("sparse model (sparse/0 or sparse)");
        context.colmap.parser_warnings.push(
            "COLMAP sparse model directory is unavailable; run align before benchmarking".into(),
        );
        return;
    };
    let text_core = ["cameras.txt", "images.txt", "points3D.txt"]
        .iter()
        .map(|name| model_dir.join(name).is_file())
        .collect::<Vec<_>>();
    let binary_core = ["cameras.bin", "images.bin", "points3D.bin"]
        .iter()
        .map(|name| model_dir.join(name).is_file())
        .collect::<Vec<_>>();
    let format = if text_core.iter().all(|present| *present) {
        "text"
    } else if binary_core.iter().all(|present| *present) {
        "binary"
    } else if text_core.iter().any(|present| *present) {
        "partial_text"
    } else if binary_core.iter().any(|present| *present) {
        "partial_binary"
    } else {
        "missing"
    };
    context.colmap.model_format = (format != "missing").then(|| format.to_owned());
    context.colmap.model_directory = Some(relative_model_label(root, &model_dir));
    context.colmap.model_converter_required = format == "binary"
        || format == "partial_binary"
        || (format == "partial_text" && binary_core.iter().any(|present| *present));
    if context.colmap.model_converter_required {
        context.colmap.parser_warnings.push(
            "COLMAP model is binary-only or mixed; run `colmap model_converter` to export cameras.txt, images.txt, and points3D.txt before collecting full metrics".into(),
        );
    }
    if format == "missing" {
        context.missing("sparse model text or binary files");
        return;
    }
    if format != "text" {
        // Binary parsing is intentionally out of scope; do not report zeroes.
        context.colmap.parser_warnings.extend(
            ["camera_count", "registered_image_count", "points3d_count"]
                .into_iter()
                .map(|metric| format!("{metric} unavailable until model_converter exports text")),
        );
        return;
    }
    let model = parse_colmap_text_model(&model_dir);
    context
        .colmap
        .parser_warnings
        .extend(model.warnings.clone());
    for missing in &model.missing_files {
        context.missing(format!("sparse/{missing}"));
    }
    context.colmap.camera_count = (!model.missing_files.iter().any(|name| name == "cameras.txt"))
        .then_some(model.cameras.len() as u64);
    context.colmap.registered_image_count = Some(model.images.len() as u64);
    context.colmap.points3d_count = Some(model.points3d.len() as u64);
    if !model.points3d.is_empty() {
        context.colmap.median_track_length = median(
            model
                .points3d
                .iter()
                .map(|point| point.track.len() as f64)
                .collect(),
        );
        context.colmap.median_reprojection_error_px = median(
            model
                .points3d
                .iter()
                .map(|point| point.reprojection_error_px)
                .filter(|value| value.is_finite())
                .collect(),
        );
        let (component_count, largest) = connected_components(&model.points3d);
        context.colmap.connected_component_count = Some(component_count);
        context.colmap.largest_connected_component_image_count = largest;
    }
    for image in model.images {
        context.registered_names.insert(image.name.clone());
        let normalized = normalize_image_name(&image.name);
        if image.name.starts_with("lens0/") || image.name.contains("/lens0/") {
            context.registered_names_by_lens[0].insert(normalized.clone());
        } else if image.name.starts_with("lens1/") || image.name.contains("/lens1/") {
            context.registered_names_by_lens[1].insert(normalized.clone());
        } else {
            // Older models may not preserve the lens directory in NAME.  A
            // filename-only entry is still useful for frame-level matching.
            context.registered_names_by_lens[0].insert(normalized.clone());
            context.registered_names_by_lens[1].insert(normalized.clone());
        }
    }
    finalize_unregistered_names(context);
}

fn finalize_unregistered_names(context: &mut CollectorContext) {
    let expected_frames = context.expected_names_by_lens[0]
        .union(&context.expected_names_by_lens[1])
        .cloned()
        .collect::<BTreeSet<_>>();
    let registered_frames = context
        .registered_names_by_lens
        .iter()
        .flat_map(|names| names.iter().cloned())
        .collect::<BTreeSet<_>>();
    if !expected_frames.is_empty() {
        context.colmap.unregistered_frame_names = Some(
            expected_frames
                .difference(&registered_frames)
                .cloned()
                .collect(),
        );
        let mut missing_images = BTreeSet::new();
        for lens in 0..2 {
            for name in context.expected_names_by_lens[lens]
                .difference(&context.registered_names_by_lens[lens])
            {
                missing_images.insert(format!("lens{lens}/{name}"));
            }
        }
        context.colmap.unregistered_image_names = Some(missing_images.into_iter().collect());
        context.colmap.registered_rig_frame_count =
            Some(expected_frames.intersection(&registered_frames).count() as u64);
        let complete = context.expected_names_by_lens[0]
            .intersection(&context.expected_names_by_lens[1])
            .filter(|name| {
                context.registered_names_by_lens[0].contains(*name)
                    && context.registered_names_by_lens[1].contains(*name)
            })
            .count();
        context.colmap.complete_registered_rig_frame_count = Some(complete as u64);
    }
}

fn read_selection_file(path: &Path) -> Result<SelectionSummary, String> {
    let value = read_json(path)?;
    let selections = value
        .get("selections")
        .and_then(Value::as_array)
        .ok_or_else(|| "selection metadata is missing selections[]".to_owned())?;
    let mut summary = SelectionSummary {
        candidate_count: selections.len() as u64,
        ..SelectionSummary::default()
    };
    for selection in selections {
        if selection.get("selected").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        summary.selected_count = summary.selected_count.saturating_add(1);
        for (lens, field) in [(0, "output_lens0"), (1, "output_lens1")] {
            if let Some(path) = selection.get(field).and_then(Value::as_str) {
                if let Some(name) = Path::new(path).file_name().and_then(|value| value.to_str()) {
                    summary.selected_names[lens].insert(name.to_owned());
                }
            }
        }
    }
    Ok(summary)
}

fn read_motion_file(path: &Path) -> Result<MotionSummary, String> {
    let value = read_json(path)?;
    let frames = value
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| "frame-motion metadata is missing frames[]".to_owned())?;
    let mut summary = MotionSummary {
        frame_count: frames.len() as u64,
        ..MotionSummary::default()
    };
    if let Some(coverage) = value.get("telemetryCoverage") {
        summary.covered_frame_count = coverage
            .get("coveredFrameCount")
            .or_else(|| coverage.get("covered_frame_count"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        summary.uncovered_frame_count = coverage
            .get("uncoveredFrameCount")
            .or_else(|| coverage.get("uncovered_frame_count"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
    }
    Ok(summary)
}

#[derive(Default)]
struct SelectionSummary {
    candidate_count: u64,
    selected_count: u64,
    selected_names: [BTreeSet<String>; 2],
}

#[derive(Default)]
struct MotionSummary {
    frame_count: u64,
    covered_frame_count: u64,
    uncovered_frame_count: u64,
}

fn read_cameras_file(path: &Path, model: &mut ColmapTextModel) -> Vec<ColmapCameraRecord> {
    let Ok(lines) = open_data_lines(path, model) else {
        return Vec::new();
    };
    lines
        .filter_map(|(line_number, line)| {
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 5 {
                model.warnings.push(format!(
                    "cameras.txt line {line_number} has fewer than 5 fields"
                ));
                return None;
            }
            Some(ColmapCameraRecord {
                camera_id: parse_u64(tokens[0], "camera id", line_number, &mut model.warnings)?,
                model: tokens[1].to_owned(),
                width: parse_u64(tokens[2], "camera width", line_number, &mut model.warnings)?,
                height: parse_u64(tokens[3], "camera height", line_number, &mut model.warnings)?,
                params: tokens[4..]
                    .iter()
                    .filter_map(|token| token.parse::<f64>().ok())
                    .collect(),
            })
        })
        .collect()
}

fn read_images_file(
    path: &Path,
    frame_links: &HashMap<u64, u64>,
    model: &mut ColmapTextModel,
) -> Vec<ColmapImageRecord> {
    let Ok(lines) = open_data_lines(path, model) else {
        return Vec::new();
    };
    let lines = lines.collect::<Vec<_>>();
    let mut images = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let (line_number, line) = lines[index].clone();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 10 {
            model.warnings.push(format!(
                "images.txt line {line_number} has fewer than 10 fields"
            ));
            index += 1;
            continue;
        }
        let Some(image_id) = parse_u64(tokens[0], "image id", line_number, &mut model.warnings)
        else {
            index += 1;
            continue;
        };
        let Some(qvec) = parse_array4(&tokens[1..5]) else {
            model
                .warnings
                .push(format!("images.txt line {line_number} has invalid qvec"));
            index += 1;
            continue;
        };
        let Some(tvec) = parse_array3(&tokens[5..8]) else {
            model
                .warnings
                .push(format!("images.txt line {line_number} has invalid tvec"));
            index += 1;
            continue;
        };
        let Some(camera_id) = parse_u64(tokens[8], "camera id", line_number, &mut model.warnings)
        else {
            index += 1;
            continue;
        };
        let name = tokens[9..].join(" ");
        let observed_point_count = lines
            .get(index + 1)
            .map(|(_, points)| {
                points
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .chunks(3)
                    .filter(|chunk| {
                        chunk.len() == 3 && chunk[2].parse::<i64>().is_ok_and(|id| id >= 0)
                    })
                    .count() as u64
            })
            .unwrap_or_default();
        images.push(ColmapImageRecord {
            image_id,
            name,
            camera_id,
            qvec_camera_from_world: qvec,
            tvec_camera_from_world: tvec,
            observed_point_count,
            frame_id: frame_links.get(&image_id).copied(),
            registered: true,
        });
        index += 2;
    }
    images
}

fn read_points3d_file(path: &Path, model: &mut ColmapTextModel) -> Vec<ColmapPoint3dRecord> {
    let Ok(lines) = open_data_lines(path, model) else {
        return Vec::new();
    };
    lines
        .filter_map(|(line_number, line)| {
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 8 {
                model.warnings.push(format!(
                    "points3D.txt line {line_number} has fewer than 8 fields"
                ));
                return None;
            }
            let point3d_id = parse_u64(tokens[0], "point3D id", line_number, &mut model.warnings)?;
            let xyz = parse_array3(&tokens[1..4])?;
            let rgb = [
                tokens[4].parse::<u8>().ok()?,
                tokens[5].parse::<u8>().ok()?,
                tokens[6].parse::<u8>().ok()?,
            ];
            let reprojection_error_px = tokens[7].parse::<f64>().ok()?;
            let track = tokens[8..]
                .chunks(2)
                .filter_map(|chunk| {
                    if chunk.len() != 2 {
                        return None;
                    }
                    Some(ColmapTrackObservation {
                        image_id: chunk[0].parse().ok()?,
                        point2d_index: chunk[1].parse().ok()?,
                    })
                })
                .collect();
            Some(ColmapPoint3dRecord {
                point3d_id,
                xyz,
                rgb,
                reprojection_error_px,
                track,
            })
        })
        .collect()
}

fn read_frames_file(path: &Path) -> Vec<ColmapFrameRecord> {
    let mut warnings = Vec::new();
    parse_frames_path(path, &mut warnings).unwrap_or_default()
}

fn parse_frames_path(
    path: &Path,
    warnings: &mut Vec<String>,
) -> Result<Vec<ColmapFrameRecord>, String> {
    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut frames = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| error.to_string())?;
        let line = line.trim().to_owned();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 10 {
            warnings.push(format!(
                "frames.txt line {} has fewer than 10 fields",
                line_number + 1
            ));
            continue;
        }
        let Some(frame_id) = tokens[0].parse::<u64>().ok() else {
            continue;
        };
        let Some(rig_id) = tokens[1].parse::<u64>().ok() else {
            continue;
        };
        let Some(qvec) = parse_array4(&tokens[2..6]) else {
            continue;
        };
        let Some(tvec) = parse_array3(&tokens[6..9]) else {
            continue;
        };
        let Some(num_data_ids) = tokens[9].parse::<usize>().ok() else {
            continue;
        };
        let mut data = Vec::new();
        let mut cursor = 10usize;
        for _ in 0..num_data_ids {
            if cursor + 2 >= tokens.len() {
                break;
            }
            let Some(sensor_id) = tokens[cursor + 1].parse::<u64>().ok() else {
                break;
            };
            let Some(data_id) = tokens[cursor + 2].parse::<u64>().ok() else {
                break;
            };
            data.push(ColmapFrameDataRef {
                sensor_type: tokens[cursor].to_owned(),
                sensor_id,
                data_id,
            });
            cursor += 3;
        }
        frames.push(ColmapFrameRecord {
            frame_id,
            rig_id,
            qvec_rig_from_world: qvec,
            tvec_rig_from_world: tvec,
            data,
        });
    }
    Ok(frames)
}

fn open_data_lines(
    path: &Path,
    model: &mut ColmapTextModel,
) -> io::Result<impl Iterator<Item = (usize, String)>> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) => {
            model.missing_files.push(
                path.file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("unknown")
                    .to_owned(),
            );
            return Err(error);
        }
    };
    Ok(BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.ok()?;
            let line = line.trim().to_owned();
            if line.is_empty() || line.starts_with('#') {
                None
            } else {
                Some((index + 1, line))
            }
        }))
}

fn parse_u64(token: &str, field: &str, line: usize, warnings: &mut Vec<String>) -> Option<u64> {
    match token.parse::<u64>() {
        Ok(value) => Some(value),
        Err(error) => {
            warnings.push(format!("invalid {field} at line {line}: {error}"));
            None
        }
    }
}

fn parse_array3(tokens: &[&str]) -> Option<[f64; 3]> {
    Some([
        tokens.first()?.parse().ok()?,
        tokens.get(1)?.parse().ok()?,
        tokens.get(2)?.parse().ok()?,
    ])
}

fn parse_array4(tokens: &[&str]) -> Option<[f64; 4]> {
    Some([
        tokens.first()?.parse().ok()?,
        tokens.get(1)?.parse().ok()?,
        tokens.get(2)?.parse().ok()?,
        tokens.get(3)?.parse().ok()?,
    ])
}

fn connected_components(points: &[ColmapPoint3dRecord]) -> (u64, Option<u64>) {
    let mut parent = HashMap::<u64, u64>::new();
    for point in points {
        for observation in &point.track {
            parent
                .entry(observation.image_id)
                .or_insert(observation.image_id);
        }
        if let Some(first) = point.track.first() {
            for observation in point.track.iter().skip(1) {
                union(&mut parent, first.image_id, observation.image_id);
            }
        }
    }
    if parent.is_empty() {
        return (0, None);
    }
    let mut sizes = HashMap::<u64, u64>::new();
    for image_id in parent.keys().copied().collect::<Vec<_>>() {
        let root = find(&mut parent, image_id);
        *sizes.entry(root).or_default() += 1;
    }
    let largest = sizes.values().copied().max();
    (sizes.len() as u64, largest)
}

fn find(parent: &mut HashMap<u64, u64>, value: u64) -> u64 {
    let root = parent.get(&value).copied().unwrap_or(value);
    if root == value {
        return value;
    }
    let resolved = find(parent, root);
    parent.insert(value, resolved);
    resolved
}

fn union(parent: &mut HashMap<u64, u64>, left: u64, right: u64) {
    let left_root = find(parent, left);
    let right_root = find(parent, right);
    if left_root != right_root {
        parent.insert(right_root, left_root);
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

fn number_field(value: &Value, names: &[&str]) -> Option<f64> {
    names.iter().find_map(|name| {
        value.get(*name).and_then(|value| match value {
            Value::Number(number) => number.as_f64(),
            Value::String(text) => text.parse::<f64>().ok(),
            _ => None,
        })
    })
}

fn bool_field(value: &Value, names: &[&str]) -> Option<bool> {
    names
        .iter()
        .find_map(|name| value.get(*name).and_then(Value::as_bool))
}

fn collect_phase_durations(value: &Value, target: &mut BTreeMap<String, f64>) {
    for key in ["phaseDurations", "phase_durations", "timings", "phases"] {
        let Some(object) = value.get(key).and_then(Value::as_object) else {
            continue;
        };
        for (phase, duration) in object {
            if let Some(duration) = duration_value(duration) {
                target.insert(phase.clone(), duration);
            }
        }
    }
    if let Some(array) = value.get("phaseDurations").and_then(Value::as_array) {
        for item in array {
            let Some(name) = item.get("phase").and_then(Value::as_str) else {
                continue;
            };
            if let Some(duration) = duration_value(item) {
                target.insert(name.to_owned(), duration);
            }
        }
    }
}

fn duration_value(value: &Value) -> Option<f64> {
    if let Some(number) = number_field(
        value,
        &["durationMs", "duration_ms", "elapsedMs", "elapsed_ms"],
    ) {
        return number.is_finite().then_some(number);
    }
    if let Some(number) = value.as_number().and_then(|number| number.as_f64()) {
        return number.is_finite().then_some(number);
    }
    None
}

fn discover_metadata_files<F>(metadata: &Path, predicate: F) -> Vec<PathBuf>
where
    F: Fn(&str) -> bool,
{
    let Ok(entries) = fs::read_dir(metadata) else {
        return Vec::new();
    };
    let mut files = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            (path.is_file() && predicate(name)).then_some(path)
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn find_model_directory(root: &Path) -> Option<PathBuf> {
    let sparse = root.join("sparse");
    let preferred = sparse.join("0");
    if preferred.is_dir() {
        return Some(preferred);
    }
    if sparse.is_dir() {
        let mut directories = fs::read_dir(&sparse)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        directories.sort();
        if let Some(directory) = directories.into_iter().next() {
            return Some(directory);
        }
        return Some(sparse);
    }
    None
}

fn normalize_image_name(name: &str) -> String {
    name.rsplit_once('/')
        .map_or(name, |(_, value)| value)
        .to_owned()
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tif" | "tiff"
            )
        })
}

fn path_label(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn relative_model_label(root: &Path, path: &Path) -> String {
    path_label(root, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_fixture(root: &Path) {
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::create_dir_all(root.join("images/lens0")).unwrap();
        fs::create_dir_all(root.join("images/lens1")).unwrap();
        fs::create_dir_all(root.join("sparse/0")).unwrap();
        fs::write(
            root.join("metadata/capture.json"),
            r#"{"schemaVersion":5,"sources":["fake.osv"],"baseFps":5,"candidateFps":10,"requestedDenseFps":20,"motionAdaptiveCadence":true}"#,
        )
        .unwrap();
        let selection = r#"{"schema_version":5,"selections":[{"sequence":1,"selected":true,"output_lens0":"images/lens0/source000_00000001.jpg","output_lens1":"images/lens1/source000_00000001.jpg"},{"sequence":2,"selected":false}]}"#;
        fs::write(root.join("metadata/source000_selection.json"), selection).unwrap();
        fs::write(
            root.join("metadata/source000_frame_motion.json"),
            r#"{"schemaVersion":1,"telemetryCoverage":{"coveredFrameCount":1,"uncoveredFrameCount":0},"frames":[{"sequence":1,"selected":true}]}"#,
        )
        .unwrap();
        fs::write(
            root.join("metadata/pairs.txt"),
            "lens0/a.jpg lens1/a.jpg\n# comment\n",
        )
        .unwrap();
        fs::write(
            root.join("metadata/align.checkpoint.json"),
            r#"{"schemaVersion":2,"completed":true,"fingerprint":"abc","featureFingerprint":"def"}"#,
        )
        .unwrap();
        fs::write(
            root.join("project.json"),
            r#"{"stages":{"align":{"durationMs":1234}},"settings":{}}"#,
        )
        .unwrap();
        fs::write(root.join("images/lens0/source000_00000001.jpg"), b"x").unwrap();
        fs::write(root.join("images/lens1/source000_00000001.jpg"), b"x").unwrap();
        fs::write(
            root.join("sparse/0/cameras.txt"),
            "1 PINHOLE 10 10 5 5 5 5\n",
        )
        .unwrap();
        fs::write(
            root.join("sparse/0/frames.txt"),
            "1 1 1 0 0 0 0 0 0 2 CAMERA 1 1 CAMERA 2 2\n",
        )
        .unwrap();
        fs::write(
            root.join("sparse/0/images.txt"),
            "1 1 0 0 0 0 0 0 1 lens0/source000_00000001.jpg\n0 0 1\n2 1 0 0 0 0 0 0 1 lens1/source000_00000001.jpg\n0 0 1\n",
        )
        .unwrap();
        fs::write(
            root.join("sparse/0/points3D.txt"),
            "1 0 0 0 255 0 0 1.0 1 0 2 0\n",
        )
        .unwrap();
    }

    #[test]
    fn report_collects_metrics_and_preserves_timing() {
        let directory = tempdir().unwrap();
        write_fixture(directory.path());
        let report = collect_benchmark_report(
            directory.path(),
            &BenchmarkRequest::new(BenchmarkVariant::BImuPruning),
        );
        assert!(!report.partial, "warnings: {:?}", report.warnings);
        assert_eq!(report.capture.pair_count, Some(1));
        assert_eq!(report.capture.selected_rig_frame_count, Some(1));
        assert_eq!(report.colmap.camera_count, Some(1));
        assert_eq!(report.colmap.registered_image_count, Some(2));
        assert_eq!(report.colmap.complete_registered_rig_frame_count, Some(1));
        assert_eq!(report.colmap.points3d_count, Some(1));
        assert_eq!(report.colmap.median_track_length, Some(2.0));
        assert_eq!(report.colmap.median_reprojection_error_px, Some(1.0));
        assert_eq!(
            report.colmap.largest_connected_component_image_count,
            Some(2)
        );
        assert_eq!(report.timing.stage_durations_ms.get("align"), Some(&1234.0));
        assert_eq!(
            report.timing.align_checkpoint.as_ref().unwrap().completed,
            Some(true)
        );
    }

    #[test]
    fn parser_exposes_image_pose_and_frame_id() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("frames.txt"),
            "7 3 0.9 0.1 0.2 0.3 1 2 3 1 CAMERA 4 9\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("images.txt"),
            "9 0.9 0.1 0.2 0.3 4 5 6 4 lens0/frame.jpg\n0 0 -1\n",
        )
        .unwrap();
        let model = parse_colmap_text_model(directory.path());
        let image = &model.images[0];
        assert_eq!(image.image_id, 9);
        assert_eq!(image.frame_id, Some(7));
        assert_eq!(image.qvec_camera_from_world, [0.9, 0.1, 0.2, 0.3]);
        assert!(image.registered);
        assert_eq!(model.frames[0].qvec_rig_from_world, [0.9, 0.1, 0.2, 0.3]);
    }

    #[test]
    fn binary_only_model_is_partial_and_requests_converter() {
        let directory = tempdir().unwrap();
        fs::create_dir_all(directory.path().join("sparse/0")).unwrap();
        for name in ["cameras.bin", "images.bin", "points3D.bin"] {
            fs::write(directory.path().join("sparse/0").join(name), b"binary").unwrap();
        }
        let report = collect_benchmark_report(
            directory.path(),
            &BenchmarkRequest::new(BenchmarkVariant::CGlobal),
        );
        assert!(report.partial);
        assert_eq!(report.colmap.model_format.as_deref(), Some("binary"));
        assert!(report.colmap.model_converter_required);
        assert!(report
            .colmap
            .parser_warnings
            .iter()
            .any(|warning| warning.contains("model_converter")));
        assert!(report.colmap.points3d_count.is_none());
    }

    #[test]
    fn missing_inputs_do_not_become_zero_metrics() {
        let directory = tempdir().unwrap();
        let report = collect_benchmark_report(
            directory.path(),
            &BenchmarkRequest::new(BenchmarkVariant::ACurrent),
        );
        assert!(report.partial);
        assert!(report.capture.pair_count.is_none());
        assert!(report.colmap.registered_image_count.is_none());
        assert!(!report.inputs.missing_inputs.is_empty());
    }
}
