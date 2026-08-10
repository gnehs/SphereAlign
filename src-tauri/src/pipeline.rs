//! Resumable stage orchestration around system FFmpeg, the native mask engine,
//! and COLMAP. External commands are always passed as argument arrays (never a
//! shell string), so paths containing spaces cannot inject commands.

use crate::colmap_feature_cache::{
    clear_matching_cache, database_has_nontrivial_rig, inspect_feature_cache,
};
use crate::doctor::find_executable;
use crate::extraction::{
    self, ExtractionRequest, ExtractionStage, SelectionMetadata, SelectionRecord,
};
use crate::fisheye::DJI_VALID_RADIUS_RATIO;
use crate::masking::{self, CancelToken, MaskRequest};
use crate::project::{self, ProjectManifest, StageName, StageStatus};
use crate::telemetry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

use crate::process::silent_command;

static JOB_COUNTER: AtomicU64 = AtomicU64::new(1);
static FULL_RES_TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

const CANDIDATE_PROXY_SIZE: usize = extraction::SHARPNESS_MAX_DIMENSION as usize;
const CANDIDATE_STREAM_WIDTH: usize = CANDIDATE_PROXY_SIZE * 2;
const CANDIDATE_STREAM_HEIGHT: usize = CANDIDATE_PROXY_SIZE;
const CANDIDATE_FRAME_BYTES: usize = CANDIDATE_STREAM_WIDTH * CANDIDATE_STREAM_HEIGHT;
const CANDIDATE_IMAGE_FORMAT: &str = "rawvideo-gray8-hstack-1024x512-memory";
const CANDIDATE_SELECTION_CHECKPOINT_SCHEMA_VERSION: u32 = 2;
const ALIGN_CHECKPOINT_SCHEMA_VERSION: u32 = 2;
// Bump this when matching or mapping semantics change while the underlying
// feature database remains reusable. Keeping it separate from the checkpoint
// schema lets an upgrade invalidate sparse/match output without redoing SIFT.
const ALIGN_PIPELINE_REVISION: u32 = 3;
const FEATURE_FINGERPRINT_SCHEMA_VERSION: u32 = 1;
const FEATURE_EXTRACTION_TYPE: &str = "SIFT";
const FEATURE_CAMERA_MODEL: &str = "OPENCV_FISHEYE";
const FEATURE_DEFAULT_FOCAL_LENGTH_FACTOR: f64 = 0.3;
const CANDIDATE_PROGRESS_INTERVAL: Duration = Duration::from_millis(500);
const COLMAP_PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const CANDIDATE_SELECTION_PROGRESS_SHARE: f32 = 0.7;
const FULL_RESOLUTION_PROGRESS_SHARE: f32 = 0.2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct AlignCheckpoint {
    schema_version: u32,
    fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    feature_fingerprint: Option<String>,
    completed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct AlignFileIdentity {
    path: String,
    size: u64,
    modified_nanos: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AlignFingerprintPayload {
    schema_version: u32,
    pipeline_revision: u32,
    settings: String,
    colmap_version: String,
    include_masks: bool,
    rig_config_sha256: String,
    pairs_sha256: String,
    frame_motion_sha256: String,
    global_mapper_priors_sha256: String,
    files: Vec<AlignFileIdentity>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FeatureFingerprintPayload {
    schema_version: u32,
    colmap_version: String,
    extractor_type: &'static str,
    camera_model: &'static str,
    default_focal_length_factor: f64,
    include_masks: bool,
    files: Vec<AlignFileIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateSelectionCheckpoint {
    schema_version: u32,
    source_path: String,
    source_size: u64,
    source_modified_nanos: Option<String>,
    base_fps: f64,
    candidate_fps: f64,
    dense_fps: f64,
    skip_blurry: bool,
    keyframe_pruning: bool,
    keyframe_thresholds: extraction::KeyframePruningConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    telemetry_sha256: Option<String>,
    image_format: String,
    selection: SelectionMetadata,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartStageRequest {
    pub project_path: String,
    pub stage: StageName,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub settings: Option<Value>,
    #[serde(default)]
    pub colmap_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartStageResponse {
    pub job_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgressEvent {
    job_id: String,
    stage: StageName,
    phase: String,
    progress: f32,
    message: String,
    completed: Option<u64>,
    total: Option<u64>,
    current_item: Option<String>,
    timestamp_ms: u64,
    elapsed_ms: Option<u64>,
    done: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LogEvent {
    job_id: String,
    level: String,
    message: String,
    timestamp_ms: u64,
}

#[derive(Clone)]
struct JobControl {
    cancelled: Arc<AtomicBool>,
    mask_cancel: CancelToken,
}

#[derive(Clone, Default)]
pub struct JobManager {
    jobs: Arc<Mutex<HashMap<String, JobControl>>>,
}

impl JobManager {
    pub fn cancel(&self, id: &str) -> bool {
        let Ok(jobs) = self.jobs.lock() else {
            return false;
        };
        let Some(job) = jobs.get(id) else {
            return false;
        };
        job.cancelled.store(true, Ordering::Release);
        job.mask_cancel.cancel();
        true
    }

    fn insert(&self, id: String, control: JobControl) -> Result<(), String> {
        let mut jobs = self.jobs.lock().map_err(|_| "job manager is unavailable")?;
        if !jobs.is_empty() {
            return Err(
                if jobs
                    .values()
                    .any(|job| job.cancelled.load(Ordering::Acquire))
                {
                    "The previous pipeline stage is still stopping".to_string()
                } else {
                    "Another pipeline stage is already running".to_string()
                },
            );
        }
        jobs.insert(id, control);
        Ok(())
    }

    fn remove(&self, id: &str) {
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.remove(id);
        }
    }
}

fn job_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or_default();
    format!(
        "job-{millis}-{}",
        JOB_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn now_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or_default()
}

fn elapsed_ms(started_at: Instant) -> u64 {
    started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn extraction_completed_count(
    completed: &AtomicU64,
    stage: ExtractionStage,
    interval: usize,
    total_intervals: usize,
) -> u64 {
    let total = total_intervals as u64;
    if matches!(stage, ExtractionStage::Completed | ExtractionStage::Skipped) {
        completed.fetch_max(interval.min(total_intervals) as u64, Ordering::AcqRel);
    }
    completed.load(Ordering::Acquire).min(total)
}

fn numeric_value(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse::<f64>().ok()))
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn probe_duration_seconds(probe: &Value) -> Option<f64> {
    let streams = probe.get("streams").and_then(Value::as_array);
    let shortest_video = streams
        .into_iter()
        .flatten()
        .filter(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"))
        .filter_map(|stream| stream.get("duration").and_then(numeric_value))
        .reduce(f64::min);
    shortest_video
        .or_else(|| probe.pointer("/format/duration").and_then(numeric_value))
        .or_else(|| {
            streams
                .into_iter()
                .flatten()
                .filter_map(|stream| stream.get("duration").and_then(numeric_value))
                .reduce(f64::min)
        })
}

fn expected_candidate_frames(probe: &Value, candidate_fps: f64) -> Option<u64> {
    let frames = (probe_duration_seconds(probe)? * candidate_fps).ceil();
    (frames.is_finite() && frames > 0.0 && frames <= u64::MAX as f64).then_some(frames as u64)
}

fn source_stage_progress(source_index: usize, total_sources: usize, source_fraction: f32) -> f32 {
    let total_sources = total_sources.max(1) as f32;
    (source_index as f32 + source_fraction.clamp(0.0, 1.0)) / total_sources
}

fn emit_progress(
    app: &AppHandle,
    id: &str,
    stage: &StageName,
    phase: &str,
    progress: f32,
    message: impl Into<String>,
    status: &str,
    done: bool,
) {
    emit_progress_detailed(
        app, id, stage, phase, progress, message, status, done, None, None, None, None,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_progress_detailed(
    app: &AppHandle,
    id: &str,
    stage: &StageName,
    phase: &str,
    progress: f32,
    message: impl Into<String>,
    status: &str,
    done: bool,
    completed: Option<u64>,
    total: Option<u64>,
    current_item: Option<String>,
    elapsed_ms: Option<u64>,
) {
    let _ = app.emit(
        "pipeline-progress",
        ProgressEvent {
            job_id: id.to_string(),
            stage: stage.clone(),
            phase: phase.to_string(),
            progress: progress.clamp(0.0, 1.0),
            message: message.into(),
            completed,
            total,
            current_item,
            timestamp_ms: now_timestamp_ms(),
            elapsed_ms,
            done,
            status: status.to_string(),
        },
    );
}

fn emit_log(app: &AppHandle, id: &str, level: &str, message: impl Into<String>) {
    let _ = app.emit(
        "pipeline-log",
        LogEvent {
            job_id: id.to_string(),
            level: level.to_string(),
            message: message.into(),
            timestamp_ms: now_timestamp_ms(),
        },
    );
}

struct CandidateProgressReporter<'a> {
    app: &'a AppHandle,
    id: &'a str,
    source_index: usize,
    total_sources: usize,
    expected_frames: Option<u64>,
    current_item: String,
    highest_processed: u64,
    last_emitted_at: Option<Instant>,
}

impl CandidateProgressReporter<'_> {
    fn report(&mut self, processed: u64) {
        self.highest_processed = self.highest_processed.max(processed);
        if self
            .last_emitted_at
            .is_some_and(|last| last.elapsed() < CANDIDATE_PROGRESS_INTERVAL)
        {
            return;
        }
        self.last_emitted_at = Some(Instant::now());
        let selection_fraction = self
            .expected_frames
            .filter(|total| *total > 0)
            .map(|total| (self.highest_processed as f32 / total as f32).clamp(0.0, 0.99))
            .unwrap_or(0.0);
        let progress = source_stage_progress(
            self.source_index,
            self.total_sources,
            selection_fraction * CANDIDATE_SELECTION_PROGRESS_SHARE,
        );
        emit_progress_detailed(
            self.app,
            self.id,
            &StageName::Extract,
            "selecting-in-memory",
            progress,
            format!(
                "正在同步解碼並評分來源 {}（已處理 {} 組候選影格）",
                self.source_index + 1,
                self.highest_processed
            ),
            "running",
            false,
            Some(
                self.expected_frames
                    .map_or(self.highest_processed, |total| {
                        self.highest_processed.min(total)
                    }),
            ),
            self.expected_frames,
            Some(self.current_item.clone()),
            None,
        );
    }
}

pub fn start_stage(
    app: AppHandle,
    manager: &JobManager,
    request: StartStageRequest,
) -> Result<StartStageResponse, String> {
    let mut manifest = project::load(&request.project_path)?;
    if let Some(settings) = request.settings.clone() {
        merge_json(&mut manifest.settings, settings);
        project::save_manifest(&manifest)?;
    }
    let id = job_id();
    let control = JobControl {
        cancelled: Arc::new(AtomicBool::new(false)),
        mask_cancel: CancelToken::new(),
    };
    manager.insert(id.clone(), control.clone())?;
    let manager = manager.clone();
    let stage = request.stage.clone();
    let force_retry = request.mode.as_deref() == Some("retry");
    let colmap_path = request.colmap_path.clone();
    let response = StartStageResponse { job_id: id.clone() };
    thread::spawn(move || {
        let stage_started_at = Instant::now();
        let skipped_mask = stage == StageName::Mask && !mask_enabled(&manifest.settings);
        let starting_message = if skipped_mask {
            "未啟用遮罩，正在略過"
        } else {
            "處理階段已開始"
        };
        let _ = project::update_stage_timed(
            &mut manifest,
            &stage,
            StageStatus::Running,
            0.0,
            starting_message,
            Vec::new(),
            Vec::new(),
            Some(stage_started_at),
        );
        emit_progress(
            &app,
            &id,
            &stage,
            "starting",
            0.0,
            starting_message,
            "running",
            false,
        );
        let result = match stage {
            StageName::Extract => run_extract(&app, &id, &manifest, &control),
            StageName::Mask => run_mask(&app, &id, &manifest, &control),
            StageName::Align => run_align(
                &app,
                &id,
                &manifest,
                colmap_path.as_deref(),
                force_retry,
                &control,
            ),
        };
        let cancelled = control.cancelled.load(Ordering::Acquire);
        if cancelled {
            let stage_elapsed_ms = elapsed_ms(stage_started_at);
            let _ = project::update_stage_timed(
                &mut manifest,
                &stage,
                StageStatus::Cancelled,
                0.0,
                "處理階段已取消，已寫入的結果可繼續使用",
                Vec::new(),
                Vec::new(),
                Some(stage_started_at),
            );
            emit_progress_detailed(
                &app,
                &id,
                &stage,
                "cancelled",
                0.0,
                "工作已取消，可稍後續作",
                "cancelled",
                true,
                None,
                None,
                None,
                Some(stage_elapsed_ms),
            );
        } else {
            match result {
                Ok(artifacts) => {
                    let stage_elapsed_ms = elapsed_ms(stage_started_at);
                    let completed_message = if skipped_mask {
                        "未啟用 YOLO 或天空過濾，已略過遮罩階段"
                    } else {
                        "處理階段已完成"
                    };
                    let _ = project::update_stage_timed(
                        &mut manifest,
                        &stage,
                        StageStatus::Completed,
                        1.0,
                        completed_message,
                        artifacts,
                        Vec::new(),
                        Some(stage_started_at),
                    );
                    emit_progress_detailed(
                        &app,
                        &id,
                        &stage,
                        if skipped_mask { "skipped" } else { "completed" },
                        1.0,
                        completed_message,
                        "completed",
                        true,
                        None,
                        None,
                        None,
                        Some(stage_elapsed_ms),
                    );
                }
                Err(error) => {
                    emit_log(&app, &id, "error", &error);
                    let stage_elapsed_ms = elapsed_ms(stage_started_at);
                    let _ = project::update_stage_timed(
                        &mut manifest,
                        &stage,
                        StageStatus::Failed,
                        0.0,
                        error.clone(),
                        Vec::new(),
                        vec![error.clone()],
                        Some(stage_started_at),
                    );
                    emit_progress_detailed(
                        &app,
                        &id,
                        &stage,
                        "failed",
                        0.0,
                        error,
                        "failed",
                        true,
                        None,
                        None,
                        None,
                        Some(stage_elapsed_ms),
                    );
                }
            }
        }
        manager.remove(&id);
    });
    Ok(response)
}

fn merge_json(target: &mut Value, incoming: Value) {
    match (target, incoming) {
        (Value::Object(target), Value::Object(incoming)) => {
            for (key, value) in incoming {
                merge_json(target.entry(key).or_insert(Value::Null), value);
            }
        }
        (target, value) => *target = value,
    }
}

fn setting_f64(settings: &Value, key: &str, default: f64) -> f64 {
    settings
        .pointer(key)
        .and_then(Value::as_f64)
        .unwrap_or(default)
}
fn setting_bool(settings: &Value, key: &str, default: bool) -> bool {
    settings
        .pointer(key)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapperMode {
    Auto,
    Incremental,
    Global,
}

fn mapper_mode(settings: &Value) -> Result<MapperMode, String> {
    match settings
        .pointer("/align/mapperMode")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("auto")
        .to_ascii_lowercase()
        .as_str()
    {
        "auto" => Ok(MapperMode::Auto),
        "incremental" => Ok(MapperMode::Incremental),
        "global" => Ok(MapperMode::Global),
        value => Err(format!(
            "align.mapperMode 必須是 auto、incremental 或 global（收到 {value}）"
        )),
    }
}

fn extract_frame_settings(settings: &Value) -> (f64, f64, bool) {
    let base_fps = setting_f64(settings, "/extract/baseFps", 3.0).clamp(0.1, 30.0);
    let dense_fps =
        setting_f64(settings, "/extract/denseFps", 12.0).clamp(base_fps * 2.0, base_fps * 10.0);
    let skip_blurry = setting_bool(settings, "/extract/skipBlurry", true);
    (base_fps, dense_fps, skip_blurry)
}

fn keyframe_pruning_settings(settings: &Value) -> (bool, extraction::KeyframePruningConfig) {
    let defaults = extraction::KeyframePruningConfig::default();
    let min_gap_ms =
        setting_f64(settings, "/extract/minGapMs", defaults.min_gap_ms).clamp(0.0, 2_000.0);
    let max_gap_ms =
        setting_f64(settings, "/extract/maxGapMs", defaults.max_gap_ms).clamp(min_gap_ms, 5_000.0);
    (
        setting_bool(settings, "/extract/keyframePruning", true),
        extraction::KeyframePruningConfig {
            min_rotation_deg: setting_f64(
                settings,
                "/extract/minRotationDeg",
                defaults.min_rotation_deg,
            )
            .clamp(0.1, 90.0),
            min_gap_ms,
            max_gap_ms,
            min_visual_novelty: setting_f64(
                settings,
                "/extract/minVisualNovelty",
                defaults.min_visual_novelty,
            )
            .clamp(0.0, 1.0),
        },
    )
}

const INVALID_GPU_INDEX_MESSAGE: &str = "align.gpuIndex 必須是 -1 或逗號分隔的非負整數（例如 0,1）";

fn parse_gpu_index(settings: &Value) -> Result<String, String> {
    let Some(value) = settings.pointer("/align/gpuIndex") else {
        return Ok("-1".to_owned());
    };
    let Some(value) = value.as_str() else {
        return Err(INVALID_GPU_INDEX_MESSAGE.to_owned());
    };
    let value = value.trim();
    if value == "-1" {
        return Ok("-1".to_owned());
    }
    if value.is_empty() {
        return Err(INVALID_GPU_INDEX_MESSAGE.to_owned());
    }
    let mut indexes = Vec::new();
    for index in value.split(',') {
        let index = index.trim();
        if index.is_empty() || !index.chars().all(|character| character.is_ascii_digit()) {
            return Err(INVALID_GPU_INDEX_MESSAGE.to_owned());
        }
        index
            .parse::<u64>()
            .map_err(|_| INVALID_GPU_INDEX_MESSAGE.to_owned())?;
        indexes.push(index.to_owned());
    }
    Ok(indexes.join(","))
}

fn mask_confidence(settings: &Value) -> f64 {
    let configured = setting_f64(settings, "/mask/confidence", 0.25);
    let version = setting_f64(settings, "/mask/confidenceVersion", 1.0);
    // Version 1 shipped 72% as the default and could not be lowered below 40%,
    // which misses heavily distorted or distant people. Migrate only that exact
    // legacy default; every explicit non-default value remains untouched.
    if version < 2.0 && (configured - 72.0).abs() < f64::EPSILON {
        25.0
    } else {
        configured
    }
}

fn mask_classes(settings: &Value) -> Vec<String> {
    let classes = settings
        .pointer("/mask/classes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let yolo_enabled = settings
        .pointer("/mask/yoloEnabled")
        .and_then(Value::as_bool)
        .unwrap_or(!classes.is_empty());
    if yolo_enabled {
        classes
    } else {
        Vec::new()
    }
}

fn mask_enabled(settings: &Value) -> bool {
    !mask_classes(settings).is_empty() || setting_bool(settings, "/mask/maskSky", false)
}

fn run_child(
    app: &AppHandle,
    id: &str,
    program: &Path,
    args: &[String],
    control: &JobControl,
) -> Result<(), String> {
    if control.cancelled.load(Ordering::Acquire) {
        return Err("cancelled".to_string());
    }
    emit_log(
        app,
        id,
        "info",
        format!(
            "執行 {}",
            program.file_name().unwrap_or_default().to_string_lossy()
        ),
    );
    let mut child = silent_command(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("啟動 {} 失敗: {error}", program.display()))?;
    let mut stdout = child.stdout.take().ok_or("無法讀取子程序輸出")?;
    let mut stderr = child.stderr.take().ok_or("無法讀取子程序錯誤輸出")?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.read_to_end(&mut bytes);
        bytes
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.read_to_end(&mut bytes);
        bytes
    });
    let status = loop {
        if control.cancelled.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err("cancelled".to_string());
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(150)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(format!("讀取子程序狀態失敗: {error}"));
            }
        }
    };
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr);
        let detail = stderr
            .lines()
            .rev()
            .take(12)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(if detail.trim().is_empty() {
            format!("{} 結束，狀態碼 {:?}", program.display(), status.code())
        } else {
            detail
        });
    }
    if !stdout.is_empty() {
        let summary = String::from_utf8_lossy(&stdout);
        if let Some(last) = summary.lines().rev().find(|line| !line.trim().is_empty()) {
            emit_log(app, id, "info", last.trim());
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ChildOutputStream {
    Stdout,
    Stderr,
}

fn read_child_output<R: Read + Send + 'static>(
    reader: R,
    stream: ChildOutputStream,
    sender: std::sync::mpsc::Sender<(ChildOutputStream, String)>,
) {
    let mut reader = BufReader::new(reader);
    loop {
        let mut bytes = Vec::new();
        match reader.read_until(b'\n', &mut bytes) {
            Ok(0) => break,
            Ok(_) => {
                let line = String::from_utf8_lossy(&bytes)
                    .trim_end_matches(['\r', '\n'])
                    .to_owned();
                if sender.send((stream, line)).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

fn run_child_with_output<F>(
    app: &AppHandle,
    id: &str,
    program: &Path,
    args: &[String],
    control: &JobControl,
    mut on_line: F,
) -> Result<(), String>
where
    F: FnMut(&str),
{
    if control.cancelled.load(Ordering::Acquire) {
        return Err("cancelled".to_string());
    }
    emit_log(
        app,
        id,
        "info",
        format!(
            "執行 {}",
            program.file_name().unwrap_or_default().to_string_lossy()
        ),
    );
    let mut child = silent_command(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("啟動 {} 失敗: {error}", program.display()))?;
    let stdout = child.stdout.take().ok_or("無法讀取子程序輸出")?;
    let stderr = child.stderr.take().ok_or("無法讀取子程序錯誤輸出")?;
    let (output_sender, output_receiver) = channel();
    let stdout_sender = output_sender.clone();
    let stdout_reader =
        thread::spawn(move || read_child_output(stdout, ChildOutputStream::Stdout, stdout_sender));
    let stderr_reader =
        thread::spawn(move || read_child_output(stderr, ChildOutputStream::Stderr, output_sender));
    let mut stderr_tail = VecDeque::with_capacity(12);
    let mut last_stdout = None;
    let mut handle_output = |(stream, line): (ChildOutputStream, String)| {
        on_line(&line);
        match stream {
            ChildOutputStream::Stdout => {
                if !line.trim().is_empty() {
                    last_stdout = Some(line);
                }
            }
            ChildOutputStream::Stderr => {
                if !line.trim().is_empty() {
                    if stderr_tail.len() == 12 {
                        stderr_tail.pop_front();
                    }
                    stderr_tail.push_back(line);
                }
            }
        }
    };
    let status = loop {
        while let Ok(output) = output_receiver.try_recv() {
            handle_output(output);
        }
        if control.cancelled.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            while let Ok(output) = output_receiver.try_recv() {
                handle_output(output);
            }
            return Err("cancelled".to_string());
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(150)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(format!("讀取子程序狀態失敗: {error}"));
            }
        }
    };
    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
    while let Ok(output) = output_receiver.try_recv() {
        handle_output(output);
    }
    if !status.success() {
        let detail = stderr_tail.into_iter().collect::<Vec<_>>().join("\n");
        return Err(if detail.trim().is_empty() {
            format!("{} 結束，狀態碼 {:?}", program.display(), status.code())
        } else {
            detail
        });
    }
    if let Some(last) = last_stdout {
        emit_log(app, id, "info", last.trim());
    }
    Ok(())
}

fn cancelled_error(error: &str, control: &JobControl) -> bool {
    error == "cancelled" || control.cancelled.load(Ordering::Acquire)
}

fn run_colmap_with_gpu_fallback<F, R>(
    app: &AppHandle,
    id: &str,
    program: &Path,
    gpu_args: &[String],
    cpu_args: &[String],
    use_gpu: bool,
    component: &str,
    control: &JobControl,
    mut reset_before_cpu_retry: R,
    mut on_line: F,
) -> Result<(), String>
where
    F: FnMut(&str),
    R: FnMut() -> Result<(), String>,
{
    if !use_gpu {
        return run_child_with_output(app, id, program, cpu_args, control, &mut on_line);
    }
    match run_child_with_output(app, id, program, gpu_args, control, &mut on_line) {
        Ok(()) => Ok(()),
        Err(error) if cancelled_error(&error, control) => Err(error),
        Err(error) => {
            emit_log(
                app,
                id,
                "warning",
                format!("COLMAP {component} GPU 執行失敗，改用 CPU 重試：{error}"),
            );
            reset_before_cpu_retry()?;
            run_child_with_output(app, id, program, cpu_args, control, &mut on_line)
        }
    }
}

fn run_mapper_with_gpu_fallback<F, R>(
    app: &AppHandle,
    id: &str,
    program: &Path,
    output_path: &Path,
    gpu_args: &[String],
    cpu_args: &[String],
    use_gpu: bool,
    component: &str,
    control: &JobControl,
    mut reset_progress_before_cpu_retry: R,
    mut on_line: F,
) -> Result<(), String>
where
    F: FnMut(&str),
    R: FnMut(),
{
    if !use_gpu {
        return run_child_with_output(app, id, program, cpu_args, control, &mut on_line);
    }
    match run_child_with_output(app, id, program, gpu_args, control, &mut on_line) {
        Ok(()) => Ok(()),
        Err(error) if cancelled_error(&error, control) => Err(error),
        Err(error) => {
            emit_log(
                app,
                id,
                "warning",
                format!("COLMAP {component} GPU 執行失敗，移除不完整輸出並改用 CPU 重試：{error}"),
            );
            if output_path.exists() {
                fs::remove_dir_all(output_path).map_err(|cleanup_error| {
                    format!("無法清理 COLMAP {component} GPU 不完整輸出：{cleanup_error}")
                })?;
            }
            fs::create_dir_all(output_path).map_err(|create_error| {
                format!("無法建立 COLMAP {component} CPU 輸出資料夾：{create_error}")
            })?;
            reset_progress_before_cpu_retry();
            run_child_with_output(app, id, program, cpu_args, control, &mut on_line)
        }
    }
}

fn is_mapper_gpu_cpu_fallback_line(line: &str) -> bool {
    let normalized = line.to_ascii_lowercase();
    normalized.contains("compiled without cuda support")
        || normalized.contains("compiled without cudss support")
        || normalized.contains("falling back to cpu-based")
}

fn maybe_log_mapper_gpu_cpu_fallback(
    app: &AppHandle,
    id: &str,
    component: &str,
    use_gpu: bool,
    warning_emitted: &mut bool,
    line: &str,
) {
    if !use_gpu || *warning_emitted {
        return;
    }
    if is_mapper_gpu_cpu_fallback_line(line) {
        *warning_emitted = true;
        emit_log(
            app,
            id,
            "warning",
            format!(
                "COLMAP {component} 的 Ceres GPU 不可用，已由 Ceres 改用 CPU：{}",
                line.trim()
            ),
        );
    }
}

/// Build the software-decoder FFmpeg command used to stream both candidate
/// fisheye lenses as one fixed-size grayscale frame. No candidate image is
/// encoded or written to disk: stdout contains consecutive 1024x512 gray8
/// frames, with lens0 on the left and lens1 on the right. `showinfo` emits the
/// post-`fps` presentation timestamp for every candidate on stderr so the
/// selector can align it with telemetry without a second decode.
fn candidate_ffmpeg_args(
    input: &Path,
    stream0: usize,
    stream1: usize,
    candidate_fps: f64,
) -> Vec<String> {
    let lens_filter = |stream: usize, label: &str| {
        format!(
            "[0:{stream}]fps={candidate_fps},scale=w='min(512,iw)':h='min(512,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,pad=512:512:(ow-iw)/2:(oh-ih)/2,format=gray[{label}]"
        )
    };
    let filter = format!(
        "{};{};[lens0][lens1]hstack=inputs=2:shortest=1,showinfo=checksum=0[out]",
        lens_filter(stream0, "lens0"),
        lens_filter(stream1, "lens1")
    );
    vec![
        "-hide_banner".into(),
        "-nostdin".into(),
        "-i".into(),
        input.to_string_lossy().into_owned(),
        "-filter_complex".into(),
        filter,
        "-map".into(),
        "[out]".into(),
        "-an".into(),
        "-sn".into(),
        "-dn".into(),
        "-fps_mode".into(),
        "passthrough".into(),
        "-c:v".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "gray".into(),
        "-f".into(),
        "rawvideo".into(),
        "pipe:1".into(),
    ]
}

/// Build a balanced addition tree of `eq(n, index)` predicates.  FFmpeg's
/// expression parser evaluates the tree recursively; keeping it balanced
/// avoids exhausting parser recursion when a dense capture produces many
/// selected frame indexes.  The expression is passed as a direct argument,
/// never through a shell, and all leaves are generated from checked integers.
fn balanced_select_expression(indexes: &[u64]) -> String {
    fn build(indexes: &[u64], expression: &mut String) {
        match indexes {
            [] => expression.push('0'),
            [index] => {
                expression.push_str("eq(n,");
                expression.push_str(&index.to_string());
                expression.push(')');
            }
            _ => {
                let midpoint = indexes.len() / 2;
                expression.push('(');
                build(&indexes[..midpoint], expression);
                expression.push('+');
                build(&indexes[midpoint..], expression);
                expression.push(')');
            }
        }
    }

    // Quoting the filter expression is FFmpeg filtergraph syntax (not shell
    // quoting).  It keeps the commas inside `eq(n,...)` from being parsed as
    // filter-chain separators while the argument remains shell-independent.
    let mut expression = String::with_capacity(indexes.len().saturating_mul(12) + 2);
    expression.push(char::from(39));
    build(indexes, &mut expression);
    expression.push(char::from(39));
    expression
}

/// Build the second-pass FFmpeg command.  `fps` is intentionally before
/// `select`: the selected indexes are zero-based positions in the candidate
/// stream produced by the first pass, after frame-rate conversion.
fn selected_ffmpeg_args(
    input: &Path,
    stream0: usize,
    stream1: usize,
    candidate_fps: f64,
    selected_indexes: &[u64],
    lens0: &Path,
    lens1: &Path,
) -> Vec<String> {
    let select = balanced_select_expression(selected_indexes);
    let filter = format!("fps={candidate_fps},select={select}");
    vec![
        "-hide_banner".into(),
        "-nostdin".into(),
        "-y".into(),
        "-i".into(),
        input.to_string_lossy().into_owned(),
        "-map".into(),
        format!("0:{stream0}"),
        "-vf".into(),
        filter.clone(),
        "-fps_mode".into(),
        "passthrough".into(),
        "-q:v".into(),
        "2".into(),
        lens0.join("%08d.jpg").to_string_lossy().into_owned(),
        "-map".into(),
        format!("0:{stream1}"),
        "-vf".into(),
        filter,
        "-fps_mode".into(),
        "passthrough".into(),
        "-q:v".into(),
        "2".into(),
        lens1.join("%08d.jpg").to_string_lossy().into_owned(),
    ]
}

/// Add FFmpeg's input-scoped automatic hardware decoder selection immediately
/// before the corresponding `-i`.  Keeping this as a pure transformation makes
/// the argument ordering easy to verify without requiring a hardware device.
fn with_hwaccel_auto(args: &[String]) -> Vec<String> {
    let mut accelerated = args.to_vec();
    let Some(input_index) = accelerated.iter().position(|arg| arg == "-i") else {
        return accelerated;
    };
    accelerated.splice(
        input_index..input_index,
        ["-hwaccel".to_owned(), "auto".to_owned()],
    );
    accelerated
}

fn reset_candidate_dirs(candidate_root: &Path, lens0: &Path, lens1: &Path) -> Result<(), String> {
    if candidate_root.exists() {
        fs::remove_dir_all(candidate_root).map_err(|error| error.to_string())?;
    }
    fs::create_dir_all(lens0).map_err(|error| error.to_string())?;
    fs::create_dir_all(lens1).map_err(|error| error.to_string())
}

struct StreamingCandidateSelector {
    base_fps: f64,
    candidate_fps: f64,
    score_candidates: bool,
    records: Vec<SelectionRecord>,
    best_by_interval: BTreeMap<usize, (usize, f64)>,
    keyframe_pruner: Option<extraction::KeyframePruner>,
    attitude_timeline: Option<telemetry::AttitudeTimeline>,
    pending_interval_best: Option<(usize, Vec<u8>)>,
}

#[derive(Clone)]
struct CandidateMotionContext {
    config: extraction::KeyframePruningConfig,
    attitude_timeline: Option<telemetry::AttitudeTimeline>,
}

impl StreamingCandidateSelector {
    fn new(base_fps: f64, candidate_fps: f64, score_candidates: bool) -> Self {
        Self {
            base_fps,
            candidate_fps,
            score_candidates,
            records: Vec::new(),
            best_by_interval: BTreeMap::new(),
            keyframe_pruner: None,
            attitude_timeline: None,
            pending_interval_best: None,
        }
    }

    fn enable_keyframe_pruning(
        &mut self,
        config: extraction::KeyframePruningConfig,
        attitude_timeline: Option<telemetry::AttitudeTimeline>,
    ) -> Result<(), String> {
        self.keyframe_pruner =
            Some(extraction::KeyframePruner::new(config).map_err(|error| error.to_string())?);
        self.attitude_timeline = attitude_timeline;
        Ok(())
    }

    fn finalize_pending_keyframe(&mut self, is_last: bool) -> Result<(), String> {
        let Some((record_index, frame)) = self.pending_interval_best.take() else {
            return Ok(());
        };
        let record = self
            .records
            .get(record_index)
            .ok_or_else(|| "候選 keyframe index 超出 selection records".to_owned())?;
        let sequence = record.sequence;
        let timestamp_ms = record.timestamp_ms;
        let timeline = self.attitude_timeline.as_ref();
        let lookup =
            |timestamp_ms: f64| timeline.and_then(|timeline| timeline.interpolate(timestamp_ms));
        let decision = self
            .keyframe_pruner
            .as_mut()
            .ok_or_else(|| "keyframe pruner 尚未初始化".to_owned())?
            .evaluate_hstack_gray8(
                sequence,
                timestamp_ms,
                CANDIDATE_STREAM_WIDTH as u32,
                CANDIDATE_STREAM_HEIGHT as u32,
                CANDIDATE_STREAM_WIDTH,
                &frame,
                Some(&lookup),
                is_last,
            )
            .map_err(|error| error.to_string())?;
        let record = self
            .records
            .get_mut(record_index)
            .ok_or_else(|| "候選 keyframe index 超出 selection records".to_owned())?;
        record.selected = decision.kept;
        record.imu_rotation_from_last_kept_deg = decision.imu_rotation_from_last_kept_deg;
        record.attitude_wxyz = decision.attitude_wxyz;
        record.angular_speed_dps = decision.angular_speed_dps;
        record.visual_novelty = decision.visual_novelty;
        record.selection_reason = decision.selection_reason;
        Ok(())
    }

    #[cfg(test)]
    fn push(&mut self, frame: &[u8]) -> Result<(), String> {
        let sequence = self.records.len() as u64 + 1;
        let timestamp_ms = (sequence.saturating_sub(1) as f64 / self.candidate_fps) * 1000.0;
        self.push_with_timestamp(frame, timestamp_ms)
    }

    fn push_with_timestamp(&mut self, frame: &[u8], timestamp_ms: f64) -> Result<(), String> {
        if frame.len() != CANDIDATE_FRAME_BYTES {
            return Err(format!(
                "候選 raw frame 大小不符：收到 {} bytes，預期 {CANDIDATE_FRAME_BYTES} bytes",
                frame.len()
            ));
        }
        let sequence = self.records.len() as u64 + 1;
        let interval = (((sequence - 1) as f64 / self.candidate_fps) * self.base_fps)
            .floor()
            .max(0.0) as usize;
        if self.keyframe_pruner.is_some()
            && self
                .pending_interval_best
                .as_ref()
                .is_some_and(|(record_index, _)| {
                    self.records
                        .get(*record_index)
                        .is_some_and(|record| record.interval != interval)
                })
        {
            self.finalize_pending_keyframe(false)?;
        }
        let (lens0_score, lens1_score) = if self.score_candidates {
            let (lens0, lens1) = rayon::join(
                || {
                    extraction::calculate_sharpness_from_gray8(
                        CANDIDATE_PROXY_SIZE as u32,
                        CANDIDATE_PROXY_SIZE as u32,
                        CANDIDATE_STREAM_WIDTH,
                        frame,
                    )
                },
                || {
                    extraction::calculate_sharpness_from_gray8(
                        CANDIDATE_PROXY_SIZE as u32,
                        CANDIDATE_PROXY_SIZE as u32,
                        CANDIDATE_STREAM_WIDTH,
                        &frame[CANDIDATE_PROXY_SIZE..],
                    )
                },
            );
            (
                lens0.map_err(|error| error.to_string())?,
                lens1.map_err(|error| error.to_string())?,
            )
        } else {
            let zero = extraction::SharpnessScore {
                laplacian_variance: 0.0,
                tenengrad_mean: 0.0,
                combined: 0.0,
            };
            (zero, zero)
        };
        let pair_score = lens0_score.combined.min(lens1_score.combined);
        let record_index = self.records.len();
        let selected = match self.best_by_interval.get(&interval).copied() {
            Some((previous_index, previous_score)) if pair_score > previous_score => {
                self.records[previous_index].selected = false;
                self.best_by_interval
                    .insert(interval, (record_index, pair_score));
                true
            }
            Some(_) => false,
            None => {
                self.best_by_interval
                    .insert(interval, (record_index, pair_score));
                true
            }
        };
        self.records.push(SelectionRecord {
            interval,
            sequence,
            lens0_source: PathBuf::from(format!("memory://lens0/{sequence:08}")),
            lens1_source: PathBuf::from(format!("memory://lens1/{sequence:08}")),
            lens0_score: lens0_score.combined,
            lens1_score: lens1_score.combined,
            pair_score,
            selected,
            skipped_existing: false,
            timestamp_ms,
            imu_rotation_from_last_kept_deg: None,
            attitude_wxyz: None,
            angular_speed_dps: None,
            visual_novelty: None,
            selection_reason: None,
            output_lens0: None,
            output_lens1: None,
        });
        if self.keyframe_pruner.is_some() && selected {
            self.pending_interval_best = Some((record_index, frame.to_vec()));
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<SelectionRecord>, String> {
        if self.records.is_empty() {
            return Err("FFmpeg 未產生任何記憶體候選影格".to_owned());
        }
        if self.keyframe_pruner.is_some() {
            self.finalize_pending_keyframe(true)?;
        }
        Ok(self.records)
    }
}

enum RawFrameMessage {
    Frame(Vec<u8>),
    Eof,
    Error(String),
}

fn read_raw_frames<R: Read>(mut stdout: R, sender: std::sync::mpsc::SyncSender<RawFrameMessage>) {
    loop {
        let mut frame = vec![0u8; CANDIDATE_FRAME_BYTES];
        let mut filled = 0usize;
        while filled < frame.len() {
            match stdout.read(&mut frame[filled..]) {
                Ok(0) if filled == 0 => {
                    let _ = sender.send(RawFrameMessage::Eof);
                    return;
                }
                Ok(0) => {
                    let _ = sender.send(RawFrameMessage::Error(format!(
                        "FFmpeg rawvideo 在 frame 中途結束：收到 {filled}/{CANDIDATE_FRAME_BYTES} bytes"
                    )));
                    return;
                }
                Ok(read) => filled += read,
                Err(error) => {
                    let _ = sender.send(RawFrameMessage::Error(format!(
                        "讀取 FFmpeg rawvideo 失敗：{error}"
                    )));
                    return;
                }
            }
        }
        if sender.send(RawFrameMessage::Frame(frame)).is_err() {
            return;
        }
    }
}

fn showinfo_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut tokens = line.split_whitespace();
    while let Some(token) = tokens.next() {
        if token == key {
            return tokens.next();
        }
        if let Some(value) = token.strip_prefix(key).filter(|value| !value.is_empty()) {
            return Some(value);
        }
    }
    None
}

/// Parse FFmpeg `showinfo` output without depending on its logger prefix.
/// Returned timestamps are milliseconds in the candidate stream's PTS domain.
fn parse_showinfo_timestamp_ms(line: &str) -> Option<(u64, f64)> {
    if !line.contains("showinfo") {
        return None;
    }
    let frame_index = showinfo_value(line, "n:")?.parse::<u64>().ok()?;
    let seconds = showinfo_value(line, "pts_time:")?.parse::<f64>().ok()?;
    seconds
        .is_finite()
        .then_some((frame_index, seconds * 1000.0))
}

fn read_candidate_stderr<R: Read>(
    stderr: R,
    timestamp_sender: std::sync::mpsc::SyncSender<(u64, f64)>,
) -> String {
    let mut detail = VecDeque::with_capacity(12);
    for line in BufReader::new(stderr).lines() {
        let Ok(line) = line else {
            continue;
        };
        if let Some(timestamp) = parse_showinfo_timestamp_ms(&line) {
            let _ = timestamp_sender.send(timestamp);
            continue;
        }
        // Per-frame showinfo continuation lines are not useful error context
        // and can otherwise grow stderr memory linearly with video duration.
        if line.contains("Parsed_showinfo_") {
            continue;
        }
        if detail.len() == 12 {
            detail.pop_front();
        }
        detail.push_back(line);
    }
    detail.into_iter().collect::<Vec<_>>().join("\n")
}

#[allow(clippy::too_many_arguments)]
fn run_candidate_stream_attempt(
    app: &AppHandle,
    id: &str,
    ffmpeg: &Path,
    args: &[String],
    base_fps: f64,
    candidate_fps: f64,
    score_candidates: bool,
    motion_context: Option<&CandidateMotionContext>,
    control: &JobControl,
    progress: &mut CandidateProgressReporter<'_>,
) -> Result<Vec<SelectionRecord>, String> {
    if control.cancelled.load(Ordering::Acquire) {
        return Err("cancelled".to_owned());
    }
    emit_log(app, id, "info", "執行 FFmpeg 記憶體候選串流");
    let mut child = silent_command(ffmpeg)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("啟動 {} 失敗: {error}", ffmpeg.display()))?;
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("無法讀取 FFmpeg rawvideo".to_owned());
    };
    let Some(stderr) = child.stderr.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("無法讀取 FFmpeg 錯誤輸出".to_owned());
    };
    let (sender, receiver) = sync_channel(2);
    let (timestamp_sender, timestamp_receiver) = sync_channel(16);
    let stdout_reader = thread::spawn(move || read_raw_frames(stdout, sender));
    let stderr_reader = thread::spawn(move || read_candidate_stderr(stderr, timestamp_sender));
    let mut selector = StreamingCandidateSelector::new(base_fps, candidate_fps, score_candidates);
    if let Some(context) = motion_context {
        selector.enable_keyframe_pruning(context.config, context.attitude_timeline.clone())?;
    }
    let mut stream_error = None;
    let mut buffered_timestamps = BTreeMap::new();
    let mut timestamp_stream_unavailable = false;
    let mut estimated_timestamp_count = 0_u64;
    loop {
        if control.cancelled.load(Ordering::Acquire) {
            stream_error = Some("cancelled".to_owned());
            break;
        }
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(RawFrameMessage::Frame(frame)) => {
                let expected_index = selector.records.len() as u64;
                let mut timestamp_ms = buffered_timestamps.remove(&expected_index);
                while timestamp_ms.is_none() && !timestamp_stream_unavailable {
                    match timestamp_receiver.recv_timeout(Duration::from_secs(1)) {
                        Ok((index, value)) if index == expected_index => {
                            timestamp_ms = Some(value);
                        }
                        Ok((index, value)) if index > expected_index => {
                            buffered_timestamps.insert(index, value);
                            // showinfo records are ordered, so an event for a
                            // later frame means this one could not be parsed.
                            break;
                        }
                        Ok(_) => {}
                        Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                            timestamp_stream_unavailable = true;
                        }
                    }
                }
                let timestamp_ms = timestamp_ms.unwrap_or_else(|| {
                    estimated_timestamp_count += 1;
                    (expected_index as f64 / candidate_fps) * 1000.0
                });
                if let Err(error) = selector.push_with_timestamp(&frame, timestamp_ms) {
                    stream_error = Some(error);
                    break;
                }
                progress.report(selector.records.len() as u64);
            }
            Ok(RawFrameMessage::Eof) => break,
            Ok(RawFrameMessage::Error(error)) => {
                stream_error = Some(error);
                break;
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                stream_error = Some("FFmpeg rawvideo reader unexpectedly disconnected".to_owned());
                break;
            }
        }
    }
    let status = if stream_error.is_some() {
        let _ = child.kill();
        let _ = child.wait();
        None
    } else {
        loop {
            if control.cancelled.load(Ordering::Acquire) {
                let _ = child.kill();
                let _ = child.wait();
                stream_error = Some("cancelled".to_owned());
                break None;
            }
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    stream_error = Some(format!("讀取 FFmpeg 候選串流狀態失敗：{error}"));
                    break None;
                }
            }
        }
    };
    drop(receiver);
    drop(timestamp_receiver);
    let _ = stdout_reader.join();
    let stderr = stderr_reader.join().unwrap_or_default();
    if let Some(error) = stream_error {
        return Err(error);
    }
    if control.cancelled.load(Ordering::Acquire) {
        return Err("cancelled".to_owned());
    }
    let status = status.ok_or("FFmpeg 候選串流沒有結束狀態")?;
    if !status.success() {
        return Err(if stderr.trim().is_empty() {
            format!("FFmpeg 候選串流結束，狀態碼 {:?}", status.code())
        } else {
            stderr
        });
    }
    if estimated_timestamp_count > 0 {
        emit_log(
            app,
            id,
            "warning",
            format!(
                "FFmpeg 有 {estimated_timestamp_count} 張候選影格未回報 PTS，已使用 candidate FPS 時間估算"
            ),
        );
    }
    selector.finish()
}

/// Try hardware decoding once, then retry the exact software-decoder command.
/// A failed attempt's in-memory scores are dropped before retrying, and a
/// cancellation never launches the fallback process.
#[allow(clippy::too_many_arguments)]
fn run_candidate_stream_with_fallback(
    app: &AppHandle,
    id: &str,
    ffmpeg: &Path,
    accelerated_args: &[String],
    software_args: &[String],
    base_fps: f64,
    candidate_fps: f64,
    score_candidates: bool,
    motion_context: Option<&CandidateMotionContext>,
    control: &JobControl,
    progress: &mut CandidateProgressReporter<'_>,
) -> Result<Vec<SelectionRecord>, String> {
    match run_candidate_stream_attempt(
        app,
        id,
        ffmpeg,
        accelerated_args,
        base_fps,
        candidate_fps,
        score_candidates,
        motion_context,
        control,
        progress,
    ) {
        Ok(records) => Ok(records),
        Err(error) if error == "cancelled" || control.cancelled.load(Ordering::Acquire) => {
            Err("cancelled".to_owned())
        }
        Err(hardware_error) => {
            emit_log(
                app,
                id,
                "warning",
                format!("FFmpeg 硬體解碼記憶體候選失敗，改用 CPU 重試：{hardware_error}"),
            );
            run_candidate_stream_attempt(
                app,
                id,
                ffmpeg,
                software_args,
                base_fps,
                candidate_fps,
                score_candidates,
                motion_context,
                control,
                progress,
            )
            .map_err(|software_error| {
                if software_error == "cancelled" || control.cancelled.load(Ordering::Acquire) {
                    "cancelled".to_owned()
                } else {
                    format!(
                        "FFmpeg 記憶體候選硬體與軟體解碼皆失敗：硬體錯誤：{hardware_error}；軟體錯誤：{software_error}"
                    )
                }
            })
        }
    }
}

/// Run one FFmpeg pass with automatic hardware decoding, falling back to the
/// exact software command if hardware setup/output validation fails.  The
/// optional expected count lets the second pass reject a partial or excessive
/// select result before it can be committed.
#[allow(clippy::too_many_arguments)]
fn run_ffmpeg_with_fallback(
    app: &AppHandle,
    id: &str,
    ffmpeg: &Path,
    accelerated_args: &[String],
    software_args: &[String],
    output_root: &Path,
    lens0: &Path,
    lens1: &Path,
    expected_count: Option<usize>,
    control: &JobControl,
) -> Result<(), String> {
    let validate_output = || {
        let count = synchronized_candidate_count(lens0, lens1)?;
        if let Some(expected) = expected_count {
            if count != expected {
                return Err(format!(
                    "FFmpeg 產生 {count} 張同步影格，但預期選定 {expected} 張"
                ));
            }
        }
        Ok(())
    };
    let hardware_result =
        run_child(app, id, ffmpeg, accelerated_args, control).and_then(|()| validate_output());
    match hardware_result {
        Ok(()) => Ok(()),
        Err(hardware_error)
            if hardware_error == "cancelled" || control.cancelled.load(Ordering::Acquire) =>
        {
            let _ = reset_candidate_dirs(output_root, lens0, lens1);
            Err("cancelled".to_owned())
        }
        Err(hardware_error) => {
            emit_log(
                app,
                id,
                "warning",
                format!("FFmpeg 硬體解碼候選影格失敗，將改用 CPU 軟體解碼重試：{hardware_error}"),
            );
            if control.cancelled.load(Ordering::Acquire) {
                return Err("cancelled".to_owned());
            }
            reset_candidate_dirs(output_root, lens0, lens1)
                .map_err(|error| format!("FFmpeg 硬體解碼失敗後，無法準備軟體解碼回退：{error}"))?;
            let software_result =
                run_child(app, id, ffmpeg, software_args, control).and_then(|()| validate_output());
            match software_result {
                Ok(()) => {
                    emit_log(app, id, "info", "FFmpeg 候選影格已安全回退至 CPU 軟體解碼");
                    Ok(())
                }
                Err(software_error)
                    if software_error == "cancelled"
                        || control.cancelled.load(Ordering::Acquire) =>
                {
                    let _ = reset_candidate_dirs(output_root, lens0, lens1);
                    Err("cancelled".to_owned())
                }
                Err(software_error) => {
                    let cleanup_suffix = reset_candidate_dirs(output_root, lens0, lens1)
                        .err()
                        .map(|cleanup_error| format!("；且無法清理不完整輸出：{cleanup_error}"))
                        .unwrap_or_default();
                    Err(format!(
                        "FFmpeg 候選影格硬體與軟體解碼皆失敗：硬體錯誤：{hardware_error}；軟體錯誤：{software_error}{cleanup_suffix}"
                    ))
                }
            }
        }
    }
}

fn probe_streams(ffprobe: &Path, input: &Path) -> Result<Value, String> {
    let output = silent_command(ffprobe).args(["-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,time_base,start_time,duration:format=duration,format_name,tags",
        "-of", "json"]).arg(input).output()
        .map_err(|error| format!("ffprobe failed: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    serde_json::from_slice(&output.stdout).map_err(|error| error.to_string())
}

fn stream_indices(probe: &Value, codec_type: &str) -> Vec<usize> {
    probe
        .get("streams")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|stream| stream.get("codec_type").and_then(Value::as_str) == Some(codec_type))
        .filter_map(|stream| {
            stream
                .get("index")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
        })
        .collect()
}

fn candidate_image_names(path: &Path) -> Result<BTreeSet<String>, String> {
    let entries = fs::read_dir(path)
        .map_err(|error| format!("無法讀取候選影格目錄 {}：{error}", path.display()))?;
    entries
        .map(|entry| {
            let entry = entry.map_err(|error| error.to_string())?;
            let entry_path = entry.path();
            let is_image = entry_path.is_file()
                && matches!(
                    entry_path
                        .extension()
                        .and_then(|value| value.to_str())
                        .map(|value| value.to_ascii_lowercase())
                        .as_deref(),
                    Some("jpg") | Some("jpeg") | Some("png")
                );
            Ok(is_image.then(|| entry.file_name().to_string_lossy().into_owned()))
        })
        .filter_map(|entry: Result<Option<String>, String>| entry.transpose())
        .collect()
}

fn prefixed_image_names(path: &Path, prefix: &str) -> Result<BTreeSet<String>, String> {
    if !path.is_dir() {
        return Ok(BTreeSet::new());
    }
    Ok(candidate_image_names(path)?
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .collect())
}

fn sequence_filename(sequence: u64) -> String {
    format!("{sequence:08}.jpg")
}

/// Move FFmpeg's re-numbered second-pass outputs into names carrying the
/// original candidate sequence.  A separate mapped directory avoids rename
/// collisions when a selected sequence already equals a temporary frame name.
fn map_full_res_candidates(
    decoded_lens0: &Path,
    decoded_lens1: &Path,
    mapped_lens0: &Path,
    mapped_lens1: &Path,
    selected_sequences: &[u64],
) -> Result<(), String> {
    if selected_sequences.is_empty() {
        fs::create_dir_all(mapped_lens0).map_err(|error| error.to_string())?;
        fs::create_dir_all(mapped_lens1).map_err(|error| error.to_string())?;
        return Ok(());
    }
    let expected = (1..=selected_sequences.len())
        .map(|index| sequence_filename(index as u64))
        .collect::<BTreeSet<_>>();
    let lens0_names = candidate_image_names(decoded_lens0)?;
    let lens1_names = candidate_image_names(decoded_lens1)?;
    if lens0_names != expected || lens1_names != expected {
        return Err(format!(
            "FFmpeg 第二遍輸出無法與選定序列對應：預期 {} 張，實際 lens0 {} 張、lens1 {} 張",
            expected.len(),
            lens0_names.len(),
            lens1_names.len()
        ));
    }
    if selected_sequences.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("選定候選序列必須嚴格遞增且不可重複".to_owned());
    }
    fs::create_dir_all(mapped_lens0).map_err(|error| error.to_string())?;
    fs::create_dir_all(mapped_lens1).map_err(|error| error.to_string())?;
    for (index, sequence) in selected_sequences.iter().copied().enumerate() {
        let temporary_name = sequence_filename(index as u64 + 1);
        let destination_name = sequence_filename(sequence);
        fs::rename(
            decoded_lens0.join(&temporary_name),
            mapped_lens0.join(&destination_name),
        )
        .map_err(|error| format!("無法對應 lens0 第二遍影格 {temporary_name}：{error}"))?;
        fs::rename(
            decoded_lens1.join(&temporary_name),
            mapped_lens1.join(&destination_name),
        )
        .map_err(|error| format!("無法對應 lens1 第二遍影格 {temporary_name}：{error}"))?;
    }
    Ok(())
}

struct RemoveDirOnDrop(PathBuf);

impl RemoveDirOnDrop {
    fn new(path: PathBuf) -> Self {
        Self(path)
    }
}

impl Drop for RemoveDirOnDrop {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cleanup_stale_full_res_dirs(candidate_root: &Path) -> Result<(), String> {
    let entries = match fs::read_dir(candidate_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.to_string()),
    };
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        if entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
            && entry.file_name().to_string_lossy().starts_with("full-res-")
        {
            fs::remove_dir_all(entry.path()).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn cleanup_obsolete_candidate_cache(candidate_root: &Path) -> Result<(), String> {
    for directory in [candidate_root.join("lens0"), candidate_root.join("lens1")] {
        if directory.exists() {
            fs::remove_dir_all(&directory).map_err(|error| {
                format!("無法移除舊候選影格快取 {}：{error}", directory.display())
            })?;
        }
    }
    let checkpoint = candidate_root.join("candidates.complete.json");
    if checkpoint.exists() {
        fs::remove_file(&checkpoint).map_err(|error| {
            format!(
                "無法移除舊候選影格 checkpoint {}：{error}",
                checkpoint.display()
            )
        })?;
    }
    Ok(())
}

fn source_modified_nanos(metadata: &fs::Metadata) -> Option<String> {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().to_string())
}

#[allow(clippy::too_many_arguments)]
fn load_candidate_selection_checkpoint(
    path: &Path,
    input: &Path,
    base_fps: f64,
    candidate_fps: f64,
    dense_fps: f64,
    skip_blurry: bool,
    keyframe_pruning: bool,
    keyframe_thresholds: extraction::KeyframePruningConfig,
    telemetry_sha256: Option<&str>,
) -> Option<SelectionMetadata> {
    let checkpoint =
        serde_json::from_slice::<CandidateSelectionCheckpoint>(&fs::read(path).ok()?).ok()?;
    let source_metadata = fs::metadata(input).ok()?;
    let selected_count = checkpoint
        .selection
        .selections
        .iter()
        .filter(|record| record.selected)
        .count();
    let records_valid =
        checkpoint
            .selection
            .selections
            .iter()
            .enumerate()
            .all(|(index, record)| {
                record.sequence == index as u64 + 1
                    && record.interval
                        == ((index as f64 / candidate_fps) * base_fps).floor().max(0.0) as usize
                    && record.lens0_score.is_finite()
                    && record.lens1_score.is_finite()
                    && record.pair_score.is_finite()
            });
    let matches = checkpoint.schema_version == CANDIDATE_SELECTION_CHECKPOINT_SCHEMA_VERSION
        && checkpoint.source_path == input.to_string_lossy()
        && checkpoint.source_size == source_metadata.len()
        && checkpoint.source_modified_nanos == source_modified_nanos(&source_metadata)
        && (checkpoint.base_fps - base_fps).abs() < 1e-6
        && (checkpoint.candidate_fps - candidate_fps).abs() < 1e-6
        && (checkpoint.dense_fps - dense_fps).abs() < 1e-6
        && checkpoint.skip_blurry == skip_blurry
        && checkpoint.keyframe_pruning == keyframe_pruning
        && checkpoint.keyframe_thresholds == keyframe_thresholds
        && checkpoint.telemetry_sha256.as_deref() == telemetry_sha256
        && checkpoint.image_format == CANDIDATE_IMAGE_FORMAT
        && checkpoint.selection.schema_version == extraction::SELECTION_METADATA_SCHEMA_VERSION
        && checkpoint.selection.candidate_storage == "memory_rawvideo"
        && (checkpoint.selection.base_fps - base_fps).abs() < 1e-6
        && (checkpoint.selection.candidate_fps - candidate_fps).abs() < 1e-6
        && (checkpoint.selection.requested_dense_fps - dense_fps).abs() < 1e-6
        && checkpoint.selection.sharpness_scoring == skip_blurry
        && !checkpoint.selection.copy_selected_outputs
        && !checkpoint.selection.outputs_committed
        && !checkpoint.selection.cancelled
        && selected_count > 0
        && checkpoint.selection.intervals == selected_count
        && records_valid;
    matches.then_some(checkpoint.selection)
}

#[allow(clippy::too_many_arguments)]
fn write_candidate_selection_checkpoint(
    path: &Path,
    input: &Path,
    base_fps: f64,
    candidate_fps: f64,
    dense_fps: f64,
    skip_blurry: bool,
    keyframe_pruning: bool,
    keyframe_thresholds: extraction::KeyframePruningConfig,
    telemetry_sha256: Option<String>,
    selection: &SelectionMetadata,
) -> Result<(), String> {
    let source_metadata = fs::metadata(input).map_err(|error| error.to_string())?;
    let checkpoint = CandidateSelectionCheckpoint {
        schema_version: CANDIDATE_SELECTION_CHECKPOINT_SCHEMA_VERSION,
        source_path: input.to_string_lossy().into_owned(),
        source_size: source_metadata.len(),
        source_modified_nanos: source_modified_nanos(&source_metadata),
        base_fps,
        candidate_fps,
        dense_fps,
        skip_blurry,
        keyframe_pruning,
        keyframe_thresholds,
        telemetry_sha256,
        image_format: CANDIDATE_IMAGE_FORMAT.to_owned(),
        selection: selection.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&checkpoint).map_err(|error| error.to_string())?;
    extraction::write_bytes_atomic(path, &bytes).map_err(|error| error.to_string())
}

fn synchronized_candidate_count(lens0: &Path, lens1: &Path) -> Result<usize, String> {
    let lens0_names = candidate_image_names(lens0)?;
    let lens1_names = candidate_image_names(lens1)?;
    if lens0_names.is_empty() || lens1_names.is_empty() {
        return Err("FFmpeg 未產生完整雙魚眼候選影格".to_owned());
    }
    if lens0_names != lens1_names {
        let lens0_only = lens0_names.difference(&lens1_names).count();
        let lens1_only = lens1_names.difference(&lens0_names).count();
        return Err(format!(
            "雙魚眼候選序列不同步：lens0 {} 張、lens1 {} 張（僅 lens0 {lens0_only} 張、僅 lens1 {lens1_only} 張）",
            lens0_names.len(),
            lens1_names.len()
        ));
    }
    Ok(lens0_names.len())
}

fn verify_selected_outputs(
    lens0_output: &Path,
    lens1_output: &Path,
    output_prefix: &str,
    selected_sequences: &[u64],
) -> Result<(), String> {
    let expected = selected_sequences
        .iter()
        .map(|sequence| format!("{output_prefix}{sequence:08}.jpg"))
        .collect::<BTreeSet<_>>();
    let lens0_names = prefixed_image_names(lens0_output, output_prefix)?;
    let lens1_names = prefixed_image_names(lens1_output, output_prefix)?;
    if lens0_names != expected || lens1_names != expected {
        return Err(format!(
            "最終雙魚眼輸出不同步或數量不符：預期 {} 張，實際 lens0 {} 張、lens1 {} 張",
            expected.len(),
            lens0_names.len(),
            lens1_names.len()
        ));
    }
    Ok(())
}

fn build_frame_motion_metadata(
    source_index: usize,
    base_fps: f64,
    candidate_fps: f64,
    thresholds: extraction::KeyframePruningConfig,
    selections: &[SelectionRecord],
    timeline: Option<&telemetry::AttitudeTimeline>,
) -> extraction::FrameMotionMetadata {
    let mut previous_kept: Option<(f64, [f64; 4])> = None;
    let mut covered_frame_count = 0usize;
    let mut uncovered_frame_count = 0usize;
    let mut frames = Vec::new();
    for selection in selections
        .iter()
        .filter(|selection| selection.selected || selection.selection_reason.is_some())
    {
        let attitude_wxyz = selection.attitude_wxyz.or_else(|| {
            timeline
                .and_then(|timeline| timeline.interpolate(selection.timestamp_ms))
                .map(|sample| sample.quaternion())
        });
        if attitude_wxyz.is_some() {
            covered_frame_count += 1;
        } else {
            uncovered_frame_count += 1;
        }
        let (derived_rotation, derived_speed) = previous_kept
            .zip(attitude_wxyz)
            .and_then(|((previous_timestamp_ms, previous), current)| {
                let elapsed_ms = selection.timestamp_ms - previous_timestamp_ms;
                let rotation_deg = telemetry::quaternion_angle_deg(previous, current);
                (elapsed_ms > f64::EPSILON && rotation_deg.is_finite())
                    .then(|| (rotation_deg, rotation_deg / (elapsed_ms / 1_000.0)))
            })
            .map_or((None, None), |(rotation, speed)| {
                (Some(rotation), Some(speed))
            });
        let rotation = selection
            .imu_rotation_from_last_kept_deg
            .or(derived_rotation);
        let angular_speed = selection.angular_speed_dps.or(derived_speed);
        frames.push(extraction::FrameMotionRecord {
            sequence: selection.sequence,
            timestamp_ms: selection.timestamp_ms,
            imu_rotation_from_last_kept_deg: rotation,
            attitude_wxyz,
            angular_speed_dps: angular_speed,
            visual_novelty: selection.visual_novelty,
            selection_reason: selection
                .selection_reason
                .clone()
                .or_else(|| selection.selected.then(|| "intervalBest".to_owned())),
            selected: selection.selected,
        });
        if selection.selected {
            if let Some(attitude) = attitude_wxyz {
                previous_kept = Some((selection.timestamp_ms, attitude));
            } else {
                previous_kept = None;
            }
        }
    }
    let telemetry_coverage = timeline.map(|timeline| {
        let diagnostics = timeline.diagnostics();
        extraction::FrameMotionTelemetryCoverage {
            sample_count: diagnostics.sample_count,
            valid_sample_count: diagnostics.valid_sample_count,
            first_timestamp_ms: diagnostics.first_timestamp_ms,
            last_timestamp_ms: diagnostics.last_timestamp_ms,
            covered_frame_count,
            uncovered_frame_count,
        }
    });
    let mut metadata = extraction::FrameMotionMetadata::new(thresholds, frames);
    metadata.source_index = Some(source_index);
    metadata.base_fps = Some(base_fps);
    metadata.candidate_fps = Some(candidate_fps);
    metadata.telemetry_coverage = telemetry_coverage;
    metadata
}

fn run_extract(
    app: &AppHandle,
    id: &str,
    manifest: &ProjectManifest,
    control: &JobControl,
) -> Result<Vec<String>, String> {
    if manifest.input_paths.is_empty() {
        return Err("此資料夾沒有原始影片；可直接執行現有影格適用的遮罩或對齊階段".to_owned());
    }
    let ffmpeg = find_executable("ffmpeg").ok_or("在系統 PATH 中找不到 FFmpeg")?;
    let ffprobe = find_executable("ffprobe").ok_or("在系統 PATH 中找不到 ffprobe")?;
    let output = PathBuf::from(&manifest.output_path);
    let (base_fps, dense_fps, skip_blurry) = extract_frame_settings(&manifest.settings);
    let (keyframe_pruning, keyframe_thresholds) = keyframe_pruning_settings(&manifest.settings);
    let candidate_fps = if skip_blurry { dense_fps } else { base_fps };
    let total_sources = manifest.input_paths.len().max(1);
    let mut telemetry_streams = Vec::new();
    let mut normalized_telemetry = Vec::new();
    for (source_index, raw_input) in manifest.input_paths.iter().enumerate() {
        let input = PathBuf::from(raw_input);
        let current_source = input.to_string_lossy().into_owned();
        let probe = probe_streams(&ffprobe, &input)?;
        let streams = stream_indices(&probe, "video");
        let candidate_total = expected_candidate_frames(&probe, candidate_fps);
        if streams.len() < 2 {
            return Err(format!(
                "{} 未包含兩路可辨識的雙魚眼 video stream",
                input.display()
            ));
        }
        let metadata = output.join("metadata");
        fs::create_dir_all(&metadata).map_err(|error| error.to_string())?;
        fs::write(
            metadata.join(format!("source{source_index:03}_streams.json")),
            serde_json::to_vec_pretty(&probe).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        // Parse/cache normalized telemetry before the candidate pass so fused
        // attitude can be SLERP-interpolated at each post-fps frame PTS. A
        // missing stream only disables the IMU term; visual novelty and the
        // max-gap fallback remain available.
        let normalized_path = metadata.join(format!("source{source_index:03}_telemetry.json"));
        let mut attitude_timeline = None;
        let telemetry_sha256 = match telemetry::parse_and_write(
            &input,
            &normalized_path,
            control.cancelled.clone(),
        ) {
            Ok(export) => {
                normalized_telemetry.push(json!({
                    "sourceIndex": source_index,
                    "path": export.path.to_string_lossy(),
                    "cameraModel": export.camera_model,
                    "normalizedImuSampleCount": export.normalized_imu_sample_count,
                    "fusedAttitudeSampleCount": export.fused_attitude_sample_count,
                    "appliedToColmap": false
                }));
                match telemetry::read_normalized_telemetry(&normalized_path) {
                    Ok(normalized) => {
                        attitude_timeline = Some(normalized.attitude_timeline());
                        file_sha256(&normalized_path).ok()
                    }
                    Err(error) => {
                        emit_log(
                            app,
                            id,
                            "warning",
                            format!(
                                "無法讀回來源 {} 的標準化 telemetry，IMU keyframe term 已停用：{error}",
                                source_index + 1
                            ),
                        );
                        None
                    }
                }
            }
            Err(error) if !control.cancelled.load(Ordering::Acquire) => {
                emit_log(
                    app,
                    id,
                    "warning",
                    format!(
                        "無法解析來源 {} 的標準化 telemetry；改用 visual novelty＋max gap：{error}",
                        source_index + 1
                    ),
                );
                None
            }
            Err(error) => return Err(error),
        };
        let motion_context = keyframe_pruning.then(|| CandidateMotionContext {
            config: keyframe_thresholds,
            attitude_timeline: attitude_timeline.clone(),
        });
        let candidate_root = output
            .join("capture")
            .join(format!("source{source_index:03}"));
        cleanup_stale_full_res_dirs(&candidate_root)?;
        cleanup_obsolete_candidate_cache(&candidate_root)?;
        let output_prefix = format!("source{source_index:03}_");
        let selection_metadata_path =
            metadata.join(format!("source{source_index:03}_selection.json"));
        let transaction_root = candidate_root.join(format!(
            "full-res-{}-{}",
            std::process::id(),
            FULL_RES_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let transaction_cleanup = RemoveDirOnDrop::new(transaction_root.clone());
        fs::create_dir_all(&transaction_root).map_err(|error| error.to_string())?;
        let pending_selection_metadata = transaction_root.join("selection.pending.json");
        let pending_frame_motion = transaction_root.join("frame-motion.pending.json");
        let frame_motion_path = metadata.join(format!("source{source_index:03}_frame_motion.json"));
        let selection_checkpoint = candidate_root.join("selection.checkpoint.json");
        let selection_metadata = if let Some(selection) = load_candidate_selection_checkpoint(
            &selection_checkpoint,
            &input,
            base_fps,
            candidate_fps,
            dense_fps,
            skip_blurry,
            keyframe_pruning,
            keyframe_thresholds,
            telemetry_sha256.as_deref(),
        ) {
            emit_log(
                app,
                id,
                "info",
                format!("來源 {} 已沿用記憶體評分 checkpoint", source_index + 1),
            );
            selection
        } else {
            emit_progress_detailed(
                app,
                id,
                &StageName::Extract,
                "selecting-in-memory",
                source_stage_progress(source_index, total_sources, 0.0),
                format!(
                    "正在記憶體中同步解碼並評分來源 {} 的雙魚眼候選影格",
                    source_index + 1
                ),
                "running",
                false,
                Some(0),
                candidate_total,
                Some(current_source.clone()),
                None,
            );
            let software_args =
                candidate_ffmpeg_args(&input, streams[0], streams[1], candidate_fps);
            let accelerated_args = with_hwaccel_auto(&software_args);
            let mut candidate_progress = CandidateProgressReporter {
                app,
                id,
                source_index,
                total_sources,
                expected_frames: candidate_total,
                current_item: current_source.clone(),
                highest_processed: 0,
                last_emitted_at: None,
            };
            let selection_records = run_candidate_stream_with_fallback(
                app,
                id,
                &ffmpeg,
                &accelerated_args,
                &software_args,
                base_fps,
                candidate_fps,
                skip_blurry,
                motion_context.as_ref(),
                control,
                &mut candidate_progress,
            )?;
            let selected_intervals = selection_records
                .iter()
                .filter(|record| record.selected)
                .count();
            let selection = SelectionMetadata {
                schema_version: extraction::SELECTION_METADATA_SCHEMA_VERSION,
                candidate_storage: "memory_rawvideo".to_owned(),
                base_fps,
                candidate_fps,
                requested_dense_fps: dense_fps,
                sharpness_scoring: skip_blurry,
                sharpness_analysis_max_dimension: skip_blurry
                    .then_some(extraction::SHARPNESS_MAX_DIMENSION),
                copy_selected_outputs: false,
                outputs_committed: false,
                intervals: selected_intervals,
                cancelled: false,
                selections: selection_records,
            };
            write_candidate_selection_checkpoint(
                &selection_checkpoint,
                &input,
                base_fps,
                candidate_fps,
                dense_fps,
                skip_blurry,
                keyframe_pruning,
                keyframe_thresholds,
                telemetry_sha256.clone(),
                &selection,
            )?;
            selection
        };
        emit_progress_detailed(
            app,
            id,
            &StageName::Extract,
            "selecting-in-memory",
            source_stage_progress(
                source_index,
                total_sources,
                CANDIDATE_SELECTION_PROGRESS_SHARE,
            ),
            format!(
                "來源 {} 已完成 {} 組候選影格評分",
                source_index + 1,
                selection_metadata.selections.len()
            ),
            "running",
            false,
            Some(selection_metadata.selections.len() as u64),
            Some(selection_metadata.selections.len() as u64),
            Some(current_source.clone()),
            None,
        );
        extraction::write_selection_metadata_atomic(
            &pending_selection_metadata,
            &selection_metadata,
        )
        .map_err(|error| error.to_string())?;
        let frame_motion = build_frame_motion_metadata(
            source_index,
            base_fps,
            candidate_fps,
            keyframe_thresholds,
            &selection_metadata.selections,
            attitude_timeline.as_ref(),
        );
        extraction::write_frame_motion_metadata_atomic(&pending_frame_motion, &frame_motion)
            .map_err(|error| error.to_string())?;
        let pruned_intervals = frame_motion
            .frames
            .iter()
            .filter(|frame| !frame.selected)
            .count();
        if keyframe_pruning {
            emit_log(
                app,
                id,
                "info",
                format!(
                    "來源 {} 的動態 keyframe 剪枝保留 {} / {} 組 base-FPS 候選（移除 {pruned_intervals} 組）",
                    source_index + 1,
                    frame_motion.frames.len().saturating_sub(pruned_intervals),
                    frame_motion.frames.len(),
                ),
            );
        }

        let mut selected_sequences = selection_metadata
            .selections
            .iter()
            .filter(|record| record.selected)
            .map(|record| record.sequence)
            .collect::<Vec<_>>();
        if selected_sequences.contains(&0) {
            return Err("選定的候選序列必須從 1 開始".to_owned());
        }
        selected_sequences.sort_unstable();
        if selected_sequences.is_empty() {
            return Err("來源沒有可提交的選定雙魚眼候選影格".to_owned());
        }
        if selected_sequences.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("選定的候選序列不可重複".to_owned());
        }
        let selected_indexes = selected_sequences
            .iter()
            .map(|sequence| sequence.saturating_sub(1))
            .collect::<Vec<_>>();

        // The first pass only scores bounded gray8 frames in memory. Decode
        // selected frames again at native resolution in temporary directories.
        // The drop guard removes every intermediate on success, failure, or
        // cancellation.
        let full_res_root = transaction_root.join("native");
        let decoded_lens0 = full_res_root.join("decoded/lens0");
        let decoded_lens1 = full_res_root.join("decoded/lens1");
        let mapped_lens0 = full_res_root.join("mapped/lens0");
        let mapped_lens1 = full_res_root.join("mapped/lens1");
        reset_candidate_dirs(&full_res_root, &decoded_lens0, &decoded_lens1)?;
        fs::create_dir_all(&mapped_lens0).map_err(|error| error.to_string())?;
        fs::create_dir_all(&mapped_lens1).map_err(|error| error.to_string())?;

        if !selected_sequences.is_empty() {
            emit_progress_detailed(
                app,
                id,
                &StageName::Extract,
                "decoding-full-resolution",
                source_stage_progress(
                    source_index,
                    total_sources,
                    CANDIDATE_SELECTION_PROGRESS_SHARE,
                ),
                format!(
                    "正在以原始解析度重新解碼來源 {} 的 {} 組選定影格",
                    source_index + 1,
                    selected_sequences.len()
                ),
                "running",
                false,
                Some(0),
                Some(selected_sequences.len() as u64),
                Some(current_source.clone()),
                None,
            );
            let software_args = selected_ffmpeg_args(
                &input,
                streams[0],
                streams[1],
                candidate_fps,
                &selected_indexes,
                &decoded_lens0,
                &decoded_lens1,
            );
            let accelerated_args = with_hwaccel_auto(&software_args);
            run_ffmpeg_with_fallback(
                app,
                id,
                &ffmpeg,
                &accelerated_args,
                &software_args,
                &full_res_root,
                &decoded_lens0,
                &decoded_lens1,
                Some(selected_sequences.len()),
                control,
            )?;
            map_full_res_candidates(
                &decoded_lens0,
                &decoded_lens1,
                &mapped_lens0,
                &mapped_lens1,
                &selected_sequences,
            )?;
            emit_progress_detailed(
                app,
                id,
                &StageName::Extract,
                "decoding-full-resolution",
                source_stage_progress(
                    source_index,
                    total_sources,
                    CANDIDATE_SELECTION_PROGRESS_SHARE + FULL_RESOLUTION_PROGRESS_SHARE,
                ),
                format!(
                    "來源 {} 已完成 {} 組原始解析度影格解碼",
                    source_index + 1,
                    selected_sequences.len()
                ),
                "running",
                false,
                Some(selected_sequences.len() as u64),
                Some(selected_sequences.len() as u64),
                Some(current_source.clone()),
                None,
            );
        }

        if control.cancelled.load(Ordering::Acquire) {
            return Err("cancelled".to_owned());
        }
        let commit_metadata_path = full_res_root.join("selection.commit.json");
        let commit_request = ExtractionRequest {
            lens0_candidates: mapped_lens0,
            lens1_candidates: mapped_lens1,
            lens0_output: output.join("images/lens0"),
            lens1_output: output.join("images/lens1"),
            output_prefix: output_prefix.clone(),
            base_fps,
            // Keep the original candidate cadence so sequence/interval identity
            // remains identical to the first scoring pass.
            candidate_fps,
            dense_fps,
            score_candidates: false,
            copy_selected_outputs: true,
            skip_completed: true,
            // Commit metadata is transient; the first pass's selection metadata
            // is the durable record and must not be replaced.
            metadata_path: Some(commit_metadata_path),
        };
        let app_clone = app.clone();
        let id_owned = id.to_owned();
        let cancelled = control.cancelled.clone();
        let completed_intervals = Arc::new(AtomicU64::new(0));
        let completed_intervals_for_callback = completed_intervals.clone();
        let source_offset = source_stage_progress(
            source_index,
            total_sources,
            CANDIDATE_SELECTION_PROGRESS_SHARE + FULL_RESOLUTION_PROGRESS_SHARE,
        );
        let source_scale =
            (1.0 - CANDIDATE_SELECTION_PROGRESS_SHARE - FULL_RESOLUTION_PROGRESS_SHARE)
                / total_sources as f32;
        let commit_summary = extraction::extract_selected_pairs(
            &commit_request,
            || cancelled.load(Ordering::Acquire),
            move |event| {
                let total_intervals = event.total_intervals as u64;
                let completed = extraction_completed_count(
                    &completed_intervals_for_callback,
                    event.stage,
                    event.interval,
                    event.total_intervals,
                );
                emit_progress_detailed(
                    &app_clone,
                    &id_owned,
                    &StageName::Extract,
                    "committing",
                    source_offset + event.fraction * source_scale,
                    format!("來源 {}：{}", source_index + 1, event.message),
                    "running",
                    false,
                    Some(completed),
                    Some(total_intervals),
                    Some(current_source.clone()),
                    None,
                )
            },
        )
        .map_err(|error| error.to_string())?;
        if commit_summary.cancelled {
            return Err("cancelled".to_owned());
        }
        if commit_summary.selected_intervals != selected_sequences.len() {
            return Err(format!(
                "第二遍選定數量不符：第一遍 {} 張、第二遍 {} 張",
                selected_sequences.len(),
                commit_summary.selected_intervals
            ));
        }
        verify_selected_outputs(
            &output.join("images/lens0"),
            &output.join("images/lens1"),
            &output_prefix,
            &selected_sequences,
        )?;
        extraction::mark_selection_outputs_committed(
            &pending_selection_metadata,
            &selected_sequences,
            &output.join("images/lens0"),
            &output.join("images/lens1"),
            &output_prefix,
        )
        .map_err(|error| error.to_string())?;
        extraction::promote_selection_metadata(
            &pending_selection_metadata,
            &selection_metadata_path,
        )
        .map_err(|error| error.to_string())?;
        extraction::promote_selection_metadata(&pending_frame_motion, &frame_motion_path)
            .map_err(|error| error.to_string())?;
        // Release native-resolution intermediates before preserving raw data
        // streams so long captures do not retain avoidable temporary storage.
        drop(transaction_cleanup);

        for stream in stream_indices(&probe, "data") {
            if control.cancelled.load(Ordering::Acquire) {
                return Err("cancelled".to_owned());
            }
            let final_path = metadata.join(format!(
                "source{source_index:03}_stream{stream}_telemetry.bin"
            ));
            if final_path.is_file() {
                telemetry_streams.push(json!({
                    "sourceIndex": source_index,
                    "streamIndex": stream,
                    "path": final_path.to_string_lossy(),
                    "format": "ffmpeg-data-stream-copy"
                }));
                continue;
            }
            let partial_path = final_path.with_extension("bin.partial");
            let args = vec![
                "-hide_banner".into(),
                "-nostdin".into(),
                "-y".into(),
                "-i".into(),
                input.to_string_lossy().into_owned(),
                "-map".into(),
                format!("0:{stream}"),
                "-c".into(),
                "copy".into(),
                "-f".into(),
                "data".into(),
                partial_path.to_string_lossy().into_owned(),
            ];
            match run_child(app, id, &ffmpeg, &args, control) {
                Ok(()) => {
                    fs::rename(&partial_path, &final_path).map_err(|error| error.to_string())?;
                    telemetry_streams.push(json!({
                        "sourceIndex": source_index,
                        "streamIndex": stream,
                        "path": final_path.to_string_lossy(),
                        "format": "ffmpeg-data-stream-copy"
                    }));
                }
                Err(error) if !control.cancelled.load(Ordering::Acquire) => {
                    let _ = fs::remove_file(&partial_path);
                    emit_log(
                        app,
                        id,
                        "warning",
                        format!("無法封裝 telemetry stream {stream}: {error}"),
                    );
                }
                Err(error) => return Err(error),
            }
        }
    }
    let metadata = output.join("metadata");
    fs::create_dir_all(&metadata).map_err(|e| e.to_string())?;
    let imu_status = if !normalized_telemetry.is_empty() {
        "normalized-telemetry-available"
    } else if telemetry_streams.is_empty() {
        "not-detected"
    } else {
        "raw-data-streams-preserved"
    };
    fs::write(metadata.join("capture.json"), serde_json::to_vec_pretty(&json!({
        "schemaVersion": 5, "canonicalProjection": "native_fisheye", "sources": manifest.input_paths,
        "lensCount": 2, "baseFps": base_fps, "candidateFps": candidate_fps,
        "requestedDenseFps": dense_fps, "skipBlurry": skip_blurry,
        "candidateImageFormat": CANDIDATE_IMAGE_FORMAT,
        "candidateProxyMaxDimension": extraction::SHARPNESS_MAX_DIMENSION,
        "candidateStorage": "memory",
        "candidatePixelFormat": "gray8",
        "selectedFrameDecode": "second-pass-native-resolution",
        "fullResolutionOutputsCommitted": true,
        "sharpness": if skip_blurry { "gaussian+laplacian+tenengrad; conservative pair minimum" } else { "disabled" },
        "sharpnessAnalysisMaxDimension": if skip_blurry { Some(extraction::SHARPNESS_MAX_DIMENSION) } else { None },
        "motionAdaptiveCadence": keyframe_pruning,
        "keyframeThresholds": keyframe_thresholds,
        "candidateTimestampSource": "ffmpeg post-fps PTS (showinfo), candidate-FPS estimate only as per-frame fallback",
        "frameIdentity": "same filename across lens folders",
        "telemetryStreams": telemetry_streams,
        "normalizedTelemetry": normalized_telemetry,
        "imu": {"status": imu_status, "appliedToColmap":false},
        "warnings":["DJI quaternion is not copied into COLMAP without a verified coordinate transform"]
    })).unwrap()).map_err(|e| e.to_string())?;
    Ok(vec![
        output.join("images").to_string_lossy().into_owned(),
        metadata.join("capture.json").to_string_lossy().into_owned(),
    ])
}

fn run_mask(
    app: &AppHandle,
    id: &str,
    manifest: &ProjectManifest,
    control: &JobControl,
) -> Result<Vec<String>, String> {
    if !mask_enabled(&manifest.settings) {
        emit_log(app, id, "info", "未啟用 YOLO 或天空過濾，已略過遮罩階段");
        return Ok(Vec::new());
    }
    let root = PathBuf::from(&manifest.output_path);
    let classes = mask_classes(&manifest.settings);
    let confidence = mask_confidence(&manifest.settings);
    let mask_sky = setting_bool(&manifest.settings, "/mask/maskSky", false);
    let mut optical_occlusions = BTreeMap::new();
    for (source_index, raw_input) in manifest.input_paths.iter().enumerate() {
        let input = PathBuf::from(raw_input);
        let is_osv = input
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("osv"));
        if !is_osv || !input.is_file() {
            continue;
        }
        match telemetry::read_dji_optical_occlusions(&input) {
            Ok(Some(calibrations)) => {
                optical_occlusions.insert(format!("source{source_index:03}_"), calibrations);
            }
            Ok(None) => emit_log(
                app,
                id,
                "warning",
                format!(
                    "{} 未提供可用的 DJI 光學遮擋曲線；改用完整 fisheye 圓",
                    input.display()
                ),
            ),
            Err(error) => emit_log(
                app,
                id,
                "warning",
                format!(
                    "無法讀取 {} 的 DJI 光學遮擋曲線（{error}）；改用完整 fisheye 圓",
                    input.display()
                ),
            ),
        }
    }
    let model_cache_dir = if classes.is_empty() && !mask_sky {
        None
    } else {
        Some(
            app.path()
                .app_data_dir()
                .map_err(|error| format!("無法取得應用程式模型資料夾：{error}"))?
                .join("models"),
        )
    };
    let request = MaskRequest {
        images_dir: root.join("images"),
        masks_dir: root.join("masks"),
        colmap_masks_dir: root.join("masks_colmap"),
        classes,
        mask_sky,
        confidence: if confidence > 1.0 {
            confidence / 100.0
        } else {
            confidence
        }
        .clamp(0.01, 0.99) as f32,
        valid_radius_ratio: DJI_VALID_RADIUS_RATIO as f32,
        optical_occlusions,
        // Resume partial runs without repeating expensive inference. The mask
        // module skips only when both outputs decode and match the source size.
        skip_verified: true,
        model_dir: manifest
            .settings
            .pointer("/mask/modelDir")
            .and_then(Value::as_str)
            .filter(|path| !path.trim().is_empty())
            .map(PathBuf::from),
        model_cache_dir,
        yolo_model: None,
        skyseg_model: None,
        execution_provider: manifest
            .settings
            .pointer("/mask/executionProvider")
            .and_then(Value::as_str)
            .map(str::to_string),
    };
    let stage = StageName::Mask;
    let app_clone = app.clone();
    let id_owned = id.to_string();
    let summary = masking::process_mask_batch(&request, &control.mask_cancel, move |event| {
        let total = (event.total > 0).then_some(event.total as u64);
        emit_progress_detailed(
            &app_clone,
            &id_owned,
            &stage,
            "masking",
            event.fraction,
            event.message,
            "running",
            false,
            total.map(|_| event.completed.min(event.total) as u64),
            total,
            Some(event.input.to_string_lossy().into_owned()),
            None,
        );
    })
    .map_err(|error| error.to_string())?;
    if summary.failed > 0 {
        return Err(format!("{} 個遮罩處理失敗，請查看處理紀錄", summary.failed));
    }
    Ok(vec![
        root.join("masks").to_string_lossy().into_owned(),
        root.join("masks_colmap").to_string_lossy().into_owned(),
    ])
}

fn is_supported_colmap_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("png") | Some("jpg") | Some("jpeg")
    )
}

fn dual_fisheye_registration_totals(root: &Path) -> Result<(u64, u64), String> {
    let lens_names = ["lens0", "lens1"]
        .map(|lens| {
            fs::read_dir(root.join("images").join(lens))
                .map_err(|error| error.to_string())?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())
                .map(|entries| {
                    entries
                        .into_iter()
                        .filter(|entry| {
                            entry.path().is_file() && is_supported_colmap_image(&entry.path())
                        })
                        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
                        .collect::<BTreeSet<_>>()
                })
        })
        .into_iter()
        .collect::<Result<Vec<_>, String>>()?;
    let independent_images = lens_names
        .iter()
        .map(BTreeSet::len)
        .sum::<usize>()
        .try_into()
        .map_err(|_| "COLMAP 獨立影像數量超出支援範圍".to_owned())?;
    let rig_frames = lens_names[0]
        .union(&lens_names[1])
        .count()
        .try_into()
        .map_err(|_| "COLMAP rig 影格數量超出支援範圍".to_owned())?;
    Ok((independent_images, rig_frames))
}

const TEMPORAL_PAIR_MAX_GAP_MS: f64 = 700.0;
const TEMPORAL_PAIR_MIN_ROTATION_DEG: f64 = 4.0;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameMotionRecord {
    #[serde(default)]
    sequence: Option<u64>,
    #[serde(default, alias = "timestamp_ms")]
    timestamp_ms: Option<f64>,
    #[serde(default, alias = "imu_rotation_from_last_kept_deg")]
    imu_rotation_from_last_kept_deg: Option<f64>,
    #[serde(default, alias = "attitude_wxyz")]
    attitude_wxyz: Option<[f64; 4]>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameMotionFile {
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    frames: Vec<FrameMotionRecord>,
    #[serde(default, alias = "max_gap_ms")]
    max_gap_ms: Option<f64>,
    #[serde(default, alias = "min_rotation_deg")]
    min_rotation_deg: Option<f64>,
    #[serde(default)]
    pruning: Option<FrameMotionPruning>,
    #[serde(default)]
    thresholds: Option<FrameMotionPruning>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameMotionPruning {
    #[serde(default, alias = "max_gap_ms")]
    max_gap_ms: Option<f64>,
    #[serde(default, alias = "min_rotation_deg")]
    min_rotation_deg: Option<f64>,
}

#[derive(Debug, Clone, Default)]
struct SourceFrameMotion {
    frames: BTreeMap<u64, FrameMotionRecord>,
    max_gap_ms: f64,
    min_rotation_deg: f64,
}

fn source_name_from_image(name: &str) -> &str {
    name.split_once('_').map_or("source", |(source, _)| source)
}

fn sequence_from_image_name(name: &str) -> Option<u64> {
    Path::new(name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.rsplit_once('_').map(|(_, sequence)| sequence))
        .and_then(|sequence| sequence.parse::<u64>().ok())
}

fn load_frame_motion_metadata(root: &Path) -> BTreeMap<String, SourceFrameMotion> {
    let metadata = root.join("metadata");
    let Ok(entries) = fs::read_dir(metadata) else {
        return BTreeMap::new();
    };
    let mut files = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let source = name.strip_suffix("_frame_motion.json")?;
            source
                .starts_with("source")
                .then_some((source.to_owned(), path))
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.0.cmp(&right.0));

    files
        .into_iter()
        .filter_map(|(source, path)| {
            let bytes = fs::read(path).ok()?;
            let payload = serde_json::from_slice::<FrameMotionFile>(&bytes).ok()?;
            if payload.schema_version != Some(1) {
                return None;
            }
            let frames = payload
                .frames
                .into_iter()
                .filter_map(|record| record.sequence.map(|sequence| (sequence, record)))
                .collect::<BTreeMap<_, _>>();
            let max_gap_ms = payload
                .thresholds
                .as_ref()
                .and_then(|thresholds| thresholds.max_gap_ms)
                .or_else(|| {
                    payload
                        .pruning
                        .as_ref()
                        .and_then(|pruning| pruning.max_gap_ms)
                })
                .or(payload.max_gap_ms)
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(TEMPORAL_PAIR_MAX_GAP_MS);
            let min_rotation_deg = payload
                .thresholds
                .as_ref()
                .and_then(|thresholds| thresholds.min_rotation_deg)
                .or_else(|| {
                    payload
                        .pruning
                        .as_ref()
                        .and_then(|pruning| pruning.min_rotation_deg)
                })
                .or(payload.min_rotation_deg)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .unwrap_or(TEMPORAL_PAIR_MIN_ROTATION_DEG);
            (!frames.is_empty()).then_some((
                source,
                SourceFrameMotion {
                    frames,
                    max_gap_ms,
                    min_rotation_deg,
                },
            ))
        })
        .collect()
}

fn motion_pair_is_compatible(
    motion: Option<&SourceFrameMotion>,
    left_sequence: u64,
    right_sequence: u64,
) -> Option<bool> {
    let motion = motion?;
    let left = motion.frames.get(&left_sequence)?;
    let right = motion.frames.get(&right_sequence)?;
    let elapsed_ms = match (left.timestamp_ms, right.timestamp_ms) {
        (Some(left), Some(right)) if left.is_finite() && right.is_finite() => {
            Some((right - left).abs())
        }
        _ => None,
    };
    let lower_sequence = left_sequence.min(right_sequence);
    let upper_sequence = left_sequence.max(right_sequence);
    let direct_rotation_deg = match (left.attitude_wxyz, right.attitude_wxyz) {
        (Some(left), Some(right)) => {
            let angle = telemetry::quaternion_angle_deg(left, right);
            if !angle.is_finite() {
                // A malformed attitude must never silently prune a link.
                return Some(true);
            }
            Some(angle)
        }
        _ => None,
    };
    let mut saw_rotation = direct_rotation_deg.is_some();
    let accumulated_rotation_deg = motion
        .frames
        .iter()
        .filter(|(sequence, _)| **sequence > lower_sequence && **sequence <= upper_sequence)
        .filter_map(|(_, frame)| frame.imu_rotation_from_last_kept_deg)
        .filter(|value| value.is_finite())
        .map(|value| {
            saw_rotation = true;
            value.abs()
        })
        .sum::<f64>();
    let rotation_deg = direct_rotation_deg.unwrap_or(accumulated_rotation_deg);
    if elapsed_ms.is_none() && !saw_rotation {
        return None;
    }
    Some(
        elapsed_ms.is_some_and(|elapsed| elapsed <= motion.max_gap_ms)
            || (saw_rotation && rotation_deg >= motion.min_rotation_deg),
    )
}

fn conditional_temporal_pair(
    pairs: &mut BTreeSet<String>,
    left_lens: usize,
    right_lens: usize,
    left: &str,
    right: &str,
    offset: usize,
    source_motion: Option<&SourceFrameMotion>,
) {
    let should_keep = if offset <= 1 {
        true
    } else {
        match (
            sequence_from_image_name(left),
            sequence_from_image_name(right),
        ) {
            (Some(left_sequence), Some(right_sequence)) => {
                // Missing or incomplete motion metadata intentionally falls
                // back to the pre-IMU pair graph instead of risking a
                // disconnected model.
                motion_pair_is_compatible(source_motion, left_sequence, right_sequence)
                    .unwrap_or(true)
            }
            // Unknown filename identity must preserve the legacy graph.
            _ => true,
        }
    };
    if should_keep {
        pairs.insert(format!("lens{left_lens}/{left} lens{right_lens}/{right}"));
    }
}

fn write_rig_and_pairs(root: &Path) -> Result<u64, String> {
    let rig_config = root.join("rig_config.json");
    let legacy_default = json!([{"cameras":[
        {"image_prefix":"lens0/","ref_sensor":true},{"image_prefix":"lens1/"}]}]);
    let calibrated_default = json!([{"cameras":[
    {"image_prefix":"lens0/","ref_sensor":true},
    {
        "image_prefix":"lens1/",
        "cam_from_rig_rotation":[0.0,0.0,1.0,0.0],
        "cam_from_rig_translation":[0.0,0.0,0.0]
    }]}]);
    let should_write_calibrated_default = if rig_config.is_file() {
        serde_json::from_slice::<Value>(&fs::read(&rig_config).map_err(|e| e.to_string())?)
            .is_ok_and(|config| config == legacy_default)
    } else {
        true
    };
    if should_write_calibrated_default {
        // DJI's two native fisheye streams are upright and back-to-back.  Model
        // them as a co-located panoramic rig: lens1 is lens0 rotated 180° about
        // the camera Y axis.  Also migrate only the exact uncalibrated default
        // emitted by older GS360 Studio versions; preserve configs that differ
        // from that generated legacy value.
        fs::write(
            &rig_config,
            serde_json::to_vec_pretty(&calibrated_default).unwrap(),
        )
        .map_err(|e| e.to_string())?;
    }
    let lens1 = root.join("images/lens1");
    let mut names: Vec<String> = fs::read_dir(root.join("images/lens0"))
        .map_err(|e| e.to_string())?
        .flatten()
        .filter(|entry| entry.path().is_file() && is_supported_colmap_image(&entry.path()))
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .filter(|name| lens1.join(name).is_file())
        .collect();
    names.sort();
    if names.len() < 2 {
        return Err("至少需要兩組同名的 lens0/lens1 影格才能對齊".to_owned());
    }
    let frame_motion = load_frame_motion_metadata(root);
    let mut pairs = BTreeSet::new();
    for (index, name) in names.iter().enumerate() {
        pairs.insert(format!("lens0/{name} lens1/{name}"));
        let source_motion = frame_motion.get(source_name_from_image(name));
        for (offset, neighbor) in names.iter().skip(index + 1).take(5).enumerate() {
            let offset = offset + 1;
            if source_name_from_image(name) != source_name_from_image(neighbor) {
                // Cross-source loop closure is intentionally left to the
                // anchor grid below; source-local motion cannot justify a
                // temporal pair across a discontinuous recording.
                continue;
            }
            // Same-sensor +1/+2 are the minimum temporal chain.  Cross-lens
            // +1 is also mandatory because the fisheye seam may be the only
            // visual bridge between the two tracks.  Longer links are kept
            // only when motion metadata says the temporal gap remains useful;
            // incomplete metadata falls back to the legacy +1..+5 graph.
            if offset <= 2 {
                pairs.insert(format!("lens0/{name} lens0/{neighbor}"));
                pairs.insert(format!("lens1/{name} lens1/{neighbor}"));
            } else {
                conditional_temporal_pair(&mut pairs, 0, 0, name, neighbor, offset, source_motion);
                conditional_temporal_pair(&mut pairs, 1, 1, name, neighbor, offset, source_motion);
            }
            conditional_temporal_pair(&mut pairs, 0, 1, name, neighbor, offset, source_motion);
            conditional_temporal_pair(&mut pairs, 1, 0, name, neighbor, offset, source_motion);
        }
    }
    // Temporal neighbors alone only connect adjacent capture filenames.  For
    // multiple OSV files, add a bounded cross-source anchor grid so recordings
    // that revisit the same space can actually form a shared reconstruction.
    let mut sources: BTreeMap<&str, Vec<&String>> = BTreeMap::new();
    for name in &names {
        let source = name.split_once('_').map_or("source", |(source, _)| source);
        sources.entry(source).or_default().push(name);
    }
    let groups: Vec<Vec<&String>> = sources
        .into_values()
        .map(|frames| {
            let step = (frames.len() / 20).max(1);
            frames.into_iter().step_by(step).take(20).collect()
        })
        .collect();
    for left_index in 0..groups.len() {
        for right_index in (left_index + 1)..groups.len() {
            for left in &groups[left_index] {
                for right in &groups[right_index] {
                    for left_lens in 0..2 {
                        for right_lens in 0..2 {
                            pairs
                                .insert(format!("lens{left_lens}/{left} lens{right_lens}/{right}"));
                        }
                    }
                }
            }
        }
    }
    fs::create_dir_all(root.join("metadata")).map_err(|e| e.to_string())?;
    fs::write(
        root.join("metadata/pairs.txt"),
        pairs.into_iter().collect::<Vec<_>>().join("\n") + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(names.len() as u64)
}

#[derive(Debug, Deserialize)]
struct RigBootstrapConfig {
    cameras: Vec<RigBootstrapCamera>,
}

#[derive(Debug, Deserialize)]
struct RigBootstrapCamera {
    image_prefix: String,
    #[serde(default)]
    ref_sensor: bool,
    cam_from_rig_rotation: Option<Vec<f64>>,
    cam_from_rig_translation: Option<Vec<f64>>,
}

impl RigBootstrapCamera {
    fn has_explicit_pose(&self) -> bool {
        self.cam_from_rig_rotation
            .as_ref()
            .is_some_and(|rotation| rotation.len() == 4)
            && self
                .cam_from_rig_translation
                .as_ref()
                .is_some_and(|translation| translation.len() == 3)
    }
}

fn rig_config_has_complete_sensor_poses(configs: &[RigBootstrapConfig]) -> bool {
    !configs.is_empty()
        && configs.iter().all(|config| {
            !config.cameras.is_empty()
                && config
                    .cameras
                    .iter()
                    .filter(|camera| camera.ref_sensor)
                    .count()
                    == 1
                && config
                    .cameras
                    .iter()
                    .all(|camera| camera.ref_sensor || camera.has_explicit_pose())
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RigMappingPlan {
    /// Configure the known rig before matching, then run exactly one mapper.
    PreconfiguredSinglePass,
    /// Reconstruct independent cameras, derive the rig, then map again.
    BootstrapThenFinal,
}

fn rig_mapping_plan(configs: &[RigBootstrapConfig]) -> RigMappingPlan {
    if rig_config_has_complete_sensor_poses(configs) {
        RigMappingPlan::PreconfiguredSinglePass
    } else {
        RigMappingPlan::BootstrapThenFinal
    }
}

/// Extract registered image names from COLMAP's text sparse-model format.
///
/// Image header lines contain nine scalar fields followed by the image name.
/// Point-observation lines cannot match one of the configured path prefixes,
/// which keeps this compatible with both the legacy and current text formats.
fn registered_rig_image_names(images_text: &str, prefixes: &BTreeSet<String>) -> BTreeSet<String> {
    fn remainder_after_fields(line: &str, fields_to_skip: usize) -> Option<&str> {
        let mut completed_fields = 0;
        let mut in_field = false;
        for (index, character) in line.char_indices() {
            if character.is_whitespace() {
                if in_field {
                    completed_fields += 1;
                    in_field = false;
                }
            } else if !in_field {
                if completed_fields == fields_to_skip {
                    return Some(line[index..].trim_end());
                }
                in_field = true;
            }
        }
        None
    }

    images_text
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .filter_map(|line| {
            let name = remainder_after_fields(line, 9)?.to_owned();
            prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix))
                .then_some(name)
        })
        .collect()
}

/// Verify rig camera coverage and the official precondition for deriving
/// unknown sensor-from-rig poses: every uncalibrated non-reference camera must
/// share at least one registered frame name with its rig's reference camera.
fn validate_rig_bootstrap_registration(
    configs: &[RigBootstrapConfig],
    registered_images: &BTreeSet<String>,
) -> Result<String, String> {
    if configs.is_empty() {
        return Err("rig_config.json 必須至少定義一組相機組".to_owned());
    }

    let mut summaries = Vec::new();
    let mut configured_prefixes = BTreeSet::new();
    for (rig_index, config) in configs.iter().enumerate() {
        if config.cameras.is_empty() {
            return Err(format!(
                "rig_config.json 的第 {} 組相機組沒有任何 camera",
                rig_index + 1
            ));
        }
        for camera in &config.cameras {
            if camera.image_prefix.is_empty() {
                return Err("rig_config.json 的 image_prefix 不得為空".to_owned());
            }
            if configured_prefixes.iter().any(|existing: &&str| {
                camera.image_prefix.starts_with(*existing)
                    || existing.starts_with(camera.image_prefix.as_str())
            }) {
                return Err(format!(
                    "rig_config.json 的 image_prefix 重複或互相重疊：{}",
                    camera.image_prefix
                ));
            }
            configured_prefixes.insert(camera.image_prefix.as_str());
            let has_rotation = camera.cam_from_rig_rotation.is_some();
            let has_translation = camera.cam_from_rig_translation.is_some();
            if has_rotation != has_translation {
                return Err(format!(
                    "rig_config.json 的 {} 必須同時提供 cam_from_rig_rotation 與 cam_from_rig_translation",
                    camera.image_prefix
                ));
            }
            if (has_rotation || has_translation) && !camera.has_explicit_pose() {
                return Err(format!(
                    "rig_config.json 的 {} 外參格式錯誤；rotation 必須為 4 個 WXYZ 值，translation 必須為 3 個 XYZ 值",
                    camera.image_prefix
                ));
            }
            if let (Some(rotation), Some(translation)) = (
                &camera.cam_from_rig_rotation,
                &camera.cam_from_rig_translation,
            ) {
                if !rotation
                    .iter()
                    .chain(translation.iter())
                    .all(|value| value.is_finite())
                    || rotation.iter().map(|value| value * value).sum::<f64>() <= f64::EPSILON
                {
                    return Err(format!(
                        "rig_config.json 的 {} 外參包含非有限值或零長度 quaternion",
                        camera.image_prefix
                    ));
                }
            }
            if camera.ref_sensor && (has_rotation || has_translation) {
                return Err(format!(
                    "rig_config.json 的參考鏡頭 {} 不得提供 cam_from_rig 外參",
                    camera.image_prefix
                ));
            }
        }
        let references = config
            .cameras
            .iter()
            .filter(|camera| camera.ref_sensor)
            .collect::<Vec<_>>();
        if references.len() != 1 {
            return Err(format!(
                "rig_config.json 的第 {} 組相機組必須恰好有一個 ref_sensor: true",
                rig_index + 1
            ));
        }
        for camera in &config.cameras {
            let registered_count = registered_images
                .iter()
                .filter(|name| name.starts_with(&camera.image_prefix))
                .count();
            if registered_count == 0 {
                return Err(format!(
                    "COLMAP 模型未註冊設定鏡頭 {} 的任何影像；無法確認相機組完整性。請改善該鏡頭的特徵／配對，或檢查 image_prefix",
                    camera.image_prefix
                ));
            }
        }
        let reference = references[0];
        let reference_frames = registered_images
            .iter()
            .filter_map(|name| name.strip_prefix(&reference.image_prefix))
            .collect::<BTreeSet<_>>();

        for camera in config
            .cameras
            .iter()
            .filter(|camera| !camera.ref_sensor && !camera.has_explicit_pose())
        {
            let camera_frames = registered_images
                .iter()
                .filter_map(|name| name.strip_prefix(&camera.image_prefix))
                .collect::<BTreeSet<_>>();
            let shared_frames = reference_frames.intersection(&camera_frames).count();
            if shared_frames == 0 {
                return Err(format!(
                    "COLMAP 模型無法估計相機組外參：參考鏡頭 {} 已註冊 {} 張、鏡頭 {} 已註冊 {} 張，但沒有任何同名影格同時註冊。請增加跨鏡頭重疊／配對品質，或提供已校正的 cam_from_rig_rotation 與 cam_from_rig_translation",
                    reference.image_prefix,
                    reference_frames.len(),
                    camera.image_prefix,
                    camera_frames.len()
                ));
            }
            summaries.push(format!(
                "{} ↔ {}：{} 組共同影格",
                reference.image_prefix, camera.image_prefix, shared_frames
            ));
        }
    }

    if summaries.is_empty() {
        Ok("相機組已提供完整外參".to_owned())
    } else {
        Ok(summaries.join("；"))
    }
}

fn validate_rigs_text_sensor_poses(
    rigs_text: &str,
    expected_sensor_counts: &[usize],
) -> Result<usize, String> {
    let mut calibrated_non_reference_sensors = 0;
    let mut rig_count = 0;
    let mut actual_sensor_counts = Vec::new();
    for line in rigs_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        rig_count += 1;
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 4 {
            return Err("COLMAP rigs.txt 欄位不足".to_owned());
        }
        let rig_id = fields[0];
        let num_sensors = fields[1]
            .parse::<usize>()
            .map_err(|_| format!("COLMAP rig {rig_id} 的 sensor 數量無效"))?;
        if num_sensors == 0 {
            return Err(format!("COLMAP rig {rig_id} 沒有任何 sensor"));
        }
        actual_sensor_counts.push(num_sensors);
        let mut index = 4;
        for _ in 1..num_sensors {
            if fields.len() < index + 3 {
                return Err(format!(
                    "COLMAP rig {rig_id} 的 non-reference sensor 欄位不足"
                ));
            }
            let sensor_type = fields[index];
            let sensor_id = fields[index + 1];
            let has_pose = fields[index + 2];
            index += 3;
            match has_pose {
                "0" => {
                    return Err(format!(
                        "COLMAP rig {rig_id} 的 {sensor_type} sensor {sensor_id} 缺少 sensor_from_rig"
                    ));
                }
                "1" => {
                    if fields.len() < index + 7 {
                        return Err(format!(
                            "COLMAP rig {rig_id} 的 {sensor_type} sensor {sensor_id} 外參欄位不足"
                        ));
                    }
                    let pose = fields[index..index + 7]
                        .iter()
                        .map(|value| {
                            let parsed = value.parse::<f64>().map_err(|_| {
                                format!(
                                    "COLMAP rig {rig_id} 的 {sensor_type} sensor {sensor_id} 外參不是有效數值"
                                )
                            })?;
                            if !parsed.is_finite() {
                                return Err(format!(
                                    "COLMAP rig {rig_id} 的 {sensor_type} sensor {sensor_id} 外參不是有限數值"
                                ));
                            }
                            Ok(parsed)
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let quaternion_norm = pose[..4]
                        .iter()
                        .map(|value| value * value)
                        .sum::<f64>()
                        .sqrt();
                    if quaternion_norm <= f64::EPSILON || (quaternion_norm - 1.0).abs() > 1e-3 {
                        return Err(format!(
                            "COLMAP rig {rig_id} 的 {sensor_type} sensor {sensor_id} quaternion 未正規化或為零"
                        ));
                    }
                    index += 7;
                    calibrated_non_reference_sensors += 1;
                }
                value => {
                    return Err(format!("COLMAP rig {rig_id} 的 HAS_POSE 值無效：{value}"));
                }
            }
        }
        if index != fields.len() {
            return Err(format!("COLMAP rig {rig_id} 含有無法辨識的額外欄位"));
        }
    }
    if rig_count == 0 {
        return Err("COLMAP rig_configurator 未輸出任何 rig".to_owned());
    }
    let mut expected_sensor_counts = expected_sensor_counts.to_vec();
    expected_sensor_counts.sort_unstable();
    actual_sensor_counts.sort_unstable();
    if actual_sensor_counts != expected_sensor_counts {
        return Err(format!(
            "COLMAP rig_configurator 輸出的相機組結構不完整：預期每組 sensor 數量為 {expected_sensor_counts:?}，實際為 {actual_sensor_counts:?}"
        ));
    }
    Ok(calibrated_non_reference_sensors)
}

fn canonical_json(value: &Value) -> Result<String, String> {
    fn write_value(value: &Value, output: &mut String) -> Result<(), String> {
        match value {
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => output.push_str(&value.to_string()),
            Value::String(value) => {
                output.push_str(&serde_json::to_string(value).map_err(|error| error.to_string())?)
            }
            Value::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        output.push(',');
                    }
                    write_value(value, output)?;
                }
                output.push(']');
            }
            Value::Object(values) => {
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                output.push('{');
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        output.push(',');
                    }
                    output
                        .push_str(&serde_json::to_string(key).map_err(|error| error.to_string())?);
                    output.push(':');
                    write_value(&values[key], output)?;
                }
                output.push('}');
            }
        }
        Ok(())
    }

    let mut output = String::new();
    write_value(value, &mut output)?;
    Ok(output)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|error| format!("無法讀取 {}：{error}", path.display()))
}

fn optional_file_sha256(path: &Path) -> Result<String, String> {
    if path.is_file() {
        file_sha256(path)
    } else {
        Ok(sha256_hex(&[]))
    }
}

fn frame_motion_metadata_sha256(root: &Path) -> Result<String, String> {
    let metadata = root.join("metadata");
    let Ok(entries) = fs::read_dir(&metadata) else {
        return Ok(sha256_hex(&[]));
    };
    let mut paths = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with("_frame_motion.json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut bytes = Vec::new();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("影格運動 metadata 檔名無效：{}", path.display()))?;
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(
            &fs::read(&path).map_err(|error| format!("無法讀取 {}：{error}", path.display()))?,
        );
        bytes.push(0);
    }
    Ok(sha256_hex(&bytes))
}

fn collect_align_file_identities(
    directory: &Path,
    project_root: &Path,
    identities: &mut Vec<AlignFileIdentity>,
) -> Result<(), String> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("無法讀取 {}：{error}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("無法列舉 {}：{error}", directory.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("無法讀取 {} 的檔案類型：{error}", path.display()))?;
        if file_type.is_dir() {
            collect_align_file_identities(&path, project_root, identities)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|error| format!("無法讀取 {} 的 metadata：{error}", path.display()))?;
        let relative = path
            .strip_prefix(project_root)
            .map_err(|error| format!("無法建立 {} 的相對路徑：{error}", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        identities.push(AlignFileIdentity {
            path: relative,
            size: metadata.len(),
            modified_nanos: source_modified_nanos(&metadata),
        });
    }
    Ok(())
}

fn build_align_fingerprint(
    root: &Path,
    settings: &Value,
    colmap_version: &str,
    include_masks: bool,
) -> Result<String, String> {
    let mut files = Vec::new();
    collect_align_file_identities(&root.join("images"), root, &mut files)?;
    if include_masks {
        collect_align_file_identities(&root.join("masks_colmap"), root, &mut files)?;
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    let payload = AlignFingerprintPayload {
        schema_version: ALIGN_CHECKPOINT_SCHEMA_VERSION,
        pipeline_revision: ALIGN_PIPELINE_REVISION,
        settings: canonical_json(settings)?,
        colmap_version: colmap_version.to_owned(),
        include_masks,
        rig_config_sha256: file_sha256(&root.join("rig_config.json"))?,
        pairs_sha256: file_sha256(&root.join("metadata/pairs.txt"))?,
        frame_motion_sha256: frame_motion_metadata_sha256(root)?,
        global_mapper_priors_sha256: optional_file_sha256(
            &root.join("metadata/global_mapper_priors.json"),
        )?,
        files,
    };
    let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn build_feature_fingerprint(
    root: &Path,
    colmap_version: &str,
    include_masks: bool,
) -> Result<String, String> {
    let mut files = Vec::new();
    collect_align_file_identities(&root.join("images"), root, &mut files)?;
    if include_masks {
        collect_align_file_identities(&root.join("masks_colmap"), root, &mut files)?;
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    let payload = FeatureFingerprintPayload {
        schema_version: FEATURE_FINGERPRINT_SCHEMA_VERSION,
        colmap_version: colmap_version.to_owned(),
        extractor_type: FEATURE_EXTRACTION_TYPE,
        camera_model: FEATURE_CAMERA_MODEL,
        default_focal_length_factor: FEATURE_DEFAULT_FOCAL_LENGTH_FACTOR,
        include_masks,
        files,
    };
    let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn load_align_checkpoint(path: &Path) -> Option<AlignCheckpoint> {
    let checkpoint = serde_json::from_slice::<AlignCheckpoint>(&fs::read(path).ok()?).ok()?;
    (checkpoint.schema_version == ALIGN_CHECKPOINT_SCHEMA_VERSION).then_some(checkpoint)
}

fn write_align_checkpoint(
    path: &Path,
    fingerprint: &str,
    feature_fingerprint: &str,
    completed: bool,
) -> Result<(), String> {
    let checkpoint = AlignCheckpoint {
        schema_version: ALIGN_CHECKPOINT_SCHEMA_VERSION,
        fingerprint: fingerprint.to_owned(),
        feature_fingerprint: Some(feature_fingerprint.to_owned()),
        completed,
    };
    let bytes = serde_json::to_vec_pretty(&checkpoint).map_err(|error| error.to_string())?;
    let parent = path
        .parent()
        .ok_or_else(|| "對齊 checkpoint 缺少父資料夾".to_owned())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("無法建立對齊 checkpoint 資料夾：{error}"))?;
    let partial = path.with_extension("json.partial");
    fs::write(&partial, bytes).map_err(|error| format!("無法寫入對齊 checkpoint：{error}"))?;
    if path.exists() {
        fs::remove_file(path).map_err(|error| format!("無法替換舊對齊 checkpoint：{error}"))?;
    }
    fs::rename(&partial, path).map_err(|error| format!("無法啟用對齊 checkpoint：{error}"))
}

fn remove_align_artifact(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
    .map_err(|error| format!("無法移除 COLMAP 對齊輸出 {}：{error}", path.display()))
}

fn colmap_database_artifacts(database: &Path) -> Result<[PathBuf; 4], String> {
    let file_name = database
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("COLMAP 資料庫路徑無效：{}", database.display()))?;
    Ok([
        database.to_owned(),
        database.with_file_name(format!("{file_name}-wal")),
        database.with_file_name(format!("{file_name}-shm")),
        database.with_file_name(format!("{file_name}-journal")),
    ])
}

fn remove_colmap_database_artifacts(database: &Path) -> Result<(), String> {
    for artifact in colmap_database_artifacts(database)? {
        remove_align_artifact(&artifact)?;
    }
    Ok(())
}

fn create_colmap_database_backup(database: &Path, backup: &Path) -> Result<(), String> {
    let backup_name = backup
        .file_name()
        .ok_or_else(|| format!("COLMAP 資料庫備份路徑無效：{}", backup.display()))?;
    let partial = backup.with_file_name(format!("{}.partial", backup_name.to_string_lossy()));
    remove_align_artifact(&partial)?;
    fs::create_dir_all(&partial)
        .map_err(|error| format!("無法建立 COLMAP 資料庫備份資料夾：{error}"))?;
    let artifacts = colmap_database_artifacts(database)?;
    let copy_result = (|| {
        // The shared-memory file is transient and SQLite recreates it from WAL.
        // Preserve the main database plus either durable recovery log format.
        for source in [&artifacts[0], &artifacts[1], &artifacts[3]] {
            if source.is_file() {
                let file_name = source
                    .file_name()
                    .ok_or_else(|| format!("COLMAP 資料庫備份來源無檔名：{}", source.display()))?;
                fs::copy(source, partial.join(file_name)).map_err(|error| {
                    format!("無法備份 COLMAP 資料庫 {}：{error}", source.display())
                })?;
            }
        }
        if !partial
            .join(database.file_name().unwrap_or_default())
            .is_file()
        {
            return Err(format!("COLMAP 資料庫不存在：{}", database.display()));
        }
        Ok(())
    })();
    if let Err(error) = copy_result {
        let cleanup = remove_align_artifact(&partial)
            .err()
            .map(|value| format!("；{value}"))
            .unwrap_or_default();
        return Err(format!("{error}{cleanup}"));
    }
    remove_align_artifact(backup)?;
    fs::rename(&partial, backup).map_err(|error| format!("無法啟用 COLMAP 資料庫備份：{error}"))?;
    Ok(())
}

fn restore_colmap_database_backup(database: &Path, backup: &Path) -> Result<(), String> {
    let database_name = database
        .file_name()
        .ok_or_else(|| format!("COLMAP 資料庫路徑無效：{}", database.display()))?;
    let backup_database = backup.join(database_name);
    if !backup_database.is_file() {
        return Err(format!(
            "COLMAP 資料庫備份不完整，缺少 {}",
            backup_database.display()
        ));
    }
    remove_colmap_database_artifacts(database)?;
    fs::copy(&backup_database, database)
        .map_err(|error| format!("無法還原 COLMAP 資料庫備份 {}：{error}", database.display()))?;
    for suffix in ["-wal", "-journal"] {
        let recovery_name = format!("{}{suffix}", database_name.to_string_lossy());
        let backup_recovery_log = backup.join(&recovery_name);
        if backup_recovery_log.is_file() {
            fs::copy(
                &backup_recovery_log,
                database.with_file_name(&recovery_name),
            )
            .map_err(|error| {
                format!(
                    "無法還原 COLMAP recovery log 備份 {}：{error}",
                    backup_recovery_log.display()
                )
            })?;
        }
    }
    Ok(())
}

fn cleanup_align_artifacts(root: &Path, preserve_database: bool) -> Result<(), String> {
    let mut paths = vec![
        root.join("sparse"),
        root.join("sparse_bootstrap"),
        root.join("metadata/.align-bootstrap-text"),
        root.join("metadata/.align-configured-rig"),
        root.join("metadata/.align-configured-rig-text"),
        root.join("metadata/.align-matching-database.backup"),
        root.join("metadata/.align-matching-database.backup.partial"),
        root.join("metadata/.align-unconfigured-database.backup"),
        root.join("metadata/.align-unconfigured-database.backup.partial"),
    ];
    if !preserve_database {
        paths.extend([
            root.join("database.db"),
            root.join("database.db-wal"),
            root.join("database.db-shm"),
            root.join("database.db-journal"),
        ]);
    }
    for path in paths {
        remove_align_artifact(&path)?;
    }
    Ok(())
}

fn validate_colmap_bootstrap_for_rig(
    app: &AppHandle,
    id: &str,
    colmap: &Path,
    root: &Path,
    bootstrap_model: &Path,
    control: &JobControl,
) -> Result<(), String> {
    let config_path = root.join("rig_config.json");
    let configs = serde_json::from_slice::<Vec<RigBootstrapConfig>>(
        &fs::read(&config_path)
            .map_err(|error| format!("無法讀取 {}：{error}", config_path.display()))?,
    )
    .map_err(|error| format!("rig_config.json 格式無效：{error}"))?;
    let prefixes = configs
        .iter()
        .flat_map(|config| config.cameras.iter())
        .map(|camera| camera.image_prefix.clone())
        .collect::<BTreeSet<_>>();

    let text_model = root.join("metadata/.align-bootstrap-text");
    remove_align_artifact(&text_model)?;
    fs::create_dir_all(&text_model)
        .map_err(|error| format!("無法建立 COLMAP 初始模型驗證資料夾：{error}"))?;
    let result = run_child(
        app,
        id,
        colmap,
        &[
            "model_converter".into(),
            "--input_path".into(),
            bootstrap_model.to_string_lossy().into_owned(),
            "--output_path".into(),
            text_model.to_string_lossy().into_owned(),
            "--output_type".into(),
            "TXT".into(),
        ],
        control,
    )
    .and_then(|()| {
        let images_path = text_model.join("images.txt");
        let images_text = fs::read_to_string(&images_path)
            .map_err(|error| format!("無法讀取 {}：{error}", images_path.display()))?;
        let registered_images = registered_rig_image_names(&images_text, &prefixes);
        validate_rig_bootstrap_registration(&configs, &registered_images)
    });
    let cleanup = remove_align_artifact(&text_model);
    match (result, cleanup) {
        (Ok(summary), Ok(())) => {
            emit_log(
                app,
                id,
                "info",
                format!("已驗證初始模型可推算相機組外參：{summary}"),
            );
            Ok(())
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(error), Err(cleanup_error)) => Err(format!("{error}；{cleanup_error}")),
    }
}

fn validate_colmap_configured_rig_model(
    app: &AppHandle,
    id: &str,
    colmap: &Path,
    root: &Path,
    configured_model: &Path,
    control: &JobControl,
) -> Result<usize, String> {
    let config_path = root.join("rig_config.json");
    let configs = serde_json::from_slice::<Vec<RigBootstrapConfig>>(
        &fs::read(&config_path)
            .map_err(|error| format!("無法讀取 {}：{error}", config_path.display()))?,
    )
    .map_err(|error| format!("rig_config.json 格式無效：{error}"))?;
    let expected_sensor_counts = configs
        .iter()
        .map(|config| config.cameras.len())
        .collect::<Vec<_>>();
    let prefixes = configs
        .iter()
        .flat_map(|config| config.cameras.iter())
        .map(|camera| camera.image_prefix.clone())
        .collect::<BTreeSet<_>>();
    let text_model = root.join("metadata/.align-configured-rig-text");
    remove_align_artifact(&text_model)?;
    fs::create_dir_all(&text_model)
        .map_err(|error| format!("無法建立 COLMAP 相機組驗證資料夾：{error}"))?;
    let result = run_child(
        app,
        id,
        colmap,
        &[
            "model_converter".into(),
            "--input_path".into(),
            configured_model.to_string_lossy().into_owned(),
            "--output_path".into(),
            text_model.to_string_lossy().into_owned(),
            "--output_type".into(),
            "TXT".into(),
        ],
        control,
    )
    .and_then(|()| {
        let rigs_path = text_model.join("rigs.txt");
        let rigs_text = fs::read_to_string(&rigs_path)
            .map_err(|error| format!("無法讀取 {}：{error}", rigs_path.display()))?;
        let calibrated_sensor_count =
            validate_rigs_text_sensor_poses(&rigs_text, &expected_sensor_counts)?;
        let images_path = text_model.join("images.txt");
        let images_text = fs::read_to_string(&images_path)
            .map_err(|error| format!("無法讀取 {}：{error}", images_path.display()))?;
        let registered_images = registered_rig_image_names(&images_text, &prefixes);
        validate_rig_bootstrap_registration(&configs, &registered_images)?;
        Ok(calibrated_sensor_count)
    });
    let cleanup = remove_align_artifact(&text_model);
    match (result, cleanup) {
        (Ok(count), Ok(())) => Ok(count),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(error), Err(cleanup_error)) => Err(format!("{error}；{cleanup_error}")),
    }
}

fn is_rig_pose_derivation_failure_line(line: &str) -> bool {
    let normalized = line.to_ascii_lowercase();
    normalized.contains("failed to derive sensor_from_rig")
        || normalized.contains("unknown sensor_from_rig")
}

fn can_reuse_align_result(
    final_complete: bool,
    checkpoint_matches: bool,
    checkpoint_completed: bool,
) -> bool {
    final_complete && checkpoint_matches && checkpoint_completed
}

fn can_reuse_feature_database(
    force_rebuild: bool,
    feature_checkpoint_matches: bool,
    legacy_align_checkpoint_matches: bool,
    feature_cache_valid: bool,
) -> bool {
    !force_rebuild
        && (feature_checkpoint_matches || legacy_align_checkpoint_matches)
        && feature_cache_valid
}

fn colmap_database_header_valid(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let Ok(metadata) = file.metadata() else {
        return false;
    };
    let mut header = [0_u8; 100];
    if file.read_exact(&mut header).is_err() || header[..16] != *b"SQLite format 3\0" {
        return false;
    }
    let encoded_page_size = u16::from_be_bytes([header[16], header[17]]);
    let page_size = if encoded_page_size == 1 {
        65_536_u64
    } else {
        u64::from(encoded_page_size)
    };
    page_size.is_power_of_two()
        && (512..=65_536).contains(&page_size)
        && metadata.len() >= page_size
}

fn sparse_model_exists(root: &Path) -> bool {
    fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            let model = entry.path();
            model.is_dir()
                && ["cameras", "images", "points3D"].iter().all(|name| {
                    model.join(format!("{name}.bin")).is_file()
                        || model.join(format!("{name}.txt")).is_file()
                })
        })
}

fn sparse_rig_model_exists(root: &Path) -> bool {
    fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            let model = entry.path();
            model.is_dir()
                && ["rigs", "cameras", "frames", "images", "points3D"]
                    .iter()
                    .all(|name| {
                        ["bin", "txt"].iter().any(|extension| {
                            fs::metadata(model.join(format!("{name}.{extension}")))
                                .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
                        })
                    })
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColmapFraction {
    current: u64,
    total: u64,
}

fn parse_fraction(value: &str) -> Option<ColmapFraction> {
    let slash = value.find('/')?;
    let current = value[..slash].trim().parse::<u64>().ok()?;
    let total_digits = value[slash + 1..]
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    let total = total_digits.parse::<u64>().ok()?;
    (current > 0 && total > 0 && current <= total).then_some(ColmapFraction { current, total })
}

fn parse_fraction_after(line: &str, marker: &str) -> Option<ColmapFraction> {
    let body = line.split_once(marker)?.1;
    let bracket = body.split_once(']').map_or(body, |(value, _)| value);
    parse_fraction(bracket)
}

fn parse_feature_progress(line: &str) -> Option<ColmapFraction> {
    [
        "Processed file [",
        "Processed image [",
        "Processing file [",
        "Processing image [",
    ]
    .into_iter()
    .find_map(|marker| parse_fraction_after(line, marker))
}

fn parse_feature_name(line: &str) -> Option<String> {
    let value = line.split_once("Name:")?.1.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn parse_matching_progress(line: &str) -> Option<ColmapFraction> {
    let body = [
        "Processing block [",
        "Matching block [",
        "Processing image [",
        "Matching image [",
    ]
    .into_iter()
    .find_map(|marker| line.split_once(marker).map(|(_, body)| body))?;
    let bracket = body.split_once(']').map_or(body, |(value, _)| value);
    let fractions = bracket
        .split(',')
        .map(parse_fraction)
        .collect::<Option<Vec<_>>>()?;
    match fractions.as_slice() {
        [fraction] => Some(*fraction),
        [row, column] => {
            let total = row.total.checked_mul(column.total)?;
            let current = row
                .current
                .saturating_sub(1)
                .checked_mul(column.total)?
                .checked_add(column.current)?;
            Some(ColmapFraction {
                current: current.min(total),
                total,
            })
        }
        _ => None,
    }
}

fn parse_mapper_registration(line: &str) -> Option<(u64, u64)> {
    let body = line.split_once("Registering image #")?.1;
    let image_id = body
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse::<u64>()
        .ok()?;
    let registered = body
        .split_once("num_reg_frames=")?
        .1
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse::<u64>()
        .ok()?;
    Some((image_id, registered))
}

fn colmap_step_progress(step: u8, fraction: f32) -> f32 {
    (step as f32 + fraction.clamp(0.0, 0.99)) / 5.0
}

#[allow(clippy::too_many_arguments)]
fn emit_colmap_progress(
    app: &AppHandle,
    id: &str,
    phase: &str,
    step: u8,
    fraction: f32,
    message: impl Into<String>,
    current_item: Option<String>,
) {
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        phase,
        colmap_step_progress(step, fraction),
        message,
        "running",
        false,
        None,
        None,
        current_item,
        None,
    );
}

fn emit_colmap_step_completed(
    app: &AppHandle,
    id: &str,
    phase: &str,
    step: u8,
    message: &str,
    current_item: &str,
) {
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        phase,
        (step as f32 + 1.0) / 5.0,
        message,
        "running",
        false,
        None,
        None,
        Some(current_item.to_owned()),
        None,
    );
}

fn feature_extractor_args(
    root: &Path,
    db: &Path,
    use_gpu: bool,
    gpu_index: &str,
    use_masks: bool,
) -> Vec<String> {
    let mut args = vec![
        "feature_extractor".into(),
        "--database_path".into(),
        db.to_string_lossy().into_owned(),
        "--image_path".into(),
        root.join("images").to_string_lossy().into_owned(),
        "--ImageReader.single_camera_per_folder".into(),
        "1".into(),
        "--ImageReader.camera_model".into(),
        FEATURE_CAMERA_MODEL.into(),
        // COLMAP otherwise initializes an EXIF-less 3840 px fisheye at its
        // perspective-camera default of 4608 px (factor 1.2).  An equidistant
        // 180–190° circular fisheye starts near width / pi, so 0.3 gives the
        // mapper a physically plausible basin while distortion remains free.
        "--ImageReader.default_focal_length_factor".into(),
        FEATURE_DEFAULT_FOCAL_LENGTH_FACTOR.to_string(),
        "--FeatureExtraction.type".into(),
        FEATURE_EXTRACTION_TYPE.into(),
        "--SiftExtraction.max_num_features".into(),
        "8192".into(),
        "--FeatureExtraction.use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--FeatureExtraction.gpu_index".into(),
        gpu_index.to_owned(),
    ];
    if use_masks {
        args.push("--ImageReader.mask_path".into());
        args.push(root.join("masks_colmap").to_string_lossy().into_owned());
    }
    args
}

fn matches_importer_args(root: &Path, db: &Path, use_gpu: bool, gpu_index: &str) -> Vec<String> {
    vec![
        "matches_importer".into(),
        "--database_path".into(),
        db.to_string_lossy().into_owned(),
        "--match_list_path".into(),
        root.join("metadata/pairs.txt")
            .to_string_lossy()
            .into_owned(),
        "--match_type".into(),
        "pairs".into(),
        "--FeatureMatching.use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--FeatureMatching.gpu_index".into(),
        gpu_index.to_owned(),
        // SIFT extraction produces at most 8192 features per image by default.
        // Keep the matcher at that same bound instead of reserving COLMAP's
        // much larger default workspace for descriptors that cannot exist.
        "--FeatureMatching.max_num_matches".into(),
        "8192".into(),
    ]
}

fn mapper_gpu_index(gpu_index: &str) -> &str {
    gpu_index.split(',').next().unwrap_or("-1")
}

fn mapper_args(
    db: &Path,
    images: &Path,
    output: &Path,
    use_gpu: bool,
    gpu_index: &str,
    disable_sensor_refinement: bool,
    reduce_global_ba_frequency: bool,
) -> Vec<String> {
    let mut args = vec![
        "mapper".into(),
        "--database_path".into(),
        db.to_string_lossy().into_owned(),
        "--image_path".into(),
        images.to_string_lossy().into_owned(),
        "--output_path".into(),
        output.to_string_lossy().into_owned(),
        "--Mapper.multiple_models".into(),
        "0".into(),
        "--Mapper.ba_local_backend".into(),
        "CERES".into(),
        "--Mapper.ba_global_backend".into(),
        "CERES".into(),
        "--Mapper.ba_use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--Mapper.ba_gpu_index".into(),
        gpu_index.to_owned(),
    ];
    if reduce_global_ba_frequency {
        // Use COLMAP's documented redundant-landmark pruning and its 1.4
        // video growth-ratio preset to reduce repeated global BA passes.
        // Keep unknown-rig bootstrap on conservative COLMAP defaults because
        // that first pass exists to maximize registration/calibration coverage.
        args.extend([
            "--Mapper.ba_global_ignore_redundant_points3D".into(),
            "1".into(),
            "--Mapper.ba_global_frames_ratio".into(),
            "1.4".into(),
            "--Mapper.ba_global_points_ratio".into(),
            "1.4".into(),
        ]);
    }
    if disable_sensor_refinement {
        args.push("--Mapper.ba_refine_sensor_from_rig".into());
        args.push("0".into());
    }
    args
}

fn global_mapper_args(
    db: &Path,
    images: &Path,
    output: &Path,
    use_gpu: bool,
    gpu_index: &str,
    use_gravity_prior: bool,
) -> Vec<String> {
    vec![
        "global_mapper".into(),
        "--database_path".into(),
        db.to_string_lossy().into_owned(),
        "--image_path".into(),
        images.to_string_lossy().into_owned(),
        "--output_path".into(),
        output.to_string_lossy().into_owned(),
        // Known rig extrinsics are a prerequisite for gravity-aligned rig
        // solving. Keep them fixed unless a future calibrated workflow opts
        // into refinement explicitly.
        "--GlobalMapper.refine_sensor_from_rig".into(),
        "0".into(),
        "--GlobalMapper.ra_use_gravity".into(),
        if use_gravity_prior {
            "1".into()
        } else {
            "0".into()
        },
        "--GlobalMapper.ra_use_stratified".into(),
        "1".into(),
        "--GlobalMapper.gp_use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--GlobalMapper.gp_gpu_index".into(),
        mapper_gpu_index(gpu_index).to_owned(),
        "--GlobalMapper.ba_ceres_use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--GlobalMapper.ba_ceres_gpu_index".into(),
        mapper_gpu_index(gpu_index).to_owned(),
    ]
}

const MIN_GLOBAL_GRAVITY_COVERAGE_RATIO: f64 = 0.8;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GlobalMapperPriorMetadata {
    schema_version: u32,
    focal_prior_valid: bool,
    #[serde(default)]
    gravity_prior_valid: bool,
    #[serde(default)]
    gravity_coverage_ratio: Option<f64>,
    #[serde(default)]
    sensor_to_camera_calibration_version: Option<String>,
    #[serde(default)]
    time_offset_ms: Option<f64>,
    #[serde(default)]
    database_pose_priors_injected: bool,
}

fn has_valid_global_mapper_priors(root: &Path, require_gravity: bool) -> bool {
    // The current extractor deliberately does not inject COLMAP focal/gravity
    // priors. Treat an explicit, validated marker as the opt-in contract for
    // a future calibration stage instead of guessing from default focal 0.3.
    let path = root.join("metadata/global_mapper_priors.json");
    let Ok(metadata) =
        serde_json::from_slice::<GlobalMapperPriorMetadata>(&fs::read(path).unwrap_or_default())
    else {
        return false;
    };
    metadata.schema_version >= 1
        && metadata.focal_prior_valid
        && (!require_gravity
            || (metadata.gravity_prior_valid
                && metadata.database_pose_priors_injected
                && metadata.gravity_coverage_ratio.is_some_and(|coverage| {
                    coverage.is_finite()
                        && (MIN_GLOBAL_GRAVITY_COVERAGE_RATIO..=1.0).contains(&coverage)
                })
                && metadata
                    .sensor_to_camera_calibration_version
                    .as_deref()
                    .is_some_and(|version| !version.trim().is_empty())
                && metadata.time_offset_ms.is_some_and(f64::is_finite)))
}

fn global_mapper_prerequisite_error(
    root: &Path,
    rig_preconfigured: bool,
    capabilities: &crate::doctor::ColmapCapabilities,
    use_gravity_prior: bool,
) -> Option<String> {
    if !capabilities.global_mapper {
        return Some("指定的 COLMAP 不提供 global_mapper command".to_owned());
    }
    if !rig_preconfigured {
        return Some("global_mapper 需要已知且完整的 rig sensor_from_rig 外參".to_owned());
    }
    if use_gravity_prior && !capabilities.global_mapper_gravity {
        return Some("指定的 global_mapper 不接受 gravity rotation averaging 相關選項".to_owned());
    }
    if !capabilities.global_mapper_gravity {
        return Some(
            "指定的 global_mapper 不提供 ra_use_gravity/ra_use_stratified 選項，無法安全建立固定 CLI"
                .to_owned(),
        );
    }
    if !capabilities.global_mapper_gp_gpu || !capabilities.global_mapper_ba_gpu {
        return Some(
            "指定的 global_mapper 缺少 gp_use_gpu/gp_gpu_index 或 ba_ceres_use_gpu/ba_ceres_gpu_index 選項"
                .to_owned(),
        );
    }
    if !has_valid_global_mapper_priors(root, use_gravity_prior) {
        return Some(
            "尚未提供已驗證的 focal prior（gravity 模式另需 database pose prior、至少 80% coverage、時間偏移與 sensor-to-camera 校正版本；metadata/global_mapper_priors.json）；global mapper 拒絕執行以避免錯誤重建"
                .to_owned(),
        );
    }
    None
}

fn run_align(
    app: &AppHandle,
    id: &str,
    manifest: &ProjectManifest,
    custom_colmap_path: Option<&str>,
    force_rebuild: bool,
    control: &JobControl,
) -> Result<Vec<String>, String> {
    let colmap = crate::doctor::resolve_colmap(custom_colmap_path)?;
    let root = PathBuf::from(&manifest.output_path);
    let gpu_index = parse_gpu_index(&manifest.settings)?;
    let requested_gpu = setting_bool(&manifest.settings, "/align/useGpu", false);
    let requested_mapper_mode = mapper_mode(&manifest.settings)?;
    let use_gravity_prior = setting_bool(&manifest.settings, "/align/useGravityPrior", false);
    let rig_frame_count = write_rig_and_pairs(&root)?;
    let pairs_path = root.join("metadata/pairs.txt");
    let pair_count = fs::read_to_string(&pairs_path)
        .map_err(|error| format!("無法讀取 {}：{error}", pairs_path.display()))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    emit_log(
        app,
        id,
        "info",
        format!("已為 {rig_frame_count} 組 rig frames 建立 {pair_count} 組 COLMAP matching pairs"),
    );
    let (independent_image_total, rig_frame_total) = dual_fisheye_registration_totals(&root)?;
    let rig_config_path = root.join("rig_config.json");
    let rig_configs = serde_json::from_slice::<Vec<RigBootstrapConfig>>(
        &fs::read(&rig_config_path)
            .map_err(|error| format!("無法讀取 {}：{error}", rig_config_path.display()))?,
    )
    .map_err(|error| format!("rig_config.json 格式無效：{error}"))?;
    let rig_mapping_plan = rig_mapping_plan(&rig_configs);
    let rig_preconfigured = rig_mapping_plan == RigMappingPlan::PreconfiguredSinglePass;
    let db = root.join("database.db");
    let sparse = root.join("sparse");
    let unconfigured_database_backup = root.join("metadata/.align-unconfigured-database.backup");
    let use_masks = mask_enabled(&manifest.settings)
        && matches!(
            manifest.stage(&StageName::Mask).status,
            StageStatus::Completed
        );
    let checkpoint_path = root.join("metadata/align.checkpoint.json");
    let colmap_version = crate::doctor::command_version(&colmap)
        .ok_or_else(|| "無法讀取 COLMAP 版本；Align 需要 COLMAP 4.1.1 以上".to_owned())?;
    if !crate::doctor::colmap_version_at_least_4_1_1(&colmap_version) {
        return Err(format!(
            "不支援的 COLMAP 版本：{colmap_version}；Align 需要 COLMAP 4.1.1 以上"
        ));
    }
    let colmap_capabilities = crate::doctor::probe_colmap_capabilities(&colmap);
    if !colmap_capabilities.feature_extractor
        || !colmap_capabilities.mapper
        || !colmap_capabilities.model_converter
        || !colmap_capabilities.rig_configurator
        || !colmap_capabilities.matches_importer
    {
        return Err(
            "指定的 COLMAP 4.1.1+ 缺少 feature_extractor、matches_importer、mapper、model_converter 或 rig_configurator，無法執行雙鏡頭 rig 重建"
                .to_owned(),
        );
    }
    let mapper_mode = match requested_mapper_mode {
        MapperMode::Incremental => MapperMode::Incremental,
        MapperMode::Global => {
            if let Some(error) = global_mapper_prerequisite_error(
                &root,
                rig_preconfigured,
                &colmap_capabilities,
                use_gravity_prior,
            ) {
                return Err(format!(
                    "align.mapperMode=global 的前提不成立：{error}；請改用 incremental 或 auto"
                ));
            }
            MapperMode::Global
        }
        MapperMode::Auto => {
            if let Some(error) = global_mapper_prerequisite_error(
                &root,
                rig_preconfigured,
                &colmap_capabilities,
                use_gravity_prior,
            ) {
                emit_log(
                    app,
                    id,
                    "info",
                    format!("mapperMode=auto：{error}，退回 incremental mapper"),
                );
                MapperMode::Incremental
            } else {
                emit_log(
                    app,
                    id,
                    "info",
                    "mapperMode=auto：global mapper 前提已驗證，使用 global_mapper",
                );
                MapperMode::Global
            }
        }
    };
    let fingerprint =
        build_align_fingerprint(&root, &manifest.settings, &colmap_version, use_masks)?;
    let feature_fingerprint = build_feature_fingerprint(&root, &colmap_version, use_masks)?;
    let checkpoint_present = checkpoint_path.exists();
    let checkpoint = load_align_checkpoint(&checkpoint_path);
    let checkpoint_matches = !force_rebuild
        && checkpoint
            .as_ref()
            .is_some_and(|value| value.fingerprint == fingerprint);
    let feature_checkpoint_matches = !force_rebuild
        && checkpoint.as_ref().is_some_and(|value| {
            value.feature_fingerprint.as_deref() == Some(feature_fingerprint.as_str())
        });
    // Schema v2 checkpoints created before featureFingerprint used the same
    // SIFT/OPENCV_FISHEYE/0.3 extraction semantics. Migrate them once when the
    // broader fingerprint still matches. If those semantics change, the align
    // checkpoint schema must be bumped rather than extending this fallback.
    let legacy_feature_checkpoint_matches = !force_rebuild
        && checkpoint_matches
        && checkpoint
            .as_ref()
            .is_some_and(|value| value.feature_fingerprint.is_none());
    let checkpoint_completed =
        !force_rebuild && checkpoint.as_ref().is_some_and(|value| value.completed);
    let feature_cache = inspect_feature_cache(&root.join("images"), &db);
    let feature_cache_complete = feature_cache
        .as_ref()
        .is_ok_and(|report| report.is_complete());
    let feature_cache_counts = feature_cache
        .as_ref()
        .ok()
        .map(|report| (report.expected, report.completed));
    let feature_cache_error = feature_cache.as_ref().err();
    let database_reusable = can_reuse_feature_database(
        force_rebuild,
        feature_checkpoint_matches,
        legacy_feature_checkpoint_matches,
        feature_cache.is_ok(),
    );
    let mut feature_cache_reusable = database_reusable && feature_cache_complete;
    let final_complete = database_reusable
        && feature_cache_reusable
        && colmap_database_header_valid(&db)
        && sparse_rig_model_exists(&sparse);
    let reuse_candidate =
        can_reuse_align_result(final_complete, checkpoint_matches, checkpoint_completed);
    let cached_validation_error = if reuse_candidate {
        match validate_colmap_configured_rig_model(
            app,
            id,
            &colmap,
            &root,
            &sparse.join("0"),
            control,
        ) {
            Ok(calibrated_sensor_count) => {
                emit_log(
                    app,
                    id,
                    "info",
                    format!(
                        "已重新驗證 checkpoint 模型與 {calibrated_sensor_count} 個 non-reference sensor 外參"
                    ),
                );
                None
            }
            Err(error) => Some(error),
        }
    } else {
        None
    };
    if reuse_candidate && cached_validation_error.is_none() {
        remove_align_artifact(&unconfigured_database_backup)?;
        emit_log(
            app,
            id,
            "info",
            "已驗證對齊 checkpoint 與現有 COLMAP 重建結果",
        );
        emit_progress_detailed(
            app,
            id,
            &StageName::Align,
            "completed",
            1.0,
            "已沿用現有 COLMAP 重建結果",
            "running",
            false,
            None,
            None,
            Some("existing sparse model".to_owned()),
            None,
        );
        return Ok(vec![
            db.to_string_lossy().into_owned(),
            root.join("rig_config.json").to_string_lossy().into_owned(),
            sparse.to_string_lossy().into_owned(),
        ]);
    }
    if let Some(error) = &cached_validation_error {
        emit_log(
            app,
            id,
            "warning",
            format!("既有 COLMAP checkpoint 模型驗證失敗，將安全重建：{error}"),
        );
    }
    if !checkpoint_matches
        || !checkpoint_completed
        || !final_complete
        || cached_validation_error.is_some()
    {
        if force_rebuild {
            emit_log(
                app,
                id,
                "info",
                "已要求重跑 Align；清理舊 COLMAP 輸出後完整重建",
            );
        } else if checkpoint_present && !checkpoint_matches {
            emit_log(
                app,
                id,
                "warning",
                "對齊輸入 checkpoint 已變更；清理舊 COLMAP 輸出後重建",
            );
        } else if checkpoint_present {
            emit_log(
                app,
                id,
                "warning",
                "先前對齊未完整完成；清理舊 COLMAP 輸出後安全重建",
            );
        } else {
            emit_log(
                app,
                id,
                "info",
                "找不到有效對齊 checkpoint；不完整 COLMAP 輸出將清理後重建",
            );
        }
        if let Some(error) = feature_cache_error {
            if db.is_file() {
                emit_log(
                    app,
                    id,
                    "warning",
                    format!("COLMAP 特徵資料庫無法使用，將刪除資料庫後重跑：{error}"),
                );
            }
        } else if database_reusable {
            let (expected, completed) = feature_cache_counts.unwrap_or((0, 0));
            if feature_cache_complete {
                emit_log(
                    app,
                    id,
                    "info",
                    format!("COLMAP 特徵快取完整（{completed} / {expected} 張影像）"),
                );
            } else {
                emit_log(
                    app,
                    id,
                    "info",
                    format!("COLMAP 特徵快取部分完成（{completed} / {expected} 張影像），保留資料庫續跑"),
                );
            }
        } else if db.is_file() {
            emit_log(
                app,
                id,
                "info",
                "COLMAP 特徵輸入 fingerprint 已變更或不存在，將重建特徵資料庫",
            );
        }
        let mut preserve_database = database_reusable && cached_validation_error.is_none();
        if preserve_database && unconfigured_database_backup.is_dir() {
            restore_colmap_database_backup(&db, &unconfigured_database_backup)?;
            emit_log(
                app,
                id,
                "info",
                "已還原套用相機組前的 COLMAP 資料庫，可沿用特徵與配對重試",
            );
        } else if preserve_database && rig_mapping_plan == RigMappingPlan::BootstrapThenFinal {
            match database_has_nontrivial_rig(&db) {
                Ok(false) => {}
                Ok(true) => {
                    preserve_database = false;
                    emit_log(
                        app,
                        id,
                        "warning",
                        "既有 COLMAP 資料庫已套用相機組，但缺少校正前備份；為確保未知外參 bootstrap 正確，將重建資料庫",
                    );
                }
                Err(error) => {
                    preserve_database = false;
                    emit_log(
                        app,
                        id,
                        "warning",
                        format!("無法確認 COLMAP 資料庫是否仍為獨立鏡頭狀態，將安全重建：{error}"),
                    );
                }
            }
        }
        cleanup_align_artifacts(&root, preserve_database)?;
        if preserve_database && !checkpoint_matches {
            match clear_matching_cache(&db) {
                Ok(()) => emit_log(
                    app,
                    id,
                    "info",
                    "特徵輸入未變，已保留特徵並清除需重新計算的影像配對快取",
                ),
                Err(error) => {
                    emit_log(
                        app,
                        id,
                        "warning",
                        format!("無法清除舊 COLMAP 配對快取，將改用乾淨資料庫重跑：{error}"),
                    );
                    remove_colmap_database_artifacts(&db)?;
                    feature_cache_reusable = false;
                }
            }
        } else if !preserve_database {
            feature_cache_reusable = false;
        }
        write_align_checkpoint(&checkpoint_path, &fingerprint, &feature_fingerprint, false)?;
    }
    let (feature_extraction_gpu, feature_matching_gpu, mapper_gpu) = if requested_gpu {
        if !colmap_capabilities.cuda_build {
            emit_log(
                app,
                id,
                "warning",
                "COLMAP 未以 CUDA 建置；SIFT 擷取與影像配對改用 CPU",
            );
        }
        if colmap_capabilities.cuda_build && !colmap_capabilities.feature_extraction_gpu {
            emit_log(
                app,
                id,
                "warning",
                "指定的 COLMAP feature_extractor 不支援 GPU 選項；特徵擷取改用 CPU",
            );
        }
        if colmap_capabilities.cuda_build && !colmap_capabilities.feature_matching_gpu {
            emit_log(
                app,
                id,
                "warning",
                "指定的 COLMAP matches_importer 不支援 GPU 選項；影像配對改用 CPU",
            );
        }
        if colmap_capabilities.cuda_build && !colmap_capabilities.mapper_ba_gpu {
            emit_log(
                app,
                id,
                "warning",
                "指定的 COLMAP mapper 不支援 Ceres GPU 選項；Bundle Adjustment 改用 CPU",
            );
        }
        (
            colmap_capabilities.feature_extraction_gpu,
            colmap_capabilities.feature_matching_gpu,
            colmap_capabilities.mapper_ba_gpu,
        )
    } else {
        (false, false, false)
    };
    if mapper_gpu {
        emit_log(
            app,
            id,
            "info",
            "將以 COLMAP Ceres CUDA/cuDSS 執行可用的 Bundle Adjustment；若執行期回退 CPU 會另外警告",
        );
    }
    let mapper_gpu_index = mapper_gpu_index(&gpu_index).to_owned();
    let feature_gpu_args = feature_extractor_args(&root, &db, true, &gpu_index, use_masks);
    let feature_cpu_args = feature_extractor_args(&root, &db, false, &gpu_index, use_masks);
    if feature_cache_reusable {
        let (expected, completed) = feature_cache_counts.unwrap_or((0, 0));
        emit_log(
            app,
            id,
            "info",
            format!(
                "COLMAP 特徵快取完整（{completed} / {expected} 張影像），略過 feature_extractor"
            ),
        );
        emit_progress_detailed(
            app,
            id,
            &StageName::Align,
            "feature-extraction",
            0.0,
            format!("已確認 {completed} / {expected} 張影像已有特徵，略過擷取"),
            "running",
            false,
            Some(completed as u64),
            Some(expected as u64),
            Some("feature cache".to_owned()),
            None,
        );
        emit_colmap_step_completed(
            app,
            id,
            "feature-extraction",
            0,
            "影像特徵已完整存在，略過特徵擷取",
            "feature cache",
        );
    } else {
        emit_progress_detailed(
            app,
            id,
            &StageName::Align,
            "feature-extraction",
            0.0,
            "正在擷取影像特徵",
            "running",
            false,
            None,
            None,
            Some("feature_extractor".to_owned()),
            None,
        );
        let mut feature_item = Some("feature_extractor".to_owned());
        let highest_feature_fraction = Cell::new(0.0_f32);
        let last_feature_emit = Cell::new(None);
        run_colmap_with_gpu_fallback(
            app,
            id,
            &colmap,
            &feature_gpu_args,
            &feature_cpu_args,
            feature_extraction_gpu,
            "feature_extractor",
            control,
            || {
                // A GPU failure may leave a partially committed SQLite database.
                // Retry CPU only after restoring the clean feature database.
                remove_colmap_database_artifacts(&db)?;
                highest_feature_fraction.set(0.0);
                last_feature_emit.set(None);
                Ok(())
            },
            |line| {
                if let Some(progress) = parse_feature_progress(line) {
                    let fraction = progress.current as f32 / progress.total as f32;
                    if fraction < highest_feature_fraction.get() {
                        return;
                    }
                    highest_feature_fraction.set(fraction);
                    let terminal = progress.current == progress.total;
                    if !terminal
                        && last_feature_emit
                            .get()
                            .is_some_and(|last: Instant| last.elapsed() < COLMAP_PROGRESS_INTERVAL)
                    {
                        return;
                    }
                    last_feature_emit.set(Some(Instant::now()));
                    emit_colmap_progress(
                        app,
                        id,
                        "feature-extraction",
                        0,
                        highest_feature_fraction.get(),
                        format!("已處理 {} / {} 張影像", progress.current, progress.total),
                        feature_item.clone(),
                    );
                    return;
                }
                if let Some(current_item) = parse_feature_name(line) {
                    feature_item = Some(current_item);
                }
            },
        )?;
        emit_colmap_step_completed(
            app,
            id,
            "feature-extraction",
            0,
            "影像特徵擷取完成",
            "feature_extractor",
        );
    }
    if rig_preconfigured {
        // With calibrated extrinsics, configure frames before mapping so both
        // back-to-back sensors contribute to one rig pose from the beginning.
        // The independent-camera bootstrap is only necessary for unknown poses.
        remove_align_artifact(&unconfigured_database_backup)?;
        create_colmap_database_backup(&db, &unconfigured_database_backup)?;
        let configure_result = run_child(
            app,
            id,
            &colmap,
            &[
                "rig_configurator".into(),
                "--database_path".into(),
                db.to_string_lossy().into_owned(),
                "--rig_config_path".into(),
                rig_config_path.to_string_lossy().into_owned(),
            ],
            control,
        );
        if let Err(error) = configure_result {
            return Err(format!(
                "{error}；已保留套用相機組前的 COLMAP 資料庫，重試時可安全還原"
            ));
        }
        remove_align_artifact(&unconfigured_database_backup)?;
        emit_log(app, id, "info", "已在初始建模前套用固定雙鏡頭相對外參");
    }
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "matching",
        1.0 / 5.0,
        "正在匯入受限影像配對",
        "running",
        false,
        None,
        None,
        Some("matches_importer".to_owned()),
        None,
    );
    let matching_progress = Cell::new(None::<ColmapFraction>);
    let highest_matching_fraction = Cell::new(0.0_f32);
    let matching_gpu_args = matches_importer_args(&root, &db, true, &gpu_index);
    let matching_cpu_args = matches_importer_args(&root, &db, false, &gpu_index);
    let matching_database_backup = root.join("metadata/.align-matching-database.backup");
    remove_align_artifact(&matching_database_backup)?;
    if feature_matching_gpu {
        create_colmap_database_backup(&db, &matching_database_backup)?;
    }
    let matching_result = run_colmap_with_gpu_fallback(
        app,
        id,
        &colmap,
        &matching_gpu_args,
        &matching_cpu_args,
        feature_matching_gpu,
        "matches_importer",
        control,
        || {
            restore_colmap_database_backup(&db, &matching_database_backup)?;
            matching_progress.set(None);
            highest_matching_fraction.set(0.0);
            Ok(())
        },
        |line| {
            let parsed = parse_matching_progress(line);
            if parsed.is_none() && line.contains(" in ") {
                let Some(progress) = matching_progress.take() else {
                    return;
                };
                let fraction = progress.current as f32 / progress.total as f32;
                if fraction < highest_matching_fraction.get() {
                    return;
                }
                highest_matching_fraction.set(fraction);
                emit_colmap_progress(
                    app,
                    id,
                    "matching",
                    1,
                    highest_matching_fraction.get(),
                    format!("已完成配對區塊 {} / {}", progress.current, progress.total),
                    Some(format!("區塊 {} / {}", progress.current, progress.total)),
                );
                return;
            }
            let Some(progress) = parsed else {
                return;
            };
            let completed = if line
                .split_once(']')
                .is_some_and(|(_, suffix)| suffix.contains(" in "))
            {
                matching_progress.set(None);
                progress.current
            } else {
                matching_progress.set(Some(progress));
                progress.current.saturating_sub(1)
            };
            let fraction = completed as f32 / progress.total as f32;
            if fraction < highest_matching_fraction.get() {
                return;
            }
            highest_matching_fraction.set(fraction);
            emit_colmap_progress(
                app,
                id,
                "matching",
                1,
                highest_matching_fraction.get(),
                format!("正在處理配對區塊 {} / {}", progress.current, progress.total),
                Some(format!("區塊 {} / {}", progress.current, progress.total)),
            );
        },
    );
    let matching_backup_cleanup = remove_align_artifact(&matching_database_backup);
    match (matching_result, matching_backup_cleanup) {
        (Ok(()), Ok(())) => {}
        (Err(error), Ok(())) => return Err(error),
        (Ok(()), Err(cleanup_error)) => return Err(cleanup_error),
        (Err(error), Err(cleanup_error)) => return Err(format!("{error}；{cleanup_error}")),
    }
    emit_colmap_step_completed(app, id, "matching", 1, "影像配對完成", "matches_importer");
    if rig_mapping_plan == RigMappingPlan::BootstrapThenFinal {
        emit_progress_detailed(
            app,
            id,
            &StageName::Align,
            "bootstrap",
            2.0 / 5.0,
            "正在建立初始模型",
            "running",
            false,
            None,
            None,
            Some("bootstrap_mapper".to_owned()),
            None,
        );
        let bootstrap = root.join("sparse_bootstrap");
        if !sparse_model_exists(&bootstrap) {
            if bootstrap.exists() {
                fs::remove_dir_all(&bootstrap).map_err(|e| e.to_string())?;
            }
            fs::create_dir_all(&bootstrap).map_err(|e| e.to_string())?;
            let highest_registered = Cell::new(0);
            let last_bootstrap_emit = Cell::new(None);
            let mut bootstrap_gpu_warning_emitted = false;
            let bootstrap_total = independent_image_total.max(1);
            let bootstrap_gpu_args = mapper_args(
                &db,
                &root.join("images"),
                &bootstrap,
                mapper_gpu,
                &mapper_gpu_index,
                false,
                false,
            );
            let bootstrap_cpu_args = mapper_args(
                &db,
                &root.join("images"),
                &bootstrap,
                false,
                &mapper_gpu_index,
                false,
                false,
            );
            run_mapper_with_gpu_fallback(
                app,
                id,
                &colmap,
                &bootstrap,
                &bootstrap_gpu_args,
                &bootstrap_cpu_args,
                mapper_gpu,
                "bootstrap_mapper",
                control,
                || {
                    highest_registered.set(0);
                    last_bootstrap_emit.set(None);
                },
                |line| {
                    maybe_log_mapper_gpu_cpu_fallback(
                        app,
                        id,
                        "bootstrap_mapper",
                        mapper_gpu,
                        &mut bootstrap_gpu_warning_emitted,
                        line,
                    );
                    let Some((image_id, registered)) = parse_mapper_registration(line) else {
                        return;
                    };
                    highest_registered.set(
                        highest_registered
                            .get()
                            .max(registered)
                            .min(bootstrap_total),
                    );
                    let terminal = highest_registered.get() == bootstrap_total;
                    if !terminal
                        && last_bootstrap_emit
                            .get()
                            .is_some_and(|last: Instant| last.elapsed() < COLMAP_PROGRESS_INTERVAL)
                    {
                        return;
                    }
                    last_bootstrap_emit.set(Some(Instant::now()));
                    emit_colmap_progress(
                        app,
                        id,
                        "bootstrap",
                        2,
                        highest_registered.get() as f32 / bootstrap_total as f32,
                        format!(
                            "正在建立獨立鏡頭初始模型，已註冊約 {} / {} 張影像",
                            highest_registered.get(),
                            bootstrap_total
                        ),
                        Some(format!("影像 #{image_id}")),
                    );
                },
            )?;
        }
        let boot0 = bootstrap.join("0");
        if !boot0.is_dir() {
            return Err("COLMAP 初始建模未產生 sparse/0".into());
        }
        validate_colmap_bootstrap_for_rig(app, id, &colmap, &root, &boot0, control)?;
        emit_colmap_step_completed(
            app,
            id,
            "bootstrap",
            2,
            "初始模型建立完成",
            "bootstrap_mapper",
        );
        emit_progress_detailed(
            app,
            id,
            &StageName::Align,
            "rig",
            3.0 / 5.0,
            "正在估計雙鏡頭相機組",
            "running",
            false,
            None,
            None,
            Some("rig_configurator".to_owned()),
            None,
        );
        let configured_bootstrap = root.join("metadata/.align-configured-rig");
        remove_align_artifact(&configured_bootstrap)?;
        remove_align_artifact(&unconfigured_database_backup)?;
        create_colmap_database_backup(&db, &unconfigured_database_backup)?;
        fs::create_dir_all(&configured_bootstrap)
            .map_err(|error| format!("無法建立 COLMAP 相機組驗證模型資料夾：{error}"))?;
        let mut rig_derivation_failure = None;
        let configure_result = run_child_with_output(
            app,
            id,
            &colmap,
            &[
                "rig_configurator".into(),
                "--database_path".into(),
                db.to_string_lossy().into_owned(),
                "--input_path".into(),
                boot0.to_string_lossy().into_owned(),
                "--rig_config_path".into(),
                root.join("rig_config.json").to_string_lossy().into_owned(),
                "--output_path".into(),
                configured_bootstrap.to_string_lossy().into_owned(),
            ],
            control,
            |line| {
                if is_rig_pose_derivation_failure_line(line) {
                    rig_derivation_failure = Some(line.trim().to_owned());
                }
            },
        )
        .and_then(|()| {
            if let Some(detail) = rig_derivation_failure {
                return Err(format!(
                    "COLMAP 無法從初始模型推算 sensor_from_rig：{detail}"
                ));
            }
            validate_colmap_configured_rig_model(
                app,
                id,
                &colmap,
                &root,
                &configured_bootstrap,
                control,
            )
        });
        let configured_cleanup = remove_align_artifact(&configured_bootstrap);
        let calibrated_sensor_count = match (configure_result, configured_cleanup) {
            (Ok(count), Ok(())) => count,
            (Err(error), cleanup) => {
                let configured_cleanup_detail = cleanup
                    .err()
                    .map(|value| format!("；{value}"))
                    .unwrap_or_default();
                match restore_colmap_database_backup(&db, &unconfigured_database_backup) {
                    Ok(()) => {
                        let backup_cleanup = remove_align_artifact(&unconfigured_database_backup);
                        let align_cleanup = cleanup_align_artifacts(&root, true);
                        let recovery_detail = backup_cleanup
                            .err()
                            .into_iter()
                            .chain(align_cleanup.err())
                            .map(|value| format!("；{value}"))
                            .collect::<String>();
                        return Err(format!(
                            "{error}；已還原相機組校正前的 COLMAP 特徵／配對資料庫，可安全重試{configured_cleanup_detail}{recovery_detail}"
                        ));
                    }
                    Err(restore_error) => {
                        let hard_cleanup = cleanup_align_artifacts(&root, false)
                            .err()
                            .map(|value| format!("；{value}"))
                            .unwrap_or_default();
                        return Err(format!(
                            "{error}；無法還原相機組校正前資料庫：{restore_error}；已清理受污染的 COLMAP 對齊快取{configured_cleanup_detail}{hard_cleanup}"
                        ));
                    }
                }
            }
            (Ok(_), Err(error)) => return Err(error),
        };
        emit_log(
            app,
            id,
            "info",
            format!("已驗證 {calibrated_sensor_count} 個 non-reference sensor 的相機組外參"),
        );
        emit_colmap_step_completed(
            app,
            id,
            "rig",
            3,
            "雙鏡頭相機組估計完成",
            "rig_configurator",
        );
    } else {
        emit_log(
            app,
            id,
            "info",
            "相機組已有完整外參，略過獨立鏡頭 bootstrap mapper 與重複 rig 推算",
        );
        emit_colmap_step_completed(
            app,
            id,
            "bootstrap",
            2,
            "相機組外參已知，略過初始模型白工",
            "bootstrap skipped",
        );
        emit_colmap_step_completed(
            app,
            id,
            "rig",
            3,
            "已沿用固定雙鏡頭相對外參",
            "preconfigured rig",
        );
    }
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "final-mapping",
        4.0 / 5.0,
        "正在重建最終模型",
        "running",
        false,
        None,
        None,
        Some("final_mapper".to_owned()),
        None,
    );
    if sparse.exists() {
        fs::remove_dir_all(&sparse).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(&sparse).map_err(|e| e.to_string())?;
    let highest_registered = Cell::new(0);
    let last_final_mapper_emit = Cell::new(None);
    let mut final_gpu_warning_emitted = false;
    let final_total = rig_frame_total.max(1);
    let final_mapper_gpu = if mapper_mode == MapperMode::Global {
        requested_gpu
            && colmap_capabilities.cuda_build
            && colmap_capabilities.global_mapper_gp_gpu
            && colmap_capabilities.global_mapper_ba_gpu
    } else {
        mapper_gpu
    };
    if mapper_mode == MapperMode::Global && requested_gpu && !final_mapper_gpu {
        emit_log(
            app,
            id,
            "warning",
            "global_mapper 的 GPU positioning/Ceres option 不完整；改用 CPU global mapper",
        );
    }
    let final_gpu_args = if mapper_mode == MapperMode::Global {
        global_mapper_args(
            &db,
            &root.join("images"),
            &sparse,
            true,
            &gpu_index,
            use_gravity_prior,
        )
    } else {
        mapper_args(
            &db,
            &root.join("images"),
            &sparse,
            final_mapper_gpu,
            &mapper_gpu_index,
            true,
            true,
        )
    };
    let final_cpu_args = if mapper_mode == MapperMode::Global {
        global_mapper_args(
            &db,
            &root.join("images"),
            &sparse,
            false,
            &gpu_index,
            use_gravity_prior,
        )
    } else {
        mapper_args(
            &db,
            &root.join("images"),
            &sparse,
            false,
            &mapper_gpu_index,
            true,
            true,
        )
    };
    let final_mapper_component = if mapper_mode == MapperMode::Global {
        "global_mapper"
    } else {
        "final_mapper"
    };
    run_mapper_with_gpu_fallback(
        app,
        id,
        &colmap,
        &sparse,
        &final_gpu_args,
        &final_cpu_args,
        final_mapper_gpu,
        final_mapper_component,
        control,
        || {
            highest_registered.set(0);
            last_final_mapper_emit.set(None);
        },
        |line| {
            maybe_log_mapper_gpu_cpu_fallback(
                app,
                id,
                final_mapper_component,
                final_mapper_gpu,
                &mut final_gpu_warning_emitted,
                line,
            );
            let Some((image_id, registered)) = parse_mapper_registration(line) else {
                return;
            };
            highest_registered.set(highest_registered.get().max(registered).min(final_total));
            let terminal = highest_registered.get() == final_total;
            if !terminal
                && last_final_mapper_emit
                    .get()
                    .is_some_and(|last: Instant| last.elapsed() < COLMAP_PROGRESS_INTERVAL)
            {
                return;
            }
            last_final_mapper_emit.set(Some(Instant::now()));
            emit_colmap_progress(
                app,
                id,
                "final-mapping",
                4,
                highest_registered.get() as f32 / final_total as f32,
                format!(
                    "正在重建最終模型，已註冊約 {} / {} 組影格",
                    highest_registered.get(),
                    final_total
                ),
                Some(format!("影像 #{image_id}")),
            );
        },
    )?;
    if !sparse_rig_model_exists(&sparse) {
        return Err("COLMAP 最終建模結束但未產生含 rigs/frames 的有效 sparse model".into());
    }
    let final_calibrated_sensor_count =
        validate_colmap_configured_rig_model(app, id, &colmap, &root, &sparse.join("0"), control)?;
    emit_log(
        app,
        id,
        "info",
        format!("已驗證最終模型與 {final_calibrated_sensor_count} 個 non-reference sensor 外參"),
    );
    write_align_checkpoint(&checkpoint_path, &fingerprint, &feature_fingerprint, true)?;
    remove_align_artifact(&unconfigured_database_backup)?;
    emit_colmap_step_completed(
        app,
        id,
        "final-mapping",
        4,
        "最終模型重建完成",
        final_mapper_component,
    );
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "completed",
        1.0,
        "對齊處理完成",
        "running",
        false,
        None,
        None,
        Some("completed".to_owned()),
        None,
    );
    Ok(vec![
        db.to_string_lossy().into_owned(),
        root.join("rig_config.json").to_string_lossy().into_owned(),
        sparse.to_string_lossy().into_owned(),
    ])
}

#[cfg(test)]
mod tests {
    use super::{
        balanced_select_expression, build_align_fingerprint, build_feature_fingerprint,
        can_reuse_align_result, can_reuse_feature_database, candidate_ffmpeg_args,
        candidate_image_names, cleanup_align_artifacts, cleanup_obsolete_candidate_cache,
        cleanup_stale_full_res_dirs, colmap_step_progress, create_colmap_database_backup,
        dual_fisheye_registration_totals, expected_candidate_frames, extract_frame_settings,
        extraction_completed_count, feature_extractor_args, global_mapper_args,
        global_mapper_prerequisite_error, has_valid_global_mapper_priors,
        is_mapper_gpu_cpu_fallback_line, is_rig_pose_derivation_failure_line,
        keyframe_pruning_settings, load_candidate_selection_checkpoint, map_full_res_candidates,
        mapper_args, mapper_gpu_index, mapper_mode, mask_classes, mask_confidence, mask_enabled,
        matches_importer_args, parse_feature_name, parse_feature_progress, parse_gpu_index,
        parse_mapper_registration, parse_matching_progress, parse_showinfo_timestamp_ms,
        probe_duration_seconds, read_raw_frames, registered_rig_image_names,
        restore_colmap_database_backup, rig_config_has_complete_sensor_poses, rig_mapping_plan,
        selected_ffmpeg_args, source_stage_progress, synchronized_candidate_count,
        validate_rig_bootstrap_registration, validate_rigs_text_sensor_poses, with_hwaccel_auto,
        write_candidate_selection_checkpoint, write_rig_and_pairs, AlignCheckpoint, ColmapFraction,
        ExtractionStage, JobControl, JobManager, LogEvent, MapperMode, ProgressEvent,
        RawFrameMessage, RigBootstrapCamera, RigBootstrapConfig, RigMappingPlan, StageName,
        StartStageRequest, StreamingCandidateSelector, CANDIDATE_FRAME_BYTES,
        CANDIDATE_IMAGE_FORMAT, CANDIDATE_PROXY_SIZE, CANDIDATE_STREAM_WIDTH,
    };
    use crate::doctor::ColmapCapabilities;
    use crate::masking::CancelToken;
    use serde_json::json;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn extract_frame_settings_use_three_fps_and_preserve_candidate_multipliers() {
        assert_eq!(extract_frame_settings(&json!({})), (3.0, 12.0, true));
        assert_eq!(
            extract_frame_settings(&json!({
                "extract": { "baseFps": 5.0, "denseFps": 50.0, "skipBlurry": false }
            })),
            (5.0, 50.0, false)
        );
        assert_eq!(
            extract_frame_settings(&json!({
                "extract": { "baseFps": 30.0, "denseFps": 500.0 }
            })),
            (30.0, 300.0, true)
        );
        assert_eq!(
            extract_frame_settings(&json!({
                "extract": { "baseFps": 30.0, "denseFps": 1.0 }
            })),
            (30.0, 60.0, true)
        );
    }

    #[test]
    fn keyframe_pruning_settings_default_and_clamp_thresholds() {
        let (enabled, defaults) = keyframe_pruning_settings(&json!({}));
        assert!(enabled);
        assert_eq!(
            defaults,
            crate::extraction::KeyframePruningConfig::default()
        );

        let (enabled, thresholds) = keyframe_pruning_settings(&json!({
            "extract": {
                "keyframePruning": false,
                "minRotationDeg": 0.0,
                "minGapMs": 3000.0,
                "maxGapMs": 100.0,
                "minVisualNovelty": 2.0
            }
        }));
        assert!(!enabled);
        assert_eq!(thresholds.min_rotation_deg, 0.1);
        assert_eq!(thresholds.min_gap_ms, 2000.0);
        assert_eq!(thresholds.max_gap_ms, 2000.0);
        assert_eq!(thresholds.min_visual_novelty, 1.0);
    }

    #[test]
    fn migrates_only_the_legacy_default_mask_confidence() {
        assert_eq!(mask_confidence(&json!({"mask": {"confidence": 72}})), 25.0);
        assert_eq!(mask_confidence(&json!({"mask": {"confidence": 60}})), 60.0);
        assert_eq!(
            mask_confidence(&json!({"mask": {"confidence": 72, "confidenceVersion": 2}})),
            72.0
        );
    }

    #[test]
    fn mask_settings_support_independent_yolo_and_sky_filters() {
        let all_off = json!({
            "mask": { "yoloEnabled": false, "classes": ["person"], "maskSky": false }
        });
        assert!(!mask_enabled(&all_off));
        assert!(mask_classes(&all_off).is_empty());

        let yolo_only = json!({
            "mask": { "yoloEnabled": true, "classes": ["person", "  "], "maskSky": false }
        });
        assert!(mask_enabled(&yolo_only));
        assert_eq!(mask_classes(&yolo_only), vec!["person"]);

        let sky_only = json!({
            "mask": { "yoloEnabled": false, "classes": [], "maskSky": true }
        });
        assert!(mask_enabled(&sky_only));

        let legacy = json!({ "mask": { "classes": ["car"], "maskSky": false } });
        assert!(mask_enabled(&legacy));
        assert_eq!(mask_classes(&legacy), vec!["car"]);
    }

    #[test]
    fn parses_gpu_index_with_safe_defaults_and_normalized_lists() {
        assert_eq!(parse_gpu_index(&json!({})), Ok("-1".to_owned()));
        assert_eq!(
            parse_gpu_index(&json!({"align": {"gpuIndex": " -1 "}})),
            Ok("-1".to_owned())
        );
        assert_eq!(
            parse_gpu_index(&json!({"align": {"gpuIndex": "0, 2,1"}})),
            Ok("0,2,1".to_owned())
        );
    }

    #[test]
    fn rejects_invalid_gpu_index_values() {
        for value in ["", " ", "-2", "1,-1", "1,,2", "gpu"] {
            assert!(
                parse_gpu_index(&json!({"align": {"gpuIndex": value}})).is_err(),
                "expected invalid gpu index to be rejected: {value:?}"
            );
        }
        assert!(parse_gpu_index(&json!({"align": {"gpuIndex": 0}})).is_err());
    }

    #[test]
    fn colmap_gpu_arguments_cover_extraction_matching_and_ceres_mapper() {
        let root = Path::new("project");
        let db = root.join("database.db");
        let images = root.join("images");
        let sparse = root.join("sparse");
        let feature = feature_extractor_args(root, &db, true, "0,1", true);
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--FeatureExtraction.use_gpu", "1"] }));
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--FeatureExtraction.gpu_index", "0,1"] }));
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--ImageReader.default_focal_length_factor", "0.3"] }));
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--FeatureExtraction.type", "SIFT"] }));
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--SiftExtraction.max_num_features", "8192"] }));
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--ImageReader.camera_model", "OPENCV_FISHEYE"] }));
        assert!(feature.contains(&"--ImageReader.mask_path".to_owned()));

        let matching = matches_importer_args(root, &db, true, "0,1");
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--FeatureMatching.use_gpu", "1"] }));
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--FeatureMatching.gpu_index", "0,1"] }));
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--FeatureMatching.max_num_matches", "8192"] }));

        let mapper_index = mapper_gpu_index("0,1");
        let mapper = mapper_args(&db, &images, &sparse, true, mapper_index, true, true);
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.multiple_models", "0"] }));
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.ba_local_backend", "CERES"] }));
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.ba_global_backend", "CERES"] }));
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.ba_use_gpu", "1"] }));
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.ba_gpu_index", "0"] }));
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.ba_global_ignore_redundant_points3D", "1"] }));
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.ba_global_frames_ratio", "1.4"] }));
        assert!(mapper
            .windows(2)
            .any(|args| { args == ["--Mapper.ba_global_points_ratio", "1.4"] }));
        assert!(!mapper.iter().any(|arg| arg.contains("CASPAR")));
    }

    #[test]
    fn mapper_uses_the_first_device_from_a_multi_gpu_sift_list() {
        assert_eq!(mapper_gpu_index("2,0,1"), "2");
        assert_eq!(mapper_gpu_index("-1"), "-1");
    }

    #[test]
    fn mapper_always_selects_ceres_on_supported_colmap() {
        let args = mapper_args(
            Path::new("database.db"),
            Path::new("images"),
            Path::new("sparse"),
            false,
            "-1",
            false,
            false,
        );
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.ba_local_backend", "CERES"]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.ba_global_backend", "CERES"]));
        assert!(!args
            .iter()
            .any(|arg| arg == "--Mapper.ba_global_ignore_redundant_points3D"));
        assert!(!args
            .iter()
            .any(|arg| arg == "--Mapper.ba_global_frames_ratio"));
    }

    #[test]
    fn detects_ceres_gpu_cpu_fallback_messages() {
        assert!(is_mapper_gpu_cpu_fallback_line(
            "Ceres was compiled without CUDA support"
        ));
        assert!(is_mapper_gpu_cpu_fallback_line(
            "Ceres was compiled without cuDSS support"
        ));
        assert!(is_mapper_gpu_cpu_fallback_line(
            "Falling back to CPU-based sparse solvers"
        ));
        assert!(!is_mapper_gpu_cpu_fallback_line(
            "Bundle adjustment converged"
        ));
    }

    #[test]
    fn bootstrap_model_requires_registered_same_name_images_for_unknown_rig_pose() {
        let config = RigBootstrapConfig {
            cameras: vec![
                RigBootstrapCamera {
                    image_prefix: "lens0/".to_owned(),
                    ref_sensor: true,
                    cam_from_rig_rotation: None,
                    cam_from_rig_translation: None,
                },
                RigBootstrapCamera {
                    image_prefix: "lens1/".to_owned(),
                    ref_sensor: false,
                    cam_from_rig_rotation: None,
                    cam_from_rig_translation: None,
                },
            ],
        };
        let prefixes = BTreeSet::from(["lens0/".to_owned(), "lens1/".to_owned()]);
        assert_eq!(
            rig_mapping_plan(std::slice::from_ref(&config)),
            RigMappingPlan::BootstrapThenFinal,
            "unknown sensor extrinsics require bootstrap and final mapper passes"
        );
        let registered = registered_rig_image_names(
            "# images\n1 1 0 0 0 0 0 0 1 lens0/frame one.png\n0 0 -1\n2 1 0 0 0 0 0 0 2 lens1/frame one.png\n0 0 -1\n",
            &prefixes,
        );
        let summary = validate_rig_bootstrap_registration(&[config], &registered).unwrap();
        assert!(summary.contains("1 組共同影格"));
    }

    #[test]
    fn bootstrap_model_reports_disconnected_rig_cameras_before_final_mapper() {
        let configs = serde_json::from_value::<Vec<RigBootstrapConfig>>(json!([{
            "cameras": [
                {"image_prefix": "lens0/", "ref_sensor": true},
                {"image_prefix": "lens1/"}
            ]
        }]))
        .unwrap();
        let registered = BTreeSet::from([
            "lens0/frame0001.png".to_owned(),
            "lens1/frame0002.png".to_owned(),
        ]);
        let error = validate_rig_bootstrap_registration(&configs, &registered).unwrap_err();
        assert!(error.contains("沒有任何同名影格同時註冊"));
        assert!(error.contains("lens0/"));
        assert!(error.contains("lens1/"));
    }

    #[test]
    fn explicit_rig_pose_requires_registered_cameras_but_not_shared_frames() {
        let configs = serde_json::from_value::<Vec<RigBootstrapConfig>>(json!([{
            "cameras": [
                {"image_prefix": "lens0/", "ref_sensor": true},
                {
                    "image_prefix": "lens1/",
                    "cam_from_rig_rotation": [1.0, 0.0, 0.0, 0.0],
                    "cam_from_rig_translation": [0.0, 0.0, 0.0]
                }
            ]
        }]))
        .unwrap();
        let registered = BTreeSet::from([
            "lens0/frame0001.png".to_owned(),
            "lens1/frame0002.png".to_owned(),
        ]);
        assert_eq!(
            validate_rig_bootstrap_registration(&configs, &registered).unwrap(),
            "相機組已提供完整外參"
        );
    }

    #[test]
    fn configured_rig_text_rejects_unknown_sensor_poses() {
        assert_eq!(
            validate_rigs_text_sensor_poses(
                "# rigs\n1 2 CAMERA 1 CAMERA 2 1 1 0 0 0 0.1 0 0\n2 1 CAMERA 3\n",
                &[2, 1],
            ),
            Ok(1)
        );
        let error = validate_rigs_text_sensor_poses("1 2 CAMERA 1 CAMERA 2 0\n", &[2]).unwrap_err();
        assert!(error.contains("缺少 sensor_from_rig"));
        let error =
            validate_rigs_text_sensor_poses("1 2 CAMERA 1 CAMERA 2 1 0 0 0 0 0.1 0 0\n", &[2])
                .unwrap_err();
        assert!(error.contains("quaternion"));
        assert!(is_rig_pose_derivation_failure_line(
            "Failed to derive sensor_from_rig transformation for camera 2"
        ));
    }

    #[test]
    fn configured_rig_text_rejects_trivial_or_missing_rigs() {
        let error =
            validate_rigs_text_sensor_poses("1 1 CAMERA 1\n2 1 CAMERA 2\n", &[2]).unwrap_err();
        assert!(error.contains("相機組結構不完整"));
    }

    #[test]
    fn align_fingerprint_changes_for_settings_and_input_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let images = temp.path().join("images");
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(images.join(lens)).unwrap();
            fs::write(images.join(lens).join("frame0001.png"), b"frame").unwrap();
        }
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        fs::write(temp.path().join("rig_config.json"), b"rig-v1").unwrap();
        fs::write(
            temp.path().join("metadata/pairs.txt"),
            b"lens0/frame0001.png lens1/frame0001.png\n",
        )
        .unwrap();
        let settings = json!({"align": {"useGpu": false, "gpuIndex": "-1"}});

        let baseline =
            build_align_fingerprint(temp.path(), &settings, "COLMAP 4.1.1", false).unwrap();
        assert_ne!(
            baseline,
            build_align_fingerprint(temp.path(), &settings, "COLMAP 4.2.0", false).unwrap()
        );
        let changed_settings = json!({"align": {"useGpu": true, "gpuIndex": "-1"}});
        assert_ne!(
            baseline,
            build_align_fingerprint(temp.path(), &changed_settings, "COLMAP 4.1.1", false).unwrap()
        );

        fs::write(
            temp.path().join("metadata/source000_frame_motion.json"),
            br#"{"schemaVersion":1,"frames":[{"sequence":1,"timestampMs":0}]}"#,
        )
        .unwrap();
        assert_ne!(
            baseline,
            build_align_fingerprint(temp.path(), &settings, "COLMAP 4.1.1", false).unwrap()
        );
        fs::write(
            temp.path().join("metadata/global_mapper_priors.json"),
            br#"{"schemaVersion":1,"focalPriorValid":true,"gravityPriorValid":true}"#,
        )
        .unwrap();
        assert_ne!(
            baseline,
            build_align_fingerprint(temp.path(), &settings, "COLMAP 4.1.1", false).unwrap()
        );

        fs::write(
            temp.path().join("metadata/pairs.txt"),
            b"lens0/frame0001.png lens1/frame0001.png\nlens0/frame0001.png lens0/frame0001.png\n",
        )
        .unwrap();
        assert_ne!(
            baseline,
            build_align_fingerprint(temp.path(), &settings, "COLMAP 4.1.1", false).unwrap()
        );

        fs::write(images.join("lens0/frame0001.png"), b"frame-with-new-size").unwrap();
        assert_ne!(
            baseline,
            build_align_fingerprint(temp.path(), &settings, "COLMAP 4.1.1", false).unwrap()
        );

        fs::create_dir_all(temp.path().join("masks_colmap")).unwrap();
        fs::write(temp.path().join("masks_colmap/frame0001.png"), b"mask").unwrap();
        assert_ne!(
            baseline,
            build_align_fingerprint(temp.path(), &settings, "COLMAP 4.1.0", true).unwrap()
        );
    }

    #[test]
    fn feature_fingerprint_only_tracks_feature_inputs_and_fixed_semantics() {
        let temp = tempfile::tempdir().unwrap();
        let images = temp.path().join("images");
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(images.join(lens)).unwrap();
            fs::write(images.join(lens).join("frame0001.png"), b"frame").unwrap();
        }
        let baseline = build_feature_fingerprint(temp.path(), "COLMAP 4.1.1", false).unwrap();
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        fs::write(temp.path().join("metadata/pairs.txt"), b"changed pairs").unwrap();
        fs::write(temp.path().join("rig_config.json"), b"changed rig").unwrap();
        assert_eq!(
            baseline,
            build_feature_fingerprint(temp.path(), "COLMAP 4.1.1", false).unwrap()
        );
        assert_ne!(
            baseline,
            build_feature_fingerprint(temp.path(), "COLMAP 4.2.0", false).unwrap()
        );
        fs::create_dir_all(temp.path().join("masks_colmap/lens0")).unwrap();
        fs::write(
            temp.path().join("masks_colmap/lens0/frame0001.png"),
            b"mask",
        )
        .unwrap();
        assert_ne!(
            baseline,
            build_feature_fingerprint(temp.path(), "COLMAP 4.1.1", true).unwrap()
        );
    }

    #[test]
    fn align_checkpoint_accepts_legacy_without_feature_fingerprint() {
        let checkpoint: AlignCheckpoint = serde_json::from_value(json!({
            "schemaVersion": 2,
            "fingerprint": "legacy",
            "completed": false
        }))
        .unwrap();
        assert_eq!(checkpoint.feature_fingerprint, None);
    }

    #[test]
    fn cleanup_can_preserve_database_artifacts_for_feature_resume() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("sparse/0")).unwrap();
        fs::write(temp.path().join("database.db"), b"db").unwrap();
        fs::write(temp.path().join("database.db-wal"), b"wal").unwrap();
        fs::write(temp.path().join("database.db-shm"), b"shm").unwrap();
        fs::write(temp.path().join("database.db-journal"), b"journal").unwrap();
        fs::create_dir_all(
            temp.path()
                .join("metadata/.align-unconfigured-database.backup"),
        )
        .unwrap();
        fs::write(
            temp.path()
                .join("metadata/.align-unconfigured-database.backup/database.db"),
            b"backup",
        )
        .unwrap();

        cleanup_align_artifacts(temp.path(), true).unwrap();
        assert!(temp.path().join("database.db").is_file());
        assert!(temp.path().join("database.db-wal").is_file());
        assert!(!temp.path().join("sparse").exists());
        assert!(!temp
            .path()
            .join("metadata/.align-unconfigured-database.backup")
            .exists());

        cleanup_align_artifacts(temp.path(), false).unwrap();
        assert!(!temp.path().join("database.db").exists());
        assert!(!temp.path().join("database.db-wal").exists());
    }

    #[test]
    fn database_backup_round_trip_is_atomic_before_destructive_restore() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("database.db");
        let backup = temp.path().join("database.backup");
        fs::write(&database, b"original database").unwrap();
        fs::write(temp.path().join("database.db-wal"), b"original wal").unwrap();

        create_colmap_database_backup(&database, &backup).unwrap();
        fs::write(&database, b"mutated database").unwrap();
        fs::write(temp.path().join("database.db-wal"), b"mutated wal").unwrap();
        fs::write(temp.path().join("database.db-shm"), b"transient shm").unwrap();
        restore_colmap_database_backup(&database, &backup).unwrap();

        assert_eq!(fs::read(&database).unwrap(), b"original database");
        assert_eq!(
            fs::read(temp.path().join("database.db-wal")).unwrap(),
            b"original wal"
        );
        assert!(!temp.path().join("database.db-shm").exists());

        let incomplete = temp.path().join("incomplete.backup");
        fs::create_dir(&incomplete).unwrap();
        fs::write(&database, b"still safe").unwrap();
        assert!(restore_colmap_database_backup(&database, &incomplete).is_err());
        assert_eq!(fs::read(&database).unwrap(), b"still safe");
    }

    #[test]
    fn align_result_reuse_requires_both_complete_output_and_matching_checkpoint() {
        assert!(can_reuse_align_result(true, true, true));
        assert!(!can_reuse_align_result(true, true, false));
        assert!(!can_reuse_align_result(true, false, true));
        assert!(!can_reuse_align_result(false, true, true));
    }

    #[test]
    fn feature_database_reuse_requires_feature_or_matching_legacy_fingerprint() {
        assert!(can_reuse_feature_database(false, true, false, true));
        assert!(can_reuse_feature_database(false, false, true, true));
        assert!(!can_reuse_feature_database(false, false, false, true));
        assert!(!can_reuse_feature_database(false, true, false, false));
        assert!(!can_reuse_feature_database(true, true, true, true));
    }

    #[test]
    fn progress_and_log_wire_payloads_include_timing_fields() {
        let progress = serde_json::to_value(ProgressEvent {
            job_id: "job-test".to_owned(),
            stage: StageName::Mask,
            phase: "masking".to_owned(),
            progress: 0.5,
            message: "處理中".to_owned(),
            completed: Some(5),
            total: Some(10),
            current_item: Some("frame.png".to_owned()),
            timestamp_ms: 123,
            elapsed_ms: Some(456),
            done: false,
            status: "running".to_owned(),
        })
        .unwrap();
        assert_eq!(progress["currentItem"], "frame.png");
        assert_eq!(progress["timestampMs"], 123);
        assert_eq!(progress["elapsedMs"], 456);

        let log = serde_json::to_value(LogEvent {
            job_id: "job-test".to_owned(),
            level: "info".to_owned(),
            message: "step".to_owned(),
            timestamp_ms: 789,
        })
        .unwrap();
        assert_eq!(log["timestampMs"], 789);
    }

    #[test]
    fn parses_colmap_feature_progress_across_log_formats() {
        assert_eq!(
            parse_feature_progress(
                "I20250611 20:54:46.728052 2394999 feature_extraction.cc:258] Processed file [17/463]"
            ),
            Some(ColmapFraction {
                current: 17,
                total: 463,
            })
        );
        assert_eq!(
            parse_feature_progress("Processing file [2/11]"),
            Some(ColmapFraction {
                current: 2,
                total: 11,
            })
        );
        assert_eq!(
            parse_feature_progress("Processed image [11/11]\r"),
            Some(ColmapFraction {
                current: 11,
                total: 11,
            })
        );
        assert_eq!(
            parse_feature_name("I0101 feature_extraction.cc:261]   Name: lens0/frame.png"),
            Some("lens0/frame.png".to_owned())
        );
        assert_eq!(parse_feature_progress("Processed file [12/10]"), None);
        assert_eq!(parse_feature_progress("Processed file [0/10]"), None);
        assert_eq!(parse_feature_progress("feature_extractor is running"), None);
    }

    #[test]
    fn parses_colmap_matching_block_variants_without_stripping_raw_brackets() {
        assert_eq!(
            parse_matching_progress("Matching block [2/4, 3/4] in 0.123s"),
            Some(ColmapFraction {
                current: 7,
                total: 16,
            })
        );
        assert_eq!(
            parse_matching_progress(
                "I20260701 12:00:00.000001 42 pairing.cc:889] Processing block [3/9]"
            ),
            Some(ColmapFraction {
                current: 3,
                total: 9,
            })
        );
        assert_eq!(
            parse_matching_progress("Matching image [13/48] in 0.250s"),
            Some(ColmapFraction {
                current: 13,
                total: 48,
            })
        );
        assert_eq!(parse_matching_progress("Processing block [x/4]"), None);
    }

    #[test]
    fn mapper_progress_uses_only_labeled_registered_frame_counts() {
        assert_eq!(
            parse_mapper_registration(
                "I20250611 incremental_pipeline.cc:607] Registering image #1921 (num_reg_frames=239)"
            ),
            Some((1921, 239))
        );
        assert_eq!(
            parse_mapper_registration("Registering image #12 (7)"),
            None,
            "legacy naked counts are attempt indices, not reliable completed counts"
        );
        assert_eq!(parse_mapper_registration("Registering image #x"), None);
    }

    #[test]
    fn colmap_substeps_map_to_monotonic_align_segments() {
        assert!((colmap_step_progress(0, 0.0) - 0.0).abs() < f32::EPSILON);
        assert!((colmap_step_progress(0, 0.5) - 0.1).abs() < f32::EPSILON);
        assert!((colmap_step_progress(1, 0.0) - 0.2).abs() < f32::EPSILON);
        assert!(colmap_step_progress(4, 1.0) < 1.0);
        assert!(colmap_step_progress(4, 1.0) > 0.99);
    }

    #[test]
    fn candidate_progress_uses_probe_duration_and_candidate_rate() {
        let probe = json!({
            "format": { "duration": "12.25" },
            "streams": [
                { "codec_type": "video", "duration": "10.0" },
                { "codec_type": "audio", "duration": "12.25" }
            ]
        });
        assert_eq!(probe_duration_seconds(&probe), Some(10.0));
        assert_eq!(expected_candidate_frames(&probe, 8.0), Some(80));

        let stream_only = json!({
            "streams": [
                { "codec_type": "video", "duration": "9.5" },
                { "codec_type": "video", "duration": "9.25" }
            ]
        });
        assert_eq!(probe_duration_seconds(&stream_only), Some(9.25));
        assert_eq!(expected_candidate_frames(&stream_only, 2.0), Some(19));
    }

    #[test]
    fn source_progress_stays_monotonic_across_multiple_sources() {
        assert_eq!(source_stage_progress(0, 2, 0.0), 0.0);
        assert_eq!(source_stage_progress(0, 2, 1.0), 0.5);
        assert_eq!(source_stage_progress(1, 2, 0.0), 0.5);
        assert_eq!(source_stage_progress(1, 2, 1.0), 1.0);
    }

    #[test]
    fn start_stage_request_accepts_a_local_colmap_path() {
        let request: StartStageRequest = serde_json::from_value(json!({
            "projectPath": "C:\\project",
            "stage": "align",
            "mode": "retry",
            "colmapPath": "C:\\COLMAP portable\\COLMAP.bat"
        }))
        .unwrap();

        assert_eq!(
            request.colmap_path.as_deref(),
            Some("C:\\COLMAP portable\\COLMAP.bat")
        );
        assert_eq!(request.mode.as_deref(), Some("retry"));
    }

    #[test]
    fn extraction_counts_advance_only_on_terminal_intervals() {
        let completed = AtomicU64::new(0);
        assert_eq!(
            extraction_completed_count(&completed, ExtractionStage::Scanning, 3, 5),
            0
        );
        assert_eq!(
            extraction_completed_count(&completed, ExtractionStage::Completed, 2, 5),
            2
        );
        assert_eq!(
            extraction_completed_count(&completed, ExtractionStage::Scoring, 4, 5),
            2
        );
        assert_eq!(
            extraction_completed_count(&completed, ExtractionStage::Skipped, 8, 5),
            5
        );
        assert_eq!(
            extraction_completed_count(&completed, ExtractionStage::Cancelled, 9, 5),
            5
        );
    }

    #[test]
    fn pair_list_connects_multiple_source_recordings() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for name in [
                "source000_00000001.png",
                "source000_00000002.png",
                "source001_00000001.png",
                "source001_00000002.png",
            ] {
                fs::write(temp.path().join("images").join(lens).join(name), b"frame").unwrap();
            }
            fs::write(
                temp.path().join("images").join(lens).join(".DS_Store"),
                b"metadata",
            )
            .unwrap();
        }

        assert_eq!(write_rig_and_pairs(temp.path()).unwrap(), 4);
        assert_eq!(
            dual_fisheye_registration_totals(temp.path()).unwrap(),
            (8, 4)
        );

        let pairs = fs::read_to_string(temp.path().join("metadata/pairs.txt")).unwrap();
        assert!(pairs.contains("lens0/source000_00000001.png lens0/source001_00000001.png"));
        assert!(pairs.contains("lens1/source000_00000001.png lens1/source001_00000001.png"));
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000001.png"));
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000002.png"));
        assert!(pairs.contains("lens1/source000_00000001.png lens0/source000_00000002.png"));
        assert!(!pairs.contains(".DS_Store"));
    }

    #[test]
    fn motion_metadata_prunes_only_optional_temporal_pairs() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for sequence in 1..=6 {
                fs::write(
                    temp.path()
                        .join("images")
                        .join(lens)
                        .join(format!("source000_{sequence:08}.png")),
                    b"frame",
                )
                .unwrap();
            }
        }
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        let frames = (1..=6)
            .map(|sequence| {
                json!({
                    "sequence": sequence,
                    "timestampMs": (sequence - 1) as f64 * 333.0,
                    "imuRotationFromLastKeptDeg": 0.0,
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            temp.path().join("metadata/source000_frame_motion.json"),
            serde_json::to_vec(&json!({"schemaVersion": 1, "frames": frames})).unwrap(),
        )
        .unwrap();

        write_rig_and_pairs(temp.path()).unwrap();
        let pairs = fs::read_to_string(temp.path().join("metadata/pairs.txt")).unwrap();
        // Same-time stereo, same-lens +1/+2, and cross-lens +1 remain.
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000001.png"));
        assert!(pairs.contains("lens0/source000_00000001.png lens0/source000_00000002.png"));
        assert!(pairs.contains("lens0/source000_00000001.png lens0/source000_00000003.png"));
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000002.png"));
        // +3 and beyond exceed the 700 ms temporal budget and have no
        // rotation novelty, so optional links are pruned.
        assert!(!pairs.contains("lens0/source000_00000001.png lens0/source000_00000004.png"));
        assert!(!pairs.contains("lens0/source000_00000001.png lens1/source000_00000004.png"));
        assert!(!pairs.contains("lens0/source000_00000001.png lens0/source000_00000006.png"));
    }

    #[test]
    fn malformed_motion_metadata_preserves_legacy_temporal_graph() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for sequence in 1..=6 {
                fs::write(
                    temp.path()
                        .join("images")
                        .join(lens)
                        .join(format!("source000_{sequence:08}.png")),
                    b"frame",
                )
                .unwrap();
            }
        }
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        fs::write(
            temp.path().join("metadata/source000_frame_motion.json"),
            br#"{"schemaVersion":1,"frames":[{"sequence":1}]}"#,
        )
        .unwrap();

        write_rig_and_pairs(temp.path()).unwrap();
        let pairs = fs::read_to_string(temp.path().join("metadata/pairs.txt")).unwrap();
        assert!(pairs.contains("lens0/source000_00000001.png lens0/source000_00000006.png"));
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000006.png"));
    }

    #[test]
    fn motion_metadata_pruning_thresholds_override_defaults() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for sequence in 1..=3 {
                fs::write(
                    temp.path()
                        .join("images")
                        .join(lens)
                        .join(format!("source000_{sequence:08}.png")),
                    b"frame",
                )
                .unwrap();
            }
        }
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        let frames = (1..=3)
            .map(|sequence| {
                json!({
                    "sequence": sequence,
                    "timestampMs": (sequence - 1) as f64 * 500.0,
                    "imuRotationFromLastKeptDeg": 0.0,
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            temp.path().join("metadata/source000_frame_motion.json"),
            serde_json::to_vec(&json!({
                "schemaVersion": 1,
                "thresholds": {
                    "minRotationDeg": 20.0,
                    "minGapMs": 200.0,
                    "maxGapMs": 1200.0,
                    "minVisualNovelty": 0.08,
                },
                "frames": frames,
            }))
            .unwrap(),
        )
        .unwrap();

        write_rig_and_pairs(temp.path()).unwrap();
        let pairs = fs::read_to_string(temp.path().join("metadata/pairs.txt")).unwrap();
        // Cross-lens +2 is optional and would fail the default 700 ms budget;
        // the top-level pruning object extends it to 1200 ms.
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000003.png"));
    }

    #[test]
    fn motion_metadata_accumulates_intermediate_rotation_for_long_links() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for sequence in 1..=4 {
                fs::write(
                    temp.path()
                        .join("images")
                        .join(lens)
                        .join(format!("source000_{sequence:08}.png")),
                    b"frame",
                )
                .unwrap();
            }
        }
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        let frames = (1..=4)
            .map(|sequence| {
                json!({
                    "sequence": sequence,
                    "timestampMs": (sequence - 1) as f64 * 500.0,
                    "imuRotationFromLastKeptDeg": if sequence == 1 { 0.0 } else { 2.0 },
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            temp.path().join("metadata/source000_frame_motion.json"),
            serde_json::to_vec(&json!({"schemaVersion": 1, "frames": frames})).unwrap(),
        )
        .unwrap();

        write_rig_and_pairs(temp.path()).unwrap();
        let pairs = fs::read_to_string(temp.path().join("metadata/pairs.txt")).unwrap();
        // Sequence 1 -> 4 is 1500 ms, but the intermediate 2° + 2° + 2°
        // rotations exceed the 4° threshold and preserve the optional link.
        assert!(pairs.contains("lens0/source000_00000001.png lens0/source000_00000004.png"));
    }

    #[test]
    fn motion_metadata_prefers_direct_attitude_rotation_when_available() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for sequence in 1..=3 {
                fs::write(
                    temp.path()
                        .join("images")
                        .join(lens)
                        .join(format!("source000_{sequence:08}.png")),
                    b"frame",
                )
                .unwrap();
            }
        }
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        let ten_deg = (5.0_f64.to_radians()).cos();
        let ten_deg_sine = (5.0_f64.to_radians()).sin();
        let frames = vec![
            json!({
                "sequence": 1,
                "timestampMs": 0.0,
                "imuRotationFromLastKeptDeg": 0.0,
                "attitudeWxyz": [1.0, 0.0, 0.0, 0.0],
            }),
            json!({
                "sequence": 2,
                "timestampMs": 500.0,
                "imuRotationFromLastKeptDeg": 0.0,
                "attitudeWxyz": [ten_deg, 0.0, 0.0, ten_deg_sine],
            }),
            json!({
                "sequence": 3,
                "timestampMs": 1000.0,
                "imuRotationFromLastKeptDeg": 0.0,
                "attitudeWxyz": [ten_deg, 0.0, 0.0, ten_deg_sine],
            }),
        ];
        fs::write(
            temp.path().join("metadata/source000_frame_motion.json"),
            serde_json::to_vec(&json!({"schemaVersion": 1, "frames": frames})).unwrap(),
        )
        .unwrap();

        write_rig_and_pairs(temp.path()).unwrap();
        let pairs = fs::read_to_string(temp.path().join("metadata/pairs.txt")).unwrap();
        // Endpoint attitude is 10° despite zero per-frame rotation fields;
        // direct relative rotation therefore retains optional cross-lens +2.
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000003.png"));
    }

    #[test]
    fn mapper_mode_defaults_to_auto_and_rejects_unknown_values() {
        assert_eq!(mapper_mode(&json!({})).unwrap(), MapperMode::Auto);
        assert_eq!(
            mapper_mode(&json!({"align": {"mapperMode": "GLOBAL"}})).unwrap(),
            MapperMode::Global
        );
        assert!(mapper_mode(&json!({"align": {"mapperMode": "bogus"}})).is_err());
    }

    #[test]
    fn global_mapper_args_use_colmap_4_1_1_option_names() {
        let args = global_mapper_args(
            Path::new("database.db"),
            Path::new("images"),
            Path::new("sparse"),
            true,
            "2,3",
            true,
        );
        let pairs = args
            .windows(2)
            .map(|pair| (pair[0].as_str(), pair[1].as_str()))
            .collect::<BTreeSet<_>>();
        assert!(pairs.contains(&("--GlobalMapper.ra_use_gravity", "1")));
        assert!(pairs.contains(&("--GlobalMapper.ra_use_stratified", "1")));
        assert!(pairs.contains(&("--GlobalMapper.gp_use_gpu", "1")));
        assert!(pairs.contains(&("--GlobalMapper.gp_gpu_index", "2")));
        assert!(pairs.contains(&("--GlobalMapper.ba_ceres_use_gpu", "1")));
        assert!(pairs.contains(&("--GlobalMapper.ba_ceres_gpu_index", "2")));
        assert!(pairs.contains(&("--GlobalMapper.refine_sensor_from_rig", "0")));
    }

    #[test]
    fn global_mapper_requires_validated_priors_before_explicit_use() {
        let temp = tempfile::tempdir().unwrap();
        let capabilities = ColmapCapabilities {
            global_mapper: true,
            global_mapper_gravity: true,
            global_mapper_gp_gpu: true,
            global_mapper_ba_gpu: true,
            ..Default::default()
        };
        let error = global_mapper_prerequisite_error(temp.path(), true, &capabilities, true)
            .expect("missing global mapper priors must block explicit mode");
        assert!(error.contains("focal prior"));
        assert!(error.contains("global_mapper_priors.json"));
    }

    #[test]
    fn gravity_global_mapper_marker_requires_calibration_offset_and_coverage() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        let marker = temp.path().join("metadata/global_mapper_priors.json");
        fs::write(
            &marker,
            serde_json::to_vec(&json!({
                "schemaVersion": 1,
                "focalPriorValid": true,
                "gravityPriorValid": true,
                "gravityCoverageRatio": 0.79,
                "sensorToCameraCalibrationVersion": "hand-eye-v1",
                "timeOffsetMs": 12.5,
                "databasePosePriorsInjected": true
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(!has_valid_global_mapper_priors(temp.path(), true));
        assert!(has_valid_global_mapper_priors(temp.path(), false));

        fs::write(
            &marker,
            serde_json::to_vec(&json!({
                "schemaVersion": 1,
                "focalPriorValid": true,
                "gravityPriorValid": true,
                "gravityCoverageRatio": 0.8,
                "sensorToCameraCalibrationVersion": "hand-eye-v1",
                "timeOffsetMs": 12.5,
                "databasePosePriorsInjected": true
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(has_valid_global_mapper_priors(temp.path(), true));
    }

    #[test]
    fn mapper_progress_totals_include_unpaired_lens_images() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for name in ["frame0001.png", "frame0002.png"] {
                fs::write(temp.path().join("images").join(lens).join(name), b"frame").unwrap();
            }
        }
        fs::write(
            temp.path().join("images/lens0/frame0003.png"),
            b"unpaired frame",
        )
        .unwrap();

        assert_eq!(
            dual_fisheye_registration_totals(temp.path()).unwrap(),
            (5, 3),
            "unknown bootstrap counts independent images; fixed-rig mapper counts union frames"
        );
    }

    #[test]
    fn rig_config_is_not_overwritten_when_already_present() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for name in ["frame0001.png", "frame0002.png"] {
                fs::write(temp.path().join("images").join(lens).join(name), b"frame").unwrap();
            }
        }
        let rig_path = temp.path().join("rig_config.json");
        let custom_rig = br#"[{"cameras":[{"image_prefix":"lens0/","ref_sensor":true},{"image_prefix":"lens1/","ref_sensor":false}]}]"#;
        fs::write(&rig_path, custom_rig).unwrap();

        write_rig_and_pairs(temp.path()).unwrap();

        assert_eq!(fs::read(&rig_path).unwrap(), custom_rig);
    }

    #[test]
    fn default_rig_uses_a_fixed_back_to_back_rotation_and_migrates_legacy_default() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for name in ["frame0001.png", "frame0002.png"] {
                fs::write(temp.path().join("images").join(lens).join(name), b"frame").unwrap();
            }
        }
        let rig_path = temp.path().join("rig_config.json");
        fs::write(
            &rig_path,
            br#"[{"cameras":[{"image_prefix":"lens0/","ref_sensor":true},{"image_prefix":"lens1/"}]}]"#,
        )
        .unwrap();

        write_rig_and_pairs(temp.path()).unwrap();

        let configs =
            serde_json::from_slice::<Vec<RigBootstrapConfig>>(&fs::read(&rig_path).unwrap())
                .unwrap();
        assert!(rig_config_has_complete_sensor_poses(&configs));
        assert_eq!(
            rig_mapping_plan(&configs),
            RigMappingPlan::PreconfiguredSinglePass,
            "known rig extrinsics must run only the final mapper pass"
        );
        assert_eq!(
            configs[0].cameras[1].cam_from_rig_rotation,
            Some(vec![0.0, 0.0, 1.0, 0.0])
        );
        assert_eq!(
            configs[0].cameras[1].cam_from_rig_translation,
            Some(vec![0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn hardware_acceleration_is_inserted_before_input_only() {
        let software = candidate_ffmpeg_args(Path::new("input.osv"), 0, 1, 8.0);
        assert!(!software.iter().any(|arg| arg == "-hwaccel"));

        let accelerated = with_hwaccel_auto(&software);
        let hwaccel_index = accelerated.iter().position(|arg| arg == "-hwaccel");
        let auto_index = accelerated.iter().position(|arg| arg == "auto");
        let input_index = accelerated.iter().position(|arg| arg == "-i");
        assert_eq!(
            accelerated.get(hwaccel_index.unwrap() + 1),
            Some(&"auto".to_owned())
        );
        assert_eq!(auto_index.unwrap() + 1, input_index.unwrap());
        assert_eq!(hwaccel_index.unwrap() + 2, input_index.unwrap());
        assert_eq!(software.iter().position(|arg| arg == "-i"), Some(2));
    }

    #[test]
    fn hardware_acceleration_helper_does_not_mutate_fallback_command() {
        let software = candidate_ffmpeg_args(Path::new("input.osv"), 3, 4, 2.0);
        let accelerated = with_hwaccel_auto(&software);

        assert_eq!(software[2], "-i");
        assert_eq!(software, {
            let mut expected = accelerated.clone();
            expected.drain(2..4);
            expected
        });
    }

    #[test]
    fn candidate_pass_scales_after_fps_without_changing_fallback_arguments() {
        let args = candidate_ffmpeg_args(Path::new("input.osv"), 0, 1, 8.0);
        let filter_index = args
            .iter()
            .position(|value| value == "-filter_complex")
            .unwrap();
        let filter = &args[filter_index + 1];
        assert_eq!(filter.matches("fps=8").count(), 2);
        assert!(!filter.contains("setpts="));
        assert_eq!(filter.matches("scale=").count(), 2);
        assert_eq!(filter.matches("pad=512:512").count(), 2);
        assert!(filter.contains("hstack=inputs=2:shortest=1,showinfo=checksum=0[out]"));
        assert!(filter.find("fps=8") < filter.find("scale="));
        assert!(filter.contains("format=gray"));
        assert!(args.windows(2).any(|pair| pair == ["-f", "rawvideo"]));
        assert!(args.windows(2).any(|pair| pair == ["-pix_fmt", "gray"]));
        assert_eq!(args.last().map(String::as_str), Some("pipe:1"));
        assert!(!args.iter().any(|argument| argument.contains(".jpg")));
        assert_eq!(
            CANDIDATE_IMAGE_FORMAT,
            "rawvideo-gray8-hstack-1024x512-memory"
        );
    }

    #[test]
    fn showinfo_parser_preserves_candidate_pts_in_milliseconds() {
        assert_eq!(
            parse_showinfo_timestamp_ms(
                "[Parsed_showinfo_7 @ 0x123] n:   42 pts: 9000 pts_time:1.5 duration:1"
            ),
            Some((42, 1500.0))
        );
        assert_eq!(
            parse_showinfo_timestamp_ms("[showinfo] n:0 pts:0 pts_time:-0.125"),
            Some((0, -125.0))
        );
        assert_eq!(parse_showinfo_timestamp_ms("n:0 pts_time:1.0"), None);
        assert_eq!(
            parse_showinfo_timestamp_ms("[showinfo] n:0 pts_time:N/A"),
            None
        );
    }

    fn composite_frame(left_sharp: bool, right_sharp: bool) -> Vec<u8> {
        let mut frame = vec![128u8; CANDIDATE_FRAME_BYTES];
        for y in 0..CANDIDATE_PROXY_SIZE {
            for x in 0..CANDIDATE_PROXY_SIZE {
                let checker = if ((x / 4) + (y / 4)).is_multiple_of(2) {
                    255
                } else {
                    0
                };
                if left_sharp {
                    frame[y * CANDIDATE_STREAM_WIDTH + x] = checker;
                }
                if right_sharp {
                    frame[y * CANDIDATE_STREAM_WIDTH + CANDIDATE_PROXY_SIZE + x] = checker;
                }
            }
        }
        frame
    }

    #[test]
    fn memory_selector_uses_pair_minimum_and_keeps_interval_identity() {
        let mut selector = StreamingCandidateSelector::new(2.0, 8.0, true);
        selector.push(&composite_frame(true, false)).unwrap();
        selector.push(&composite_frame(true, true)).unwrap();
        let records = selector.finish().unwrap();
        assert!(!records[0].selected);
        assert!(records[1].selected);
        assert_eq!(records[0].interval, 0);
        assert_eq!(records[1].sequence, 2);
        assert!(records[0].pair_score < records[1].pair_score);
        assert!(records.iter().all(|record| {
            record
                .lens0_source
                .to_string_lossy()
                .starts_with("memory://")
        }));
    }

    #[test]
    fn memory_selector_keeps_earliest_tie_and_starts_next_base_interval() {
        let mut selector = StreamingCandidateSelector::new(2.0, 8.0, false);
        let frame = vec![0u8; CANDIDATE_FRAME_BYTES];
        for _ in 0..5 {
            selector.push(&frame).unwrap();
        }
        let records = selector.finish().unwrap();
        let selected = records
            .iter()
            .filter(|record| record.selected)
            .map(|record| (record.interval, record.sequence))
            .collect::<Vec<_>>();
        assert_eq!(selected, vec![(0, 1), (1, 5)]);
    }

    #[test]
    fn memory_selector_prunes_interval_bests_but_keeps_first_and_last() {
        let mut selector = StreamingCandidateSelector::new(2.0, 8.0, false);
        selector
            .enable_keyframe_pruning(
                crate::extraction::KeyframePruningConfig {
                    min_rotation_deg: 90.0,
                    min_gap_ms: 0.0,
                    max_gap_ms: 10_000.0,
                    min_visual_novelty: 1.0,
                },
                None,
            )
            .unwrap();
        let frame = vec![0u8; CANDIDATE_FRAME_BYTES];
        for _ in 0..12 {
            selector.push(&frame).unwrap();
        }
        let records = selector.finish().unwrap();
        let selected = records
            .iter()
            .filter(|record| record.selected)
            .map(|record| record.sequence)
            .collect::<Vec<_>>();
        assert_eq!(selected, vec![1, 9]);
        assert_eq!(
            records[4].selection_reason.as_deref(),
            Some("belowThreshold")
        );
        assert_eq!(records[8].selection_reason.as_deref(), Some("last"));
    }

    #[test]
    fn raw_frame_reader_accepts_complete_frames_and_rejects_partial_tail() {
        let complete = vec![7u8; CANDIDATE_FRAME_BYTES];
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        read_raw_frames(std::io::Cursor::new(complete), sender);
        assert!(
            matches!(receiver.recv().unwrap(), RawFrameMessage::Frame(frame) if frame.len() == CANDIDATE_FRAME_BYTES)
        );
        assert!(matches!(receiver.recv().unwrap(), RawFrameMessage::Eof));

        let partial = vec![9u8; CANDIDATE_FRAME_BYTES + 3];
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        read_raw_frames(std::io::Cursor::new(partial), sender);
        assert!(matches!(
            receiver.recv().unwrap(),
            RawFrameMessage::Frame(_)
        ));
        assert!(
            matches!(receiver.recv().unwrap(), RawFrameMessage::Error(message) if message.contains("3/"))
        );
    }

    #[test]
    fn selected_pass_uses_balanced_zero_based_select_after_fps() {
        let expression = balanced_select_expression(&(0..128).collect::<Vec<_>>());
        assert!(expression.starts_with('\'') && expression.ends_with('\''));
        assert_eq!(expression.matches("eq(n,").count(), 128);
        let mut depth = 0usize;
        let mut max_depth = 0usize;
        for character in expression.chars() {
            match character {
                '(' => {
                    depth += 1;
                    max_depth = max_depth.max(depth);
                }
                ')' => depth -= 1,
                _ => {}
            }
        }
        assert_eq!(depth, 0);
        assert!(max_depth <= 10, "select tree should remain balanced");

        let args = selected_ffmpeg_args(
            Path::new("input.osv"),
            0,
            1,
            8.0,
            &[0, 127],
            Path::new("capture/lens0"),
            Path::new("capture/lens1"),
        );
        let filters = args
            .iter()
            .enumerate()
            .filter(|(_, value)| *value == "-vf")
            .map(|(index, _)| &args[index + 1])
            .collect::<Vec<_>>();
        assert_eq!(filters.len(), 2);
        assert!(filters.iter().all(|filter| {
            filter.starts_with("fps=8,select='")
                && filter.contains("eq(n,0)")
                && filter.contains("eq(n,127)")
        }));
    }

    #[test]
    fn second_pass_mapping_avoids_renaming_collisions_and_keeps_pairs_synced() {
        let temp = TempDir::new().unwrap();
        let decoded0 = temp.path().join("decoded/lens0");
        let decoded1 = temp.path().join("decoded/lens1");
        let mapped0 = temp.path().join("mapped/lens0");
        let mapped1 = temp.path().join("mapped/lens1");
        fs::create_dir_all(&decoded0).unwrap();
        fs::create_dir_all(&decoded1).unwrap();
        for sequence in 1..=2 {
            fs::write(decoded0.join(format!("{sequence:08}.jpg")), b"lens0").unwrap();
            fs::write(decoded1.join(format!("{sequence:08}.jpg")), b"lens1").unwrap();
        }

        map_full_res_candidates(&decoded0, &decoded1, &mapped0, &mapped1, &[2, 3]).unwrap();
        assert_eq!(
            candidate_image_names(&mapped0).unwrap(),
            BTreeSet::from(["00000002.jpg".to_owned(), "00000003.jpg".to_owned()])
        );
        assert_eq!(
            candidate_image_names(&mapped1).unwrap(),
            BTreeSet::from(["00000002.jpg".to_owned(), "00000003.jpg".to_owned()])
        );
        assert!(!decoded0.join("00000001.jpg").exists());
        assert!(!decoded1.join("00000002.jpg").exists());
    }

    #[test]
    fn empty_selected_outputs_are_valid_when_output_directories_do_not_exist() {
        let temp = TempDir::new().unwrap();
        super::verify_selected_outputs(
            &temp.path().join("images/lens0"),
            &temp.path().join("images/lens1"),
            "source000_",
            &[],
        )
        .unwrap();
    }

    #[test]
    fn stale_full_resolution_temp_dirs_are_removed_without_touching_candidates() {
        let temp = TempDir::new().unwrap();
        let candidates = temp.path().join("lens0");
        let stale = temp.path().join("full-res-interrupted");
        fs::create_dir_all(&candidates).unwrap();
        fs::create_dir_all(stale.join("decoded/lens0")).unwrap();
        fs::write(candidates.join("00000001.jpg"), b"candidate").unwrap();
        fs::write(stale.join("decoded/lens0/00000001.jpg"), b"frame").unwrap();

        cleanup_stale_full_res_dirs(temp.path()).unwrap();

        assert!(candidates.join("00000001.jpg").exists());
        assert!(!stale.exists());
    }

    #[test]
    fn obsolete_disk_candidate_cache_is_removed_without_touching_other_files() {
        let temp = TempDir::new().unwrap();
        let lens0 = temp.path().join("lens0");
        let lens1 = temp.path().join("lens1");
        fs::create_dir_all(&lens0).unwrap();
        fs::create_dir_all(&lens1).unwrap();
        fs::write(lens0.join("00000001.jpg"), b"candidate").unwrap();
        fs::write(lens1.join("00000001.jpg"), b"candidate").unwrap();
        fs::write(temp.path().join("candidates.complete.json"), b"old").unwrap();
        fs::write(temp.path().join("keep.txt"), b"keep").unwrap();

        cleanup_obsolete_candidate_cache(temp.path()).unwrap();

        assert!(!lens0.exists());
        assert!(!lens1.exists());
        assert!(!temp.path().join("candidates.complete.json").exists());
        assert_eq!(fs::read(temp.path().join("keep.txt")).unwrap(), b"keep");
    }

    #[test]
    fn scalar_selection_checkpoint_reuses_only_matching_source_and_settings() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("input.osv");
        let checkpoint = temp.path().join("selection.checkpoint.json");
        fs::write(&input, b"video-v1").unwrap();
        let mut selector = StreamingCandidateSelector::new(2.0, 8.0, false);
        selector.push(&vec![0u8; CANDIDATE_FRAME_BYTES]).unwrap();
        let records = selector.finish().unwrap();
        let selection = crate::extraction::SelectionMetadata {
            schema_version: crate::extraction::SELECTION_METADATA_SCHEMA_VERSION,
            candidate_storage: "memory_rawvideo".to_owned(),
            base_fps: 2.0,
            candidate_fps: 8.0,
            requested_dense_fps: 8.0,
            sharpness_scoring: false,
            sharpness_analysis_max_dimension: None,
            copy_selected_outputs: false,
            outputs_committed: false,
            intervals: 1,
            cancelled: false,
            selections: records,
        };
        let thresholds = crate::extraction::KeyframePruningConfig::default();
        write_candidate_selection_checkpoint(
            &checkpoint,
            &input,
            2.0,
            8.0,
            8.0,
            false,
            true,
            thresholds,
            Some("telemetry-v1".to_owned()),
            &selection,
        )
        .unwrap();

        assert!(load_candidate_selection_checkpoint(
            &checkpoint,
            &input,
            2.0,
            8.0,
            8.0,
            false,
            true,
            thresholds,
            Some("telemetry-v1"),
        )
        .is_some());
        assert!(load_candidate_selection_checkpoint(
            &checkpoint,
            &input,
            1.0,
            8.0,
            8.0,
            false,
            true,
            thresholds,
            Some("telemetry-v1"),
        )
        .is_none());
        assert!(load_candidate_selection_checkpoint(
            &checkpoint,
            &input,
            2.0,
            8.0,
            8.0,
            false,
            false,
            thresholds,
            Some("telemetry-v1"),
        )
        .is_none());
        assert!(load_candidate_selection_checkpoint(
            &checkpoint,
            &input,
            2.0,
            8.0,
            8.0,
            false,
            true,
            thresholds,
            Some("telemetry-v2"),
        )
        .is_none());
        fs::write(&input, b"video-v2-longer").unwrap();
        assert!(load_candidate_selection_checkpoint(
            &checkpoint,
            &input,
            2.0,
            8.0,
            8.0,
            false,
            true,
            thresholds,
            Some("telemetry-v1"),
        )
        .is_none());
    }

    #[test]
    fn candidate_sequences_must_match_across_both_lenses() {
        let temp = TempDir::new().unwrap();
        let lens0 = temp.path().join("lens0");
        let lens1 = temp.path().join("lens1");
        fs::create_dir_all(&lens0).unwrap();
        fs::create_dir_all(&lens1).unwrap();
        fs::write(lens0.join("00000001.jpg"), []).unwrap();
        fs::write(lens1.join("00000001.jpg"), []).unwrap();
        assert_eq!(synchronized_candidate_count(&lens0, &lens1), Ok(1));

        fs::write(lens0.join("00000002.jpg"), []).unwrap();
        let error = synchronized_candidate_count(&lens0, &lens1).unwrap_err();
        assert!(error.contains("候選序列不同步"));
        assert!(error.contains("lens0 2 張、lens1 1 張"));
    }

    #[test]
    fn cancelled_job_must_exit_before_another_stage_starts() {
        let manager = JobManager::default();
        let control = || JobControl {
            cancelled: Arc::new(AtomicBool::new(false)),
            mask_cancel: CancelToken::new(),
        };
        manager.insert("first".to_owned(), control()).unwrap();
        assert!(manager.cancel("first"));

        let error = manager.insert("second".to_owned(), control()).unwrap_err();
        assert_eq!(error, "The previous pipeline stage is still stopping");

        manager.remove("first");
        manager.insert("second".to_owned(), control()).unwrap();
    }
}
