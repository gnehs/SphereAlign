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
use rusqlite::{Connection, OpenFlags};
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
pub(crate) const ALIGN_PIPELINE_REVISION: u32 = 20;
const FEATURE_FINGERPRINT_SCHEMA_VERSION: u32 = 4;
const FEATURE_EXTRACTION_TYPE: &str = "SIFT";
const FEATURE_CAMERA_MODEL: &str = "OPENCV_FISHEYE";
const FEATURE_DEFAULT_FOCAL_LENGTH_FACTOR: f64 = 0.3;
const FEATURE_MAX_NUM_FEATURES: usize = 10_240;
const FEATURE_PEAK_THRESHOLD: f64 = 0.006;
const MATCH_MAX_NUM_MATCHES: usize = 10_240;
const CANDIDATE_PROGRESS_INTERVAL: Duration = Duration::from_millis(500);
const COLMAP_PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const COLMAP_MAX_IMAGE_ID: i64 = 2_147_483_647;
const MAX_BOOTSTRAP_INITIAL_PAIR_RETRIES: usize = 4;
const MIN_BOOTSTRAP_INITIAL_PAIR_INLIERS: usize = 100;
const PREFERRED_RIG_BOOTSTRAP_SHARED_FRAMES: usize = 3;
const CANDIDATE_SELECTION_PROGRESS_SHARE: f32 = 0.7;
const FULL_RESOLUTION_PROGRESS_SHARE: f32 = 0.2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct AlignCheckpoint {
    schema_version: u32,
    fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    feature_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    effective_mapper: Option<String>,
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
    imu_calibration_sha256: String,
    orientation_priors_sha256: String,
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
    quality_profile: &'static str,
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

struct StageRunOutput {
    artifacts: Vec<String>,
    registration: Option<RegistrationSummary>,
    capability_updates: BTreeMap<String, bool>,
}

impl StageRunOutput {
    fn plain(artifacts: Vec<String>) -> Self {
        Self {
            artifacts,
            registration: None,
            capability_updates: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegistrationSummary {
    registered: u64,
    total: u64,
}

impl RegistrationSummary {
    fn completion_message(self) -> String {
        let percentage = self.registered as f64 / self.total as f64 * 100.0;
        format!(
            "對齊處理完成：已註冊 {} / {} 組相機組影格（{percentage:.1}%）",
            self.registered, self.total
        )
    }
}

#[derive(Clone, Default)]
pub struct JobManager {
    jobs: Arc<Mutex<HashMap<String, JobControl>>>,
}

struct JobCompletionGuard {
    manager: JobManager,
    id: String,
}

impl Drop for JobCompletionGuard {
    fn drop(&mut self) {
        self.manager.remove(&self.id);
    }
}

impl JobManager {
    pub fn is_running(&self) -> bool {
        self.jobs.lock().is_ok_and(|jobs| !jobs.is_empty())
    }

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
    }
    reset_capabilities_for_stage_start(&mut manifest, &request.stage);
    project::save_manifest(&manifest)?;
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
        let _completion_guard = JobCompletionGuard {
            manager: manager.clone(),
            id: id.clone(),
        };
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
            StageName::Extract => {
                run_extract(&app, &id, &manifest, &control).map(StageRunOutput::plain)
            }
            StageName::Mask => run_mask(&app, &id, &manifest, &control).map(StageRunOutput::plain),
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
                Ok(output) => {
                    let stage_elapsed_ms = elapsed_ms(stage_started_at);
                    for (capability, enabled) in output.capability_updates {
                        manifest.capabilities.insert(capability, enabled);
                    }
                    let completed_message = if skipped_mask {
                        "未啟用 YOLO 或天空過濾，已略過遮罩階段".to_owned()
                    } else if let Some(registration) = output.registration {
                        registration.completion_message()
                    } else {
                        "處理階段已完成".to_owned()
                    };
                    let _ = project::update_stage_timed(
                        &mut manifest,
                        &stage,
                        StageStatus::Completed,
                        1.0,
                        &completed_message,
                        output.artifacts,
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
                        output.registration.map(|summary| summary.registered),
                        output.registration.map(|summary| summary.total),
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

fn reset_capabilities_for_stage_start(
    manifest: &mut project::ProjectManifest,
    stage: &StageName,
) {
    // A previous successful Align must not remain visible while a retry is
    // running, cancelled, or failed. Successful output derives the new value
    // from the effective mapper and validated prior artifacts.
    if *stage == StageName::Align {
        manifest.capabilities.insert("imuApplied".to_owned(), false);
    }
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

fn effective_mapper_matches_checkpoint(
    requested_mode: MapperMode,
    external_orientation_requested: bool,
    effective_mapper: Option<&str>,
) -> bool {
    let Some(effective_mapper) = effective_mapper else {
        return requested_mode == MapperMode::Incremental;
    };
    if external_orientation_requested {
        return effective_mapper == "external_orientation_ba";
    }
    match requested_mode {
        MapperMode::Incremental => {
            matches!(effective_mapper, "final_mapper" | "bootstrap_mapper")
        }
        MapperMode::Global => effective_mapper == "global_mapper",
        MapperMode::Auto => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColmapQualityProfile {
    Baseline,
    Tuned,
}

impl ColmapQualityProfile {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Tuned => "tuned",
        }
    }
}

fn colmap_quality_profile(settings: &Value) -> Result<ColmapQualityProfile, String> {
    match settings
        .pointer("/align/colmapQualityProfile")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("baseline")
        .to_ascii_lowercase()
        .as_str()
    {
        "baseline" => Ok(ColmapQualityProfile::Baseline),
        "tuned" => Ok(ColmapQualityProfile::Tuned),
        value => Err(format!(
            "align.colmapQualityProfile must be baseline or tuned (got {value})"
        )),
    }
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
const TEMPORAL_RESCUE_OFFSETS: [usize; 3] = [8, 12, 16];

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

#[allow(clippy::too_many_arguments)] // Pair policy inputs are intentionally explicit and independently tested.
fn conditional_temporal_pair(
    pairs: &mut BTreeSet<String>,
    left_lens: usize,
    right_lens: usize,
    left: &str,
    right: &str,
    offset: usize,
    source_motion: Option<&SourceFrameMotion>,
    calibrated_fov_overlap: Option<bool>,
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
                    && calibrated_fov_overlap.unwrap_or(true)
            }
            // Unknown filename identity must preserve the legacy graph.
            _ => true,
        }
    };
    if should_keep {
        pairs.insert(format!("lens{left_lens}/{left} lens{right_lens}/{right}"));
    }
}

fn load_calibrated_camera_orientations(root: &Path) -> Result<BTreeMap<String, [f64; 4]>, String> {
    let bundle = serde_json::from_slice::<ImuCalibrationBundle>(
        &fs::read(root.join("metadata/imu_calibration.json"))
            .map_err(|error| format!("找不到目前的 IMU calibration bundle：{error}"))?,
    )
    .map_err(|error| format!("IMU calibration bundle 格式無效：{error}"))?;
    let rig_configs = serde_json::from_slice::<Vec<RigBootstrapConfig>>(
        &fs::read(root.join("rig_config.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let camera_rotations = rig_camera_rotations(&rig_configs)?;
    let metadata = root.join("metadata");
    let mut manifest_paths = fs::read_dir(&metadata)
        .map_err(|error| error.to_string())?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("orientation_priors_source") && name.ends_with(".json")
                })
        })
        .collect::<Vec<_>>();
    manifest_paths.sort();
    let mut output = BTreeMap::new();
    for path in manifest_paths {
        let manifest = crate::orientation_constraints::OrientationPriorManifest::read_json(&path)?;
        let source_id = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("orientation_priors_"))
            .and_then(|name| name.strip_suffix(".json"))
            .ok_or_else(|| format!("orientation prior 檔名無法辨識：{}", path.display()))?;
        let source = bundle
            .sources
            .iter()
            .find(|source| source.source_id == source_id && source.model.valid)
            .ok_or_else(|| format!("{source_id} 沒有目前有效的 IMU calibration"))?;
        let current_telemetry_hash =
            optional_file_sha256(&metadata.join(format!("{source_id}_telemetry.json")))?;
        if manifest.calibration_version != bundle.calibration_version
            || manifest.source.telemetry_sha256.as_deref() != Some(source.telemetry_sha256.as_str())
            || current_telemetry_hash != source.telemetry_sha256
        {
            return Err(format!("{source_id} 的 orientation prior 已過期"));
        }
        for prior in manifest.priors {
            if source_name_from_image(&prior.rig_frame_id) != source_id
                || !root
                    .join("images/lens0")
                    .join(&prior.rig_frame_id)
                    .is_file()
            {
                return Err(format!(
                    "{source_id} 的 orientation prior 不對應目前影格：{}",
                    prior.rig_frame_id
                ));
            }
            for (lens, cam_from_rig) in camera_rotations.iter().enumerate() {
                let camera_from_world = crate::orientation_constraints::multiply_quaternions(
                    *cam_from_rig,
                    prior.rig_quaternion_wxyz,
                )
                .ok_or_else(|| "calibrated camera orientation 無效".to_owned())?;
                let world_from_camera = invert_quaternion(camera_from_world)
                    .ok_or_else(|| "calibrated camera orientation 無法反轉".to_owned())?;
                output.insert(
                    format!("lens{lens}/{}", prior.rig_frame_id),
                    world_from_camera,
                );
            }
        }
    }
    Ok(output)
}

fn calibrated_pair_overlap(
    orientations: &BTreeMap<String, [f64; 4]>,
    left_lens: usize,
    left: &str,
    right_lens: usize,
    right: &str,
) -> Option<bool> {
    if orientations.is_empty() {
        return None;
    }
    let left = orientations.get(&format!("lens{left_lens}/{left}"))?;
    let right = orientations.get(&format!("lens{right_lens}/{right}"))?;
    // Slightly wider than a mathematical hemisphere preserves DJI's seam
    // transition while still rejecting clearly disjoint long-range views.
    crate::visual_retrieval::camera_views_overlap(*left, *right, 95.0, 95.0)
}

#[cfg(test)]
fn write_rig_and_pairs(root: &Path) -> Result<u64, String> {
    write_rig_and_pairs_with_options(root, true, false, true)
}

fn write_rig_and_pairs_with_options(
    root: &Path,
    use_visual_retrieval: bool,
    use_calibrated_fov: bool,
    include_cross_source_pairs: bool,
) -> Result<u64, String> {
    let rig_config = root.join("rig_config.json");
    let unknown_default = json!([{"cameras":[
        {"image_prefix":"lens0/","ref_sensor":true},{"image_prefix":"lens1/"}]}]);
    let deprecated_colocated_default = json!([{"cameras":[
    {"image_prefix":"lens0/","ref_sensor":true},
    {
        "image_prefix":"lens1/",
        "cam_from_rig_rotation":[0.0,0.0,1.0,0.0],
        "cam_from_rig_translation":[0.0,0.0,0.0]
    }]}]);
    let should_write_unknown_default = if rig_config.is_file() {
        serde_json::from_slice::<Value>(&fs::read(&rig_config).map_err(|e| e.to_string())?)
            .is_ok_and(|config| config == deprecated_colocated_default)
    } else {
        true
    };
    if should_write_unknown_default {
        // Native Osmo 360 streams come from two physical lenses. Their baseline
        // and mounting error are not equivalent to the exact, co-located poses
        // of virtual panorama faces. Migrate only the exact synthetic default
        // emitted by older versions; preserve every other user-supplied config.
        fs::write(
            &rig_config,
            serde_json::to_vec_pretty(&unknown_default).unwrap(),
        )
        .map_err(|e| e.to_string())?;
    }
    let rig_has_complete_sensor_poses = serde_json::from_slice::<Vec<RigBootstrapConfig>>(
        &fs::read(&rig_config).map_err(|error| error.to_string())?,
    )
    .is_ok_and(|configs| rig_config_has_complete_sensor_poses(&configs));
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
    // Calibrated FOV pruning is an optimization only. Any missing, stale, or
    // malformed calibration must preserve the conservative temporal graph.
    let calibrated_camera_orientations = if use_calibrated_fov {
        load_calibrated_camera_orientations(root).unwrap_or_default()
    } else {
        BTreeMap::new()
    };
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
                conditional_temporal_pair(
                    &mut pairs,
                    0,
                    0,
                    name,
                    neighbor,
                    offset,
                    source_motion,
                    None,
                );
                conditional_temporal_pair(
                    &mut pairs,
                    1,
                    1,
                    name,
                    neighbor,
                    offset,
                    source_motion,
                    None,
                );
            }
            let overlap_01 =
                calibrated_pair_overlap(&calibrated_camera_orientations, 0, name, 1, neighbor);
            let overlap_10 =
                calibrated_pair_overlap(&calibrated_camera_orientations, 1, name, 0, neighbor);
            conditional_temporal_pair(
                &mut pairs,
                0,
                1,
                name,
                neighbor,
                offset,
                source_motion,
                overlap_01,
            );
            conditional_temporal_pair(
                &mut pairs,
                1,
                0,
                name,
                neighbor,
                offset,
                source_motion,
                overlap_10,
            );
        }
        // After the physical rig pose is calibrated, a local +1..+5 graph can
        // become permanently disconnected when a
        // short blurry or texture-poor interval prevents any new frame from
        // acquiring enough 2D-to-3D correspondences.  Add a small, fixed set
        // of longer same-sensor skip links so a later usable frame can attempt
        // registration directly against the established model. Keep these
        // edges out of unknown-rig bootstrap so speculative long links cannot
        // change the component used to estimate physical extrinsics. The fixed
        // offsets keep work O(n), stay inside one source recording, and still
        // pass through COLMAP's geometric verification.
        for offset in if rig_has_complete_sensor_poses {
            TEMPORAL_RESCUE_OFFSETS.as_slice()
        } else {
            &[]
        } {
            let Some(rescue) = names.get(index + *offset) else {
                continue;
            };
            if source_name_from_image(name) != source_name_from_image(rescue) {
                continue;
            }
            pairs.insert(format!("lens0/{name} lens0/{rescue}"));
            pairs.insert(format!("lens1/{name} lens1/{rescue}"));

            // Long cross-lens links are useful after a large rig rotation, but
            // only add them when calibrated per-frame view directions confirm
            // that the two physical fisheyes overlap.
            if calibrated_pair_overlap(&calibrated_camera_orientations, 0, name, 1, rescue)
                == Some(true)
            {
                pairs.insert(format!("lens0/{name} lens1/{rescue}"));
            }
            if calibrated_pair_overlap(&calibrated_camera_orientations, 1, name, 0, rescue)
                == Some(true)
            {
                pairs.insert(format!("lens1/{name} lens0/{rescue}"));
            }
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
    let groups: Vec<(&str, Vec<&String>)> = sources
        .into_iter()
        .map(|(source, frames)| {
            let step = (frames.len() / 20).max(1);
            (source, frames.into_iter().step_by(step).take(20).collect())
        })
        .collect();
    let retrieval_sources = groups
        .iter()
        .map(
            |(source, frames)| crate::visual_retrieval::RetrievalSource {
                source_id: (*source).to_owned(),
                anchors: frames
                    .iter()
                    .map(|name| crate::visual_retrieval::RetrievalAnchor {
                        frame_id: (*name).clone(),
                        path: root.join("images/lens0").join(name),
                        timestamp_ms: sequence_from_image_name(name).and_then(|sequence| {
                            frame_motion
                                .get(*source)
                                .and_then(|motion| motion.frames.get(&sequence))
                                .and_then(|frame| frame.timestamp_ms)
                        }),
                    })
                    .collect(),
            },
        )
        .collect::<Vec<_>>();
    let retrieval_report = (include_cross_source_pairs && use_visual_retrieval).then(|| {
        crate::visual_retrieval::retrieve_cross_source_candidates(
            &retrieval_sources,
            &crate::visual_retrieval::RetrievalConfig::default(),
        )
    });
    let use_legacy_cross_source = include_cross_source_pairs
        && retrieval_report
            .as_ref()
            .is_none_or(crate::visual_retrieval::RetrievalReport::requires_fallback);
    if use_legacy_cross_source {
        for left_index in 0..groups.len() {
            for right_index in (left_index + 1)..groups.len() {
                for left in &groups[left_index].1 {
                    for right in &groups[right_index].1 {
                        for left_lens in 0..2 {
                            for right_lens in 0..2 {
                                pairs.insert(format!(
                                    "lens{left_lens}/{left} lens{right_lens}/{right}"
                                ));
                            }
                        }
                    }
                }
            }
        }
    } else if let Some(report) = &retrieval_report {
        for candidate in report.frame_candidates() {
            for left_lens in 0..2 {
                for right_lens in 0..2 {
                    pairs.insert(format!(
                        "lens{left_lens}/{} lens{right_lens}/{}",
                        candidate.frame_a, candidate.frame_b
                    ));
                }
            }
        }
    }
    fs::create_dir_all(root.join("metadata")).map_err(|e| e.to_string())?;
    if let Some(report) = &retrieval_report {
        fs::write(
            root.join("metadata/cross_source_retrieval.json"),
            serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    }
    fs::write(
        root.join("metadata/pairs.txt"),
        pairs.into_iter().collect::<Vec<_>>().join("\n") + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(names.len() as u64)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RigBootstrapConfig {
    cameras: Vec<RigBootstrapCamera>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RigBootstrapCamera {
    image_prefix: String,
    #[serde(default)]
    ref_sensor: bool,
    cam_from_rig_rotation: Option<Vec<f64>>,
    cam_from_rig_translation: Option<Vec<f64>>,
}

fn persist_rig_config_from_database(
    root: &Path,
    database: &Path,
) -> Result<Vec<RigBootstrapConfig>, String> {
    let configs = rig_configs_from_camera_extrinsics(
        crate::colmap_priors::read_rig_camera_extrinsics(database)?,
    );
    write_json_atomic(&root.join("rig_config.json"), &configs)?;
    Ok(configs)
}

fn rig_configs_from_camera_extrinsics(
    cameras: Vec<crate::colmap_priors::RigCameraExtrinsic>,
) -> Vec<RigBootstrapConfig> {
    let mut rigs = BTreeMap::<i64, Vec<RigBootstrapCamera>>::new();
    for camera in cameras {
        rigs.entry(camera.rig_id)
            .or_default()
            .push(RigBootstrapCamera {
                image_prefix: camera.image_prefix,
                ref_sensor: camera.ref_sensor,
                cam_from_rig_rotation: (!camera.ref_sensor)
                    .then(|| camera.cam_from_rig_rotation.to_vec()),
                cam_from_rig_translation: (!camera.ref_sensor)
                    .then(|| camera.cam_from_rig_translation.to_vec()),
            });
    }
    let configs = rigs
        .into_values()
        .map(|cameras| RigBootstrapConfig { cameras })
        .collect::<Vec<_>>();
    configs
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
    /// Run one mapper with independent cameras, then convert that model to a rig.
    BootstrapThenConfigure,
}

fn rig_mapping_plan(configs: &[RigBootstrapConfig]) -> RigMappingPlan {
    if rig_config_has_complete_sensor_poses(configs) {
        RigMappingPlan::PreconfiguredSinglePass
    } else {
        RigMappingPlan::BootstrapThenConfigure
    }
}

fn mapper_still_required_after_rig_setup(plan: RigMappingPlan) -> bool {
    plan == RigMappingPlan::PreconfiguredSinglePass
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

fn complete_registered_dual_fisheye_frames(images_text: &str) -> u64 {
    let prefixes = BTreeSet::from(["lens0/".to_owned(), "lens1/".to_owned()]);
    let registered = registered_rig_image_names(images_text, &prefixes);
    let lens0 = registered
        .iter()
        .filter_map(|name| name.strip_prefix("lens0/"))
        .collect::<BTreeSet<_>>();
    let lens1 = registered
        .iter()
        .filter_map(|name| name.strip_prefix("lens1/"))
        .collect::<BTreeSet<_>>();
    lens0.intersection(&lens1).count() as u64
}

fn registration_summary_from_text_model(root: &Path, total: u64) -> Option<RegistrationSummary> {
    if total == 0 {
        return None;
    }
    let images_text = fs::read_to_string(root.join("metadata/final-model-text/images.txt")).ok()?;
    Some(RegistrationSummary {
        registered: complete_registered_dual_fisheye_frames(&images_text).min(total),
        total,
    })
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
        imu_calibration_sha256: optional_file_sha256(&root.join("metadata/imu_calibration.json"))?,
        orientation_priors_sha256: optional_file_sha256(
            &root.join("metadata/orientation_priors.json"),
        )?,
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
    quality_profile: ColmapQualityProfile,
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
        quality_profile: quality_profile.as_str(),
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
    effective_mapper: Option<&str>,
) -> Result<(), String> {
    let checkpoint = AlignCheckpoint {
        schema_version: ALIGN_CHECKPOINT_SCHEMA_VERSION,
        fingerprint: fingerprint.to_owned(),
        feature_fingerprint: Some(feature_fingerprint.to_owned()),
        effective_mapper: effective_mapper.map(str::to_owned),
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
        root.join("sparse_bootstrap_retry"),
        root.join("metadata/.align-bootstrap-text"),
        root.join("metadata/.align-configured-rig"),
        root.join("metadata/.align-configured-rig-text"),
        root.join("metadata/.align-matching-database.backup"),
        root.join("metadata/.align-matching-database.backup.partial"),
        root.join("metadata/.align-unconfigured-database.backup"),
        root.join("metadata/.align-unconfigured-database.backup.partial"),
    ];
    if !preserve_database {
        invalidate_calibrated_prior_artifacts(root)?;
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

fn invalidate_calibrated_prior_artifacts(root: &Path) -> Result<(), String> {
    let metadata = root.join("metadata");
    let mut paths = vec![
        metadata.join("global_mapper_priors.json"),
        metadata.join("imu_calibration.json"),
        metadata.join("orientation_priors.json"),
    ];
    if let Ok(entries) = fs::read_dir(&metadata) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if (name.starts_with("orientation_priors_source")
                || name.starts_with("rolling_shutter_source"))
                && name.ends_with(".json")
            {
                paths.push(path);
            }
        }
    }
    for path in paths {
        remove_align_artifact(&path)?;
    }
    Ok(())
}

#[derive(Debug)]
struct RigBootstrapModelCandidate {
    path: PathBuf,
    shared_frame_count: usize,
    registered_image_count: usize,
    summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifiedBootstrapInitialPair {
    image_id1: i64,
    image_id2: i64,
    inlier_count: usize,
    image_names: String,
    same_frame: bool,
}

fn colmap_image_pair_id(image_id1: i64, image_id2: i64) -> i64 {
    let (smaller, larger) = if image_id1 < image_id2 {
        (image_id1, image_id2)
    } else {
        (image_id2, image_id1)
    };
    smaller * COLMAP_MAX_IMAGE_ID + larger
}

fn verified_bootstrap_initial_pairs(
    database: &Path,
    configs: &[RigBootstrapConfig],
) -> Result<Vec<VerifiedBootstrapInitialPair>, String> {
    let connection = Connection::open_with_flags(database, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| format!("無法唯讀開啟 COLMAP 資料庫 {}：{error}", database.display()))?;
    let mut image_statement = connection
        .prepare("SELECT image_id, name FROM images")
        .map_err(|error| format!("無法讀取 COLMAP images 表：{error}"))?;
    let images = image_statement
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| format!("無法查詢 COLMAP images 表：{error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("無法解析 COLMAP images 表：{error}"))?
        .into_iter()
        .filter(|(image_id, _)| (0..COLMAP_MAX_IMAGE_ID).contains(image_id))
        .collect::<Vec<_>>();
    let mut geometry_statement = connection
        .prepare("SELECT pair_id, rows, config FROM two_view_geometries")
        .map_err(|error| format!("無法讀取 COLMAP two_view_geometries 表：{error}"))?;
    let geometries = geometry_statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|error| format!("無法查詢 COLMAP two_view_geometries 表：{error}"))?
        .filter_map(Result::ok)
        // COLMAP's DEGENERATE/WATERMARK/MULTIPLE configs do not provide a
        // usable two-view pose for incremental initialization. The remaining
        // calibrated, uncalibrated, planar, panoramic, mixed, and rig configs
        // still pass through COLMAP's own pose and triangulation-angle gates.
        .filter(|(_, rows, config)| {
            matches!(*config, 2 | 3 | 4 | 5 | 6 | 9)
                && *rows >= MIN_BOOTSTRAP_INITIAL_PAIR_INLIERS as i64
        })
        .map(|(pair_id, rows, _)| (pair_id, rows as usize))
        .collect::<HashMap<_, _>>();

    let mut candidates = Vec::new();
    for config in configs {
        let Some(reference) = config.cameras.iter().find(|camera| camera.ref_sensor) else {
            continue;
        };
        let reference_images = images
            .iter()
            .filter_map(|(image_id, name)| {
                name.strip_prefix(&reference.image_prefix)
                    .map(|frame| (*image_id, frame))
            })
            .collect::<Vec<_>>();
        for camera in config
            .cameras
            .iter()
            .filter(|camera| !camera.ref_sensor && !camera.has_explicit_pose())
        {
            let camera_images = images.iter().filter_map(|(image_id, name)| {
                name.strip_prefix(&camera.image_prefix)
                    .map(|frame| (*image_id, frame))
            });
            for (image_id2, camera_frame) in camera_images {
                for (image_id1, reference_frame) in &reference_images {
                    let pair_id = colmap_image_pair_id(*image_id1, image_id2);
                    let Some(inlier_count) = geometries.get(&pair_id) else {
                        continue;
                    };
                    candidates.push(VerifiedBootstrapInitialPair {
                        image_id1: *image_id1,
                        image_id2,
                        inlier_count: *inlier_count,
                        image_names: format!("{reference_frame} ↔ {camera_frame}"),
                        same_frame: reference_frame == &camera_frame,
                    });
                }
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .same_frame
            .cmp(&left.same_frame)
            .then_with(|| right.inlier_count.cmp(&left.inlier_count))
            .then_with(|| left.image_names.cmp(&right.image_names))
            .then_with(|| left.image_id1.cmp(&right.image_id1))
            .then_with(|| left.image_id2.cmp(&right.image_id2))
    });
    candidates.dedup_by_key(|candidate| {
        (
            candidate.image_id1.min(candidate.image_id2),
            candidate.image_id1.max(candidate.image_id2),
        )
    });
    let mut selected = candidates
        .iter()
        .filter(|candidate| candidate.same_frame)
        .take(MAX_BOOTSTRAP_INITIAL_PAIR_RETRIES / 2)
        .cloned()
        .collect::<Vec<_>>();
    selected.extend(
        candidates
            .iter()
            .filter(|candidate| !candidate.same_frame)
            .take(MAX_BOOTSTRAP_INITIAL_PAIR_RETRIES - selected.len())
            .cloned(),
    );
    if selected.len() < MAX_BOOTSTRAP_INITIAL_PAIR_RETRIES {
        for candidate in candidates {
            if selected.len() == MAX_BOOTSTRAP_INITIAL_PAIR_RETRIES {
                break;
            }
            if !selected.iter().any(|existing| {
                existing.image_id1 == candidate.image_id1
                    && existing.image_id2 == candidate.image_id2
            }) {
                selected.push(candidate);
            }
        }
    }
    Ok(selected)
}

fn select_best_bootstrap_candidate(
    candidates: Vec<RigBootstrapModelCandidate>,
) -> Option<RigBootstrapModelCandidate> {
    let require_robust_shared_coverage = candidates
        .iter()
        .any(|candidate| candidate.shared_frame_count >= PREFERRED_RIG_BOOTSTRAP_SHARED_FRAMES);
    let mut best = None;
    for candidate in candidates {
        if require_robust_shared_coverage
            && candidate.shared_frame_count < PREFERRED_RIG_BOOTSTRAP_SHARED_FRAMES
        {
            continue;
        }
        let should_replace = best
            .as_ref()
            .is_none_or(|current: &RigBootstrapModelCandidate| {
                // Rig calibration is only constrained by frames where all
                // required sensors co-register. Prefer that evidence first;
                // single-sensor images are useful only as a tie-breaker.
                (
                    candidate.shared_frame_count,
                    candidate.registered_image_count,
                ) > (
                    current.shared_frame_count,
                    current.registered_image_count,
                )
            });
        if should_replace {
            best = Some(candidate);
        }
    }
    best
}

fn rig_bootstrap_shared_frame_count(
    configs: &[RigBootstrapConfig],
    registered_images: &BTreeSet<String>,
) -> usize {
    configs
        .iter()
        .map(|config| {
            let Some(reference) = config.cameras.iter().find(|camera| camera.ref_sensor) else {
                return 0;
            };
            let reference_frames = registered_images
                .iter()
                .filter_map(|name| name.strip_prefix(&reference.image_prefix))
                .collect::<BTreeSet<_>>();
            config
                .cameras
                .iter()
                .filter(|camera| !camera.ref_sensor && !camera.has_explicit_pose())
                .map(|camera| {
                    let camera_frames = registered_images
                        .iter()
                        .filter_map(|name| name.strip_prefix(&camera.image_prefix))
                        .collect::<BTreeSet<_>>();
                    reference_frames.intersection(&camera_frames).count()
                })
                .sum::<usize>()
        })
        .sum()
}

fn sparse_model_directories(root: &Path) -> Vec<PathBuf> {
    let mut models = fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let index = entry.file_name().to_str()?.parse::<u64>().ok()?;
            path.is_dir().then_some((index, path))
        })
        .collect::<Vec<_>>();
    models.sort_by_key(|(index, _)| *index);
    models.into_iter().map(|(_, path)| path).collect()
}

fn select_colmap_bootstrap_for_rig(
    app: &AppHandle,
    id: &str,
    colmap: &Path,
    root: &Path,
    bootstrap_root: &Path,
    control: &JobControl,
) -> Result<PathBuf, String> {
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

    let models = sparse_model_directories(bootstrap_root);
    if models.is_empty() {
        return Err("COLMAP 初始建模未產生任何 sparse 子模型".to_owned());
    }

    let text_model = root.join("metadata/.align-bootstrap-text");
    let mut candidates = Vec::new();
    let mut failures = Vec::new();
    for bootstrap_model in models {
        remove_align_artifact(&text_model)?;
        fs::create_dir_all(&text_model)
            .map_err(|error| format!("無法建立 COLMAP 初始模型驗證資料夾：{error}"))?;
        let conversion_result = run_child(
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
            Ok(registered_rig_image_names(&images_text, &prefixes))
        });
        let cleanup_result = remove_align_artifact(&text_model);
        let registered_images = match (conversion_result, cleanup_result) {
            (Ok(registered), Ok(())) => registered,
            (Err(error), Ok(())) => return Err(error),
            (Ok(_), Err(error)) => return Err(error),
            (Err(error), Err(cleanup_error)) => {
                return Err(format!("{error}；{cleanup_error}"));
            }
        };
        let label = bootstrap_model
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("?")
            .to_owned();
        match validate_rig_bootstrap_registration(&configs, &registered_images) {
            Ok(summary) => {
                let shared_frame_count =
                    rig_bootstrap_shared_frame_count(&configs, &registered_images);
                let candidate = RigBootstrapModelCandidate {
                    path: bootstrap_model,
                    shared_frame_count,
                    registered_image_count: registered_images.len(),
                    summary,
                };
                emit_log(
                    app,
                    id,
                    "info",
                    format!(
                        "bootstrap 子模型 {label} 候選：{} 張已註冊影像、{} 組共同影格",
                        candidate.registered_image_count, candidate.shared_frame_count
                    ),
                );
                candidates.push(candidate);
            }
            Err(error) => {
                emit_log(
                    app,
                    id,
                    "warning",
                    format!("bootstrap 子模型 {label} 不合格：{error}"),
                );
                failures.push(format!("子模型 {label}：{error}"));
            }
        }
    }
    remove_align_artifact(&text_model)?;

    let Some(best) = select_best_bootstrap_candidate(candidates) else {
        let mut failure_summary = failures
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("；");
        if failures.len() > 5 {
            failure_summary.push_str(&format!("；另有 {} 個不合格子模型", failures.len() - 5));
        }
        return Err(format!(
            "COLMAP 產生了初始子模型，但沒有任何一個能估計相機組外參。{}",
            failure_summary
        ));
    };
    emit_log(
        app,
        id,
        "info",
        format!(
            "已從 COLMAP 初始子模型選出 {}（{} 張已註冊影像、{} 組共同影格）：{}",
            best.path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("?"),
            best.registered_image_count,
            best.shared_frame_count,
            best.summary
        ),
    );
    Ok(best.path)
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
    quality_profile: ColmapQualityProfile,
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
        if quality_profile == ColmapQualityProfile::Tuned {
            FEATURE_MAX_NUM_FEATURES.to_string()
        } else {
            "8192".into()
        },
        "--FeatureExtraction.use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--FeatureExtraction.gpu_index".into(),
        gpu_index.to_owned(),
    ];
    if quality_profile == ColmapQualityProfile::Tuned {
        // Osmo 360 imagery often contains low-contrast wall and distant
        // detail. Retain more of those stable extrema than COLMAP's default
        // without enabling affine/DSP SIFT, which would disable the fast GPU
        // extraction path on common Windows builds.
        args.extend([
            "--SiftExtraction.peak_threshold".into(),
            FEATURE_PEAK_THRESHOLD.to_string(),
        ]);
    }
    if use_masks {
        args.push("--ImageReader.mask_path".into());
        args.push(root.join("masks_colmap").to_string_lossy().into_owned());
    }
    args
}

fn matches_importer_args(
    root: &Path,
    db: &Path,
    use_gpu: bool,
    gpu_index: &str,
    quality_profile: ColmapQualityProfile,
) -> Vec<String> {
    let mut args = vec![
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
        "--FeatureMatching.max_num_matches".into(),
        if quality_profile == ColmapQualityProfile::Tuned {
            MATCH_MAX_NUM_MATCHES.to_string()
        } else {
            "8192".into()
        },
    ];
    if quality_profile == ColmapQualityProfile::Tuned {
        args.extend([
        // Spend modest extra RANSAC work on retrieval/cross-source pairs instead of
        // accepting a weaker model merely because the default trial budget ran
        // out in repetitive indoor scenes.
            "--TwoViewGeometry.confidence".into(),
            "0.999".into(),
            "--TwoViewGeometry.max_num_trials".into(),
            "15000".into(),
        ]);
    }
    args
}

fn commit_configured_rig_model(configured_model: &Path, sparse_root: &Path) -> Result<(), String> {
    let required = ["rigs", "cameras", "frames", "images", "points3D"];
    if !configured_model.is_dir()
        || required.iter().any(|name| {
            !["bin", "txt"].iter().any(|extension| {
                fs::metadata(configured_model.join(format!("{name}.{extension}")))
                    .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
            })
        })
    {
        return Err(format!(
            "COLMAP rig configurator 輸出不完整：{}",
            configured_model.display()
        ));
    }

    remove_align_artifact(sparse_root)?;
    fs::create_dir_all(sparse_root)
        .map_err(|error| format!("無法建立最終 sparse 模型資料夾：{error}"))?;
    let target = sparse_root.join("0");
    if let Err(error) = fs::rename(configured_model, &target) {
        let cleanup = remove_align_artifact(sparse_root)
            .err()
            .map(|value| format!("；{value}"))
            .unwrap_or_default();
        return Err(format!(
            "無法提交 rig configurator 模型 {}：{error}{cleanup}",
            target.display()
        ));
    }
    Ok(())
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
    options: MapperOptions,
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
        if options.multiple_models {
            "1".into()
        } else {
            "0".into()
        },
        "--Mapper.ba_local_backend".into(),
        "CERES".into(),
        "--Mapper.ba_global_backend".into(),
        "CERES".into(),
        "--Mapper.ba_use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--Mapper.ba_gpu_index".into(),
        gpu_index.to_owned(),
    ];
    if options.multiple_models {
        // Unknown-rig calibration only needs one jointly registered frame.
        // COLMAP otherwise discards secondary models smaller than 10 images,
        // which can hide the only component that satisfies that official rig
        // calibration precondition. Validation below still rejects components
        // without both cameras and a shared frame name.
        args.extend(["--Mapper.min_model_size".into(), "2".into()]);
    }
    if let Some((image_id1, image_id2)) = options.initial_image_pair {
        args.extend([
            "--Mapper.init_image_id1".into(),
            image_id1.to_string(),
            "--Mapper.init_image_id2".into(),
            image_id2.to_string(),
            "--Mapper.init_num_trials".into(),
            "1".into(),
        ]);
    }
    if options.disable_sensor_refinement {
        args.push("--Mapper.ba_refine_sensor_from_rig".into());
        args.push("0".into());
    }
    args
}

#[derive(Debug, Clone, Copy)]
struct MapperOptions {
    multiple_models: bool,
    initial_image_pair: Option<(i64, i64)>,
    disable_sensor_refinement: bool,
}

#[derive(Debug, Clone, Copy)]
struct GlobalMapperOptions {
    use_gravity_prior: bool,
    fixed_rotation_ba: bool,
    disable_sensor_refinement: bool,
    quality_refinement: bool,
}

fn global_mapper_args(
    db: &Path,
    images: &Path,
    output: &Path,
    use_gpu: bool,
    gpu_index: &str,
    options: GlobalMapperOptions,
) -> Vec<String> {
    let mut args = vec![
        "global_mapper".into(),
        "--database_path".into(),
        db.to_string_lossy().into_owned(),
        "--image_path".into(),
        images.to_string_lossy().into_owned(),
        "--output_path".into(),
        output.to_string_lossy().into_owned(),
        // Only a rig supplied with complete extrinsics before this alignment is
        // treated as pre-calibrated. A pose inferred by the bootstrap pass is
        // an initialization and remains refinable in the final reconstruction.
        "--GlobalMapper.refine_sensor_from_rig".into(),
        if options.disable_sensor_refinement {
            "0".into()
        } else {
            "1".into()
        },
        "--GlobalMapper.ra_use_gravity".into(),
        if options.use_gravity_prior {
            "1".into()
        } else {
            "0".into()
        },
        "--GlobalMapper.ra_use_stratified".into(),
        if options.use_gravity_prior {
            "1".into()
        } else {
            "0".into()
        },
        "--GlobalMapper.gp_use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--GlobalMapper.gp_gpu_index".into(),
        mapper_gpu_index(gpu_index).to_owned(),
        "--GlobalMapper.ba_ceres_use_gpu".into(),
        if use_gpu { "1".into() } else { "0".into() },
        "--GlobalMapper.ba_ceres_gpu_index".into(),
        mapper_gpu_index(gpu_index).to_owned(),
    ];
    if options.quality_refinement {
        // COLMAP's global defaults favor broad registration (15 px track
        // completion/merge gates). Tighten the final geometry and give global
        // positioning plus BA enough iterations for dual-fisheye video.
        args.extend([
            "--GlobalMapper.gp_max_num_iterations".into(),
            "120".into(),
            "--GlobalMapper.tri_complete_max_reproj_error".into(),
            "8".into(),
            "--GlobalMapper.tri_merge_max_reproj_error".into(),
            "8".into(),
            "--GlobalMapper.max_normalized_reproj_error".into(),
            "0.008".into(),
        ]);
    }
    if options.fixed_rotation_ba {
        args.extend([
            "--GlobalMapper.ba_skip_joint_optimization_stage".into(),
            "1".into(),
        ]);
    }
    args
}

fn view_graph_calibrator_args(database: &Path) -> Vec<String> {
    vec![
        "view_graph_calibrator".into(),
        "--database_path".into(),
        database.to_string_lossy().into_owned(),
        // Keep COLMAP 4.1.1's documented cross-validation and relative-pose
        // refresh enabled.  The command writes an estimated focal length and
        // sets has_prior_focal_length only after calibration succeeds.
        "--cross_validate_prior_focal_lengths".into(),
        "1".into(),
        "--reestimate_relative_pose".into(),
        "1".into(),
    ]
}

fn view_graph_calibrator_retry_args(database: &Path) -> Vec<String> {
    let mut args = view_graph_calibrator_args(database);
    // This is a bounded recovery for sparse/pruned graphs. The result still
    // has to pass parameter-shape, non-default focal, flag, and 100% camera
    // round-trip validation before it can become a global prior.
    args.extend([
        "--min_calibrated_pair_ratio".into(),
        "0.25".into(),
        "--default_random_seed".into(),
        "0".into(),
    ]);
    args
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceImuCalibration {
    source_id: String,
    visual_model: String,
    telemetry_sha256: String,
    visual_sample_count: usize,
    telemetry_sample_count: usize,
    model: crate::imu_calibration::CalibrationModel,
    candidates: Vec<SourceImuCalibrationCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceImuCalibrationCandidate {
    visual_model: String,
    visual_sample_count: usize,
    model: crate::imu_calibration::CalibrationModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImuCalibrationBundle {
    schema_version: u32,
    calibration_version: String,
    valid_source_count: usize,
    source_count: usize,
    sources: Vec<SourceImuCalibration>,
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} 沒有 parent directory", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.partial");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::rename(&temporary, path).map_err(|error| error.to_string())
}

fn export_colmap_text_model(
    app: &AppHandle,
    id: &str,
    colmap: &Path,
    input: &Path,
    output: &Path,
    control: &JobControl,
) -> Result<(), String> {
    remove_align_artifact(output)?;
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    run_child(
        app,
        id,
        colmap,
        &[
            "model_converter".into(),
            "--input_path".into(),
            input.to_string_lossy().into_owned(),
            "--output_path".into(),
            output.to_string_lossy().into_owned(),
            "--output_type".into(),
            "TXT".into(),
        ],
        control,
    )
}

fn visual_samples_by_source(
    root: &Path,
    model: &crate::reconstruction_benchmark::ColmapTextModel,
) -> BTreeMap<String, Vec<crate::imu_calibration::VisualRotationSample>> {
    let motion = load_frame_motion_metadata(root);
    let mut sources = BTreeMap::<String, Vec<_>>::new();
    for image in &model.images {
        let Some(name) = image.name.strip_prefix("lens0/") else {
            continue;
        };
        let source = source_name_from_image(name);
        let Some(sequence) = sequence_from_image_name(name) else {
            continue;
        };
        let Some(timestamp_ms) = motion
            .get(source)
            .and_then(|source| source.frames.get(&sequence))
            .and_then(|frame| frame.timestamp_ms)
            .filter(|timestamp| timestamp.is_finite())
        else {
            continue;
        };
        sources.entry(source.to_owned()).or_default().push(
            crate::imu_calibration::VisualRotationSample {
                timestamp_ms,
                rotation_wxyz: image.qvec_camera_from_world,
            },
        );
    }
    for samples in sources.values_mut() {
        samples.sort_by(|left, right| left.timestamp_ms.total_cmp(&right.timestamp_ms));
    }
    sources
}

fn calibrate_imu_sources(
    root: &Path,
    text_models: &[(String, crate::reconstruction_benchmark::ColmapTextModel)],
) -> Result<ImuCalibrationBundle, String> {
    // Disconnected recordings often become separate bootstrap components.
    // Evaluate each source against every component independently, then select
    // the strongest valid calibration without mixing unrelated coordinates or
    // assuming the component used to derive rig extrinsics contains all sources.
    let mut visual = BTreeMap::<
        String,
        Vec<(String, Vec<crate::imu_calibration::VisualRotationSample>)>,
    >::new();
    for (model_name, text_model) in text_models {
        for (source_id, samples) in visual_samples_by_source(root, text_model) {
            visual
                .entry(source_id)
                .or_default()
                .push((model_name.clone(), samples));
        }
    }
    let mut sources = Vec::new();
    for (source_id, visual_candidates) in visual {
        let telemetry_path = root
            .join("metadata")
            .join(format!("{source_id}_telemetry.json"));
        let telemetry_sha256 = optional_file_sha256(&telemetry_path)?;
        let normalized = match telemetry::read_normalized_telemetry(&telemetry_path) {
            Ok(value) => value,
            Err(error) => {
                let (visual_model, visual_sample_count) = visual_candidates
                    .iter()
                    .max_by_key(|(_, samples)| samples.len())
                    .map(|(name, samples)| (name.clone(), samples.len()))
                    .unwrap_or_else(|| ("none".to_owned(), 0));
                sources.push(SourceImuCalibration {
                    source_id,
                    visual_model,
                    telemetry_sha256,
                    visual_sample_count,
                    telemetry_sample_count: 0,
                    model: crate::imu_calibration::CalibrationModel::invalid(error),
                    candidates: Vec::new(),
                });
                continue;
            }
        };
        let telemetry_sample_count = normalized.fused_attitude.len();
        let candidates = visual_candidates
            .into_iter()
            .map(|(visual_model, visual_samples)| {
                let model = crate::imu_calibration::estimate_calibration(
                    &visual_samples,
                    &normalized.fused_attitude,
                    crate::imu_calibration::CalibrationConfig::default(),
                )
                .unwrap_or_else(|error| {
                    crate::imu_calibration::CalibrationModel::invalid(error.to_string())
                });
                SourceImuCalibrationCandidate {
                    visual_model,
                    visual_sample_count: visual_samples.len(),
                    model,
                }
            })
            .collect::<Vec<_>>();
        let selected = candidates
            .iter()
            .filter(|candidate| candidate.model.valid)
            .max_by(|left, right| {
                left.model
                    .paired_sample_count
                    .cmp(&right.model.paired_sample_count)
                    .then_with(|| {
                        right
                            .model
                            .residual_deg
                            .unwrap_or(f64::INFINITY)
                            .total_cmp(&left.model.residual_deg.unwrap_or(f64::INFINITY))
                    })
            })
            .or_else(|| candidates.iter().max_by_key(|candidate| candidate.visual_sample_count))
            .ok_or_else(|| format!("{source_id} 沒有 visual calibration candidate"))?;
        let visual_model = selected.visual_model.clone();
        let visual_sample_count = selected.visual_sample_count;
        let model = selected.model.clone();
        sources.push(SourceImuCalibration {
            source_id,
            visual_model,
            telemetry_sha256,
            visual_sample_count,
            telemetry_sample_count,
            model,
            candidates,
        });
    }
    let valid_source_count = sources.iter().filter(|source| source.model.valid).count();
    // Orientation manifests are rig-frame values, so the calibration identity
    // must change when either telemetry/hand-eye results or rig extrinsics do.
    let version_payload = serde_json::to_vec(&json!({
        "sources": &sources,
        "rigConfigSha256": file_sha256(&root.join("rig_config.json"))?,
    }))
    .map_err(|error| error.to_string())?;
    let digest = sha256_hex(&version_payload);
    let bundle = ImuCalibrationBundle {
        schema_version: crate::imu_calibration::IMU_CALIBRATION_SCHEMA_VERSION,
        calibration_version: format!("imu-hand-eye-v1-{}", &digest[..16]),
        valid_source_count,
        source_count: sources.len(),
        sources,
    };
    write_json_atomic(&root.join("metadata/imu_calibration.json"), &bundle)?;
    Ok(bundle)
}

fn invert_quaternion(value: [f64; 4]) -> Option<[f64; 4]> {
    telemetry::normalize_quaternion(value).map(|value| [value[0], -value[1], -value[2], -value[3]])
}

fn rig_camera_rotations(configs: &[RigBootstrapConfig]) -> Result<[[f64; 4]; 2], String> {
    if configs.len() != 1 {
        return Err(format!(
            "orientation/FOV priors require exactly one rig, found {} distinct rig groups",
            configs.len()
        ));
    }
    let mut rotations = [[1.0, 0.0, 0.0, 0.0]; 2];
    let mut found = [false; 2];
    for camera in configs.iter().flat_map(|config| &config.cameras) {
        let lens = if camera.image_prefix == "lens0/" {
            Some(0)
        } else if camera.image_prefix == "lens1/" {
            Some(1)
        } else {
            None
        };
        let Some(lens) = lens else { continue };
        let rotation = if camera.ref_sensor {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            let values = camera
                .cam_from_rig_rotation
                .as_ref()
                .filter(|values| values.len() == 4)
                .ok_or_else(|| format!("lens{lens} 缺少 cam_from_rig rotation"))?;
            telemetry::normalize_quaternion([values[0], values[1], values[2], values[3]])
                .ok_or_else(|| format!("lens{lens} cam_from_rig rotation 無效"))?
        };
        rotations[lens] = rotation;
        found[lens] = true;
    }
    if found != [true, true] {
        return Err("rig_config.json 必須同時包含 lens0/ 與 lens1/".to_owned());
    }
    Ok(rotations)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct OrientationPriorIndex {
    schema_version: u32,
    format: &'static str,
    calibration_version: String,
    source_manifests: Vec<String>,
    valid_source_count: usize,
}

fn source_frames_with_timestamps(root: &Path, source_id: &str) -> Vec<(String, f64)> {
    let motion = load_frame_motion_metadata(root);
    let Some(source_motion) = motion.get(source_id) else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(root.join("images/lens0")) else {
        return Vec::new();
    };
    let mut frames = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_owned();
            if source_name_from_image(&name) != source_id {
                return None;
            }
            let sequence = sequence_from_image_name(&name)?;
            let timestamp_ms = source_motion.frames.get(&sequence)?.timestamp_ms?;
            timestamp_ms.is_finite().then_some((name, timestamp_ms))
        })
        .collect::<Vec<_>>();
    frames.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    frames
}

fn build_orientation_and_gravity_priors(
    root: &Path,
    bundle: &ImuCalibrationBundle,
    rig_configs: &[RigBootstrapConfig],
    export_rolling_shutter: bool,
) -> Result<Vec<crate::colmap_priors::GravityPriorInput>, String> {
    let camera_rotations = rig_camera_rotations(rig_configs)?;
    let rig_from_camera0 = invert_quaternion(camera_rotations[0])
        .ok_or_else(|| "lens0 cam_from_rig quaternion 無效".to_owned())?;
    let mut gravity = Vec::new();
    let mut source_manifests = Vec::new();
    for source in &bundle.sources {
        if !source.model.valid {
            continue;
        }
        let time_offset_ms = source
            .model
            .time_offset_ms
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("{} 缺少有效 time offset", source.source_id))?;
        let sensor_to_camera = source
            .model
            .sensor_to_camera_quaternion
            .ok_or_else(|| format!("{} 缺少 sensor-to-camera quaternion", source.source_id))?;
        let sensor_to_rig = crate::orientation_constraints::multiply_quaternions(
            rig_from_camera0,
            sensor_to_camera,
        )
        .ok_or_else(|| format!("{} sensor-to-rig quaternion 無效", source.source_id))?;
        let normalized = telemetry::read_normalized_telemetry(
            &root
                .join("metadata")
                .join(format!("{}_telemetry.json", source.source_id)),
        )?;
        let timeline = normalized.attitude_timeline();
        let frames = source_frames_with_timestamps(root, &source.source_id);
        if frames.is_empty() {
            continue;
        }
        let invert_telemetry = matches!(
            source.model.telemetry_orientation_convention,
            Some(crate::imu_calibration::TelemetryOrientationConvention::Inverted)
        );
        let provenance = crate::orientation_constraints::OrientationProvenance {
            source: "dji_fused_attitude_calibrated".to_owned(),
            telemetry_sha256: Some(source.telemetry_sha256.clone()),
            parser_revision: Some(normalized.parser_revision.clone()),
            timestamp_source: "ffmpeg_pts_exposure_center".to_owned(),
            coordinate_transform: bundle.calibration_version.clone(),
        };
        let residuals = vec![source.model.residual_deg.unwrap_or(8.0); frames.len()];
        let priors = crate::orientation_constraints::build_orientation_priors(
            &frames,
            &timeline,
            time_offset_ms,
            invert_telemetry,
            sensor_to_rig,
            None,
            None,
            &provenance,
            &bundle.calibration_version,
            Some(&residuals),
        )?;
        let validation = crate::orientation_constraints::OrientationValidationConfig {
            min_coverage_ratio: MIN_GLOBAL_GRAVITY_COVERAGE_RATIO,
            max_residual_angle_deg: 8.0,
            max_abs_time_offset_ms: 500.0,
            max_timestamp_error_ms: 1.0,
            expected_calibration_version: Some(bundle.calibration_version.clone()),
        };
        let manifest = crate::orientation_constraints::OrientationPriorManifest::new(
            priors,
            &frames.iter().map(|frame| frame.1).collect::<Vec<_>>(),
            timeline.coverage().map(|(start, end)| [start, end]),
            time_offset_ms,
            bundle.calibration_version.clone(),
            provenance,
            &validation,
        )?;
        let manifest_name = format!("orientation_priors_{}.json", source.source_id);
        manifest.write_json(&root.join("metadata").join(&manifest_name))?;
        source_manifests.push(manifest_name);
        if export_rolling_shutter {
            let mut trajectories = Vec::new();
            if let Some(readout_time_ms) = normalized
                .sensor_readout_time_ms
                .filter(|value| value.is_finite() && *value >= 0.0)
            {
                for (frame_id, timestamp_ms) in &frames {
                    let image_path = root.join("images/lens0").join(frame_id);
                    let Ok((_, image_height)) = image::image_dimensions(&image_path) else {
                        continue;
                    };
                    trajectories.push(
                        crate::orientation_constraints::sample_rolling_shutter_trajectory(
                            frame_id,
                            *timestamp_ms,
                            readout_time_ms,
                            time_offset_ms,
                            invert_telemetry,
                            sensor_to_rig,
                            image_height,
                            9,
                            &timeline,
                        )?,
                    );
                }
            }
            write_json_atomic(
                &root
                    .join("metadata")
                    .join(format!("rolling_shutter_{}.json", source.source_id)),
                &json!({
                    "schemaVersion": 1,
                    "sourceId": source.source_id,
                    "readoutTimeMs": normalized.sensor_readout_time_ms,
                    "available": normalized.sensor_readout_time_ms.is_some(),
                    "pixelsModified": false,
                    "trajectories": trajectories,
                }),
            )?;
        }
        for prior in &manifest.priors {
            for (lens, cam_from_rig) in camera_rotations.iter().enumerate() {
                let camera_from_world = crate::orientation_constraints::multiply_quaternions(
                    *cam_from_rig,
                    prior.rig_quaternion_wxyz,
                )
                .ok_or_else(|| format!("{} lens{lens} pose 無效", prior.rig_frame_id))?;
                let gravity_camera =
                    crate::visual_retrieval::rotate_vector(camera_from_world, [0.0, 0.0, 1.0])
                        .ok_or_else(|| format!("{} lens{lens} gravity 無效", prior.rig_frame_id))?;
                gravity.push(crate::colmap_priors::GravityPriorInput {
                    image_name: format!("lens{lens}/{}", prior.rig_frame_id),
                    gravity: gravity_camera,
                });
            }
        }
    }
    let canonical_path = root.join("metadata/orientation_priors.json");
    if source_manifests.len() == 1 {
        let manifest = crate::orientation_constraints::OrientationPriorManifest::read_json(
            &root.join("metadata").join(&source_manifests[0]),
        )?;
        manifest.write_json(&canonical_path)?;
    } else {
        let index = OrientationPriorIndex {
            schema_version: 1,
            format: "gs360.orientation-prior-index/v1",
            calibration_version: bundle.calibration_version.clone(),
            valid_source_count: source_manifests.len(),
            source_manifests,
        };
        write_json_atomic(&canonical_path, &index)?;
    }
    Ok(gravity)
}

fn representative_time_offset(bundle: &ImuCalibrationBundle) -> Option<f64> {
    let mut offsets = bundle
        .sources
        .iter()
        .filter(|source| source.model.valid)
        .filter_map(|source| source.model.time_offset_ms)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    offsets.sort_by(f64::total_cmp);
    offsets.get(offsets.len() / 2).copied()
}

#[allow(clippy::too_many_arguments)] // This orchestration boundary carries progress and cancellation context.
fn prepare_global_mapper_priors_from_seed(
    app: &AppHandle,
    id: &str,
    root: &Path,
    colmap: &Path,
    seed_model: &Path,
    database: &Path,
    rig_configs: &[RigBootstrapConfig],
    export_rolling_shutter: bool,
    control: &JobControl,
) -> Result<crate::colmap_priors::GlobalMapperPriorReport, String> {
    let text_model_dir = root.join("metadata/final-model-text");
    export_colmap_text_model(app, id, colmap, seed_model, &text_model_dir, control)?;
    let text_model = crate::reconstruction_benchmark::read_colmap_text_model(&text_model_dir)?;
    let mut calibration_models = vec![("selected-seed".to_owned(), text_model)];
    let component_text_root = root.join("metadata/imu-bootstrap-components");
    remove_align_artifact(&component_text_root)?;
    for component in sparse_model_directories(&root.join("sparse_bootstrap")) {
        let component_name = component
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_owned();
        let output = component_text_root.join(&component_name);
        match export_colmap_text_model(app, id, colmap, &component, &output, control) {
            Ok(()) => {
                let model = crate::reconstruction_benchmark::read_colmap_text_model(&output)?;
                calibration_models.push((format!("bootstrap/{component_name}"), model));
            }
            Err(error) if cancelled_error(&error, control) => return Err(error),
            Err(error) => emit_log(
                app,
                id,
                "warning",
                format!("略過無法匯出的 bootstrap component {component_name}：{error}"),
            ),
        }
    }
    let bundle = calibrate_imu_sources(root, &calibration_models)?;
    if bundle.valid_source_count == 0 {
        return Err("沒有來源通過時間偏移與 rotational hand-eye 校正".to_owned());
    }
    let gravity =
        build_orientation_and_gravity_priors(root, &bundle, rig_configs, export_rolling_shutter)?;
    if gravity.is_empty() {
        return Err("校正通過但沒有影格落在 fused attitude coverage 內".to_owned());
    }
    let focal = crate::colmap_priors::read_focal_prior_inputs(database, "view_graph_calibrator")?;
    let calibration = crate::colmap_priors::PriorCalibrationMetadata {
        calibration_version: Some(bundle.calibration_version.clone()),
        time_offset_ms: representative_time_offset(&bundle),
        require_complete_rig_extrinsics: true,
    };
    let report = crate::colmap_priors::inject_global_mapper_priors(
        database,
        &focal,
        &gravity,
        &calibration,
    )?;
    if !report.marker.focal_prior_valid || !report.marker.gravity_prior_valid {
        return Err(format!(
            "prior coverage 不足：focal {:.1}%、gravity {:.1}%",
            report.marker.focal_coverage_ratio * 100.0,
            report.marker.gravity_coverage_ratio * 100.0
        ));
    }
    crate::colmap_priors::write_global_mapper_prior_marker(
        &root.join("metadata/global_mapper_priors.json"),
        &report,
    )?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn refresh_calibrated_pair_matches(
    app: &AppHandle,
    id: &str,
    root: &Path,
    colmap: &Path,
    database: &Path,
    gpu_index: &str,
    use_gpu: bool,
    use_calibrated_fov: bool,
    quality_profile: ColmapQualityProfile,
    control: &JobControl,
) -> Result<bool, String> {
    let pairs_path = root.join("metadata/pairs.txt");
    let original_pairs = fs::read(&pairs_path).map_err(|error| error.to_string())?;
    let original_hash = sha256_hex(&original_pairs);
    let database_backup = root.join("metadata/.align-pre-calibrated-pairs.backup");
    remove_align_artifact(&database_backup)?;
    // Create the database rollback point before pairs.txt can change. A
    // backup failure therefore leaves both halves of the match graph intact.
    create_colmap_database_backup(database, &database_backup)?;
    // Retrieval descriptors were already computed for the original graph.
    // Regenerate only source-local temporal/FOV pairs, and preserve only the
    // original cross-source retrieval edges. Unioning the entire old graph
    // would resurrect local pairs deliberately removed by calibrated FOV;
    // letting the generator fall back would also inject a legacy anchor grid.
    let calibrated_pairs = (|| {
        write_rig_and_pairs_with_options(root, false, use_calibrated_fov, false)?;
        let refreshed_pairs = fs::read(&pairs_path).map_err(|error| error.to_string())?;
        let original_cross_source_pairs = cross_source_pair_lines(&original_pairs)?;
        let calibrated_pairs = merge_pair_lists(&original_cross_source_pairs, &refreshed_pairs)?;
        fs::write(&pairs_path, &calibrated_pairs).map_err(|error| error.to_string())?;
        Ok::<_, String>(calibrated_pairs)
    })();
    let calibrated_pairs = match calibrated_pairs {
        Ok(pairs) => pairs,
        Err(error) => {
            return match rollback_calibrated_pair_transaction(
                database,
                &database_backup,
                &pairs_path,
                &original_pairs,
            ) {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(format!(
                    "{error}；calibrated pair transaction rollback failed: {rollback_error}"
                )),
            };
        }
    };
    if sha256_hex(&calibrated_pairs) == original_hash {
        remove_align_artifact(&database_backup)?;
        return Ok(false);
    }
    let result = clear_matching_cache(database)
        .map_err(|error| error.to_string())
        .and_then(|()| {
            let args =
                matches_importer_args(root, database, use_gpu, gpu_index, quality_profile);
            run_child(app, id, colmap, &args, control)
        });
    match result {
        Ok(()) => {
            remove_align_artifact(&database_backup)?;
            Ok(true)
        }
        Err(error) => {
            match rollback_calibrated_pair_transaction(
                database,
                &database_backup,
                &pairs_path,
                &original_pairs,
            ) {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(format!(
                    "{error}；calibrated pair transaction rollback failed: {rollback_error}"
                )),
            }
        }
    }
}

const GLOBAL_CANDIDATE_MIN_POINT_RATIO: f64 = 0.25;
const GLOBAL_CANDIDATE_MIN_TRACK_RATIO: f64 = 0.60;
const GLOBAL_CANDIDATE_MAX_REPROJECTION_RATIO: f64 = 1.50;
const GLOBAL_CANDIDATE_MAX_REPROJECTION_INCREASE_PX: f64 = 0.25;
const GLOBAL_CANDIDATE_MAX_COMPONENT_COVERAGE_DROP: f64 = 0.10;
const GLOBAL_CANDIDATE_MAX_ADDITIONAL_COMPONENTS: u64 = 2;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GlobalCandidateQualityMetrics {
    complete_registered_rig_frames: u64,
    #[serde(flatten)]
    model: crate::reconstruction_benchmark::ColmapModelQualityMetrics,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GlobalCandidateQualityGate {
    accepted: bool,
    seed: GlobalCandidateQualityMetrics,
    candidate: GlobalCandidateQualityMetrics,
    issues: Vec<String>,
}

fn global_candidate_quality_metrics(
    text_model_dir: &Path,
) -> Result<GlobalCandidateQualityMetrics, String> {
    let model = crate::reconstruction_benchmark::read_colmap_text_model(text_model_dir)?;
    let images_text = fs::read_to_string(text_model_dir.join("images.txt"))
        .map_err(|error| error.to_string())?;
    Ok(GlobalCandidateQualityMetrics {
        complete_registered_rig_frames: complete_registered_dual_fisheye_frames(&images_text),
        model: crate::reconstruction_benchmark::colmap_model_quality_metrics(&model),
    })
}

fn evaluate_global_candidate_quality(
    seed: GlobalCandidateQualityMetrics,
    candidate: GlobalCandidateQualityMetrics,
) -> GlobalCandidateQualityGate {
    let mut issues = Vec::new();
    if candidate.complete_registered_rig_frames < seed.complete_registered_rig_frames {
        issues.push(format!(
            "complete-rig coverage regressed: {} < {}",
            candidate.complete_registered_rig_frames, seed.complete_registered_rig_frames
        ));
    }
    if candidate.model.largest_component_coverage_ratio
        + GLOBAL_CANDIDATE_MAX_COMPONENT_COVERAGE_DROP
        < seed.model.largest_component_coverage_ratio
    {
        issues.push(format!(
            "largest-component coverage regressed: {:.3} < {:.3} - {:.3}",
            candidate.model.largest_component_coverage_ratio,
            seed.model.largest_component_coverage_ratio,
            GLOBAL_CANDIDATE_MAX_COMPONENT_COVERAGE_DROP
        ));
    }
    if (candidate.model.points3d_count as f64)
        < seed.model.points3d_count as f64 * GLOBAL_CANDIDATE_MIN_POINT_RATIO
    {
        issues.push(format!(
            "point support regressed: {} < {} * {:.2}",
            candidate.model.points3d_count,
            seed.model.points3d_count,
            GLOBAL_CANDIDATE_MIN_POINT_RATIO
        ));
    }
    match (
        seed.model.median_track_length,
        candidate.model.median_track_length,
    ) {
        (Some(seed_track), Some(candidate_track))
            if candidate_track < seed_track * GLOBAL_CANDIDATE_MIN_TRACK_RATIO =>
        {
            issues.push(format!(
                "median track support regressed: {candidate_track:.3} < {seed_track:.3} * {:.2}",
                GLOBAL_CANDIDATE_MIN_TRACK_RATIO
            ));
        }
        (Some(_), None) => issues.push("candidate median track support is unavailable".to_owned()),
        _ => {}
    }
    match (
        seed.model.median_reprojection_error_px,
        candidate.model.median_reprojection_error_px,
    ) {
        (Some(seed_error), Some(candidate_error)) => {
            let maximum = (seed_error * GLOBAL_CANDIDATE_MAX_REPROJECTION_RATIO)
                .max(seed_error + GLOBAL_CANDIDATE_MAX_REPROJECTION_INCREASE_PX);
            if candidate_error > maximum {
                issues.push(format!(
                    "median reprojection error regressed: {candidate_error:.3} > {maximum:.3} px"
                ));
            }
        }
        (Some(_), None) => {
            issues.push("candidate median reprojection error is unavailable".to_owned())
        }
        _ => {}
    }
    if candidate.model.connected_component_count
        > seed
            .model
            .connected_component_count
            .saturating_add(GLOBAL_CANDIDATE_MAX_ADDITIONAL_COMPONENTS)
    {
        issues.push(format!(
            "connected components increased abnormally: {} > {} + {}",
            candidate.model.connected_component_count,
            seed.model.connected_component_count,
            GLOBAL_CANDIDATE_MAX_ADDITIONAL_COMPONENTS
        ));
    }
    GlobalCandidateQualityGate {
        accepted: issues.is_empty(),
        seed,
        candidate,
        issues,
    }
}

fn rollback_calibrated_pair_transaction(
    database: &Path,
    database_backup: &Path,
    pairs_path: &Path,
    original_pairs: &[u8],
) -> Result<(), String> {
    let mut errors = Vec::new();
    if let Err(error) = restore_colmap_database_backup(database, database_backup) {
        errors.push(error);
    }
    if let Err(error) = fs::write(pairs_path, original_pairs) {
        errors.push(format!("無法還原 pairs.txt：{error}"));
    }
    if errors.is_empty() {
        remove_align_artifact(database_backup)?;
        Ok(())
    } else {
        Err(format!(
            "{}；備份保留於 {}",
            errors.join("；"),
            database_backup.display()
        ))
    }
}

fn calibrated_pair_refresh_requires_fail_closed(error: &str) -> bool {
    error.contains("calibrated pair transaction rollback failed")
}

fn merge_pair_lists(first: &[u8], second: &[u8]) -> Result<Vec<u8>, String> {
    let first = std::str::from_utf8(first).map_err(|error| error.to_string())?;
    let second = std::str::from_utf8(second).map_err(|error| error.to_string())?;
    let pairs = first
        .lines()
        .chain(second.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<BTreeSet<_>>();
    let mut merged = pairs.into_iter().collect::<Vec<_>>().join("\n").into_bytes();
    if !merged.is_empty() {
        merged.push(b'\n');
    }
    Ok(merged)
}

fn cross_source_pair_lines(input: &[u8]) -> Result<Vec<u8>, String> {
    let input = std::str::from_utf8(input).map_err(|error| error.to_string())?;
    let mut output = Vec::new();
    for line in input.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut images = line.split_whitespace();
        let left = images
            .next()
            .ok_or_else(|| format!("invalid pair line: {line}"))?;
        let right = images
            .next()
            .ok_or_else(|| format!("invalid pair line: {line}"))?;
        if images.next().is_some() {
            return Err(format!("invalid pair line: {line}"));
        }
        let left_name = left
            .split_once('/')
            .map(|(_, name)| name)
            .ok_or_else(|| format!("invalid COLMAP image name in pair: {left}"))?;
        let right_name = right
            .split_once('/')
            .map(|(_, name)| name)
            .ok_or_else(|| format!("invalid COLMAP image name in pair: {right}"))?;
        if source_name_from_image(left_name) != source_name_from_image(right_name) {
            output.extend_from_slice(line.as_bytes());
            output.push(b'\n');
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)] // External solver invocation keeps all validated paths explicit.
fn run_external_orientation_ba_candidate(
    app: &AppHandle,
    id: &str,
    root: &Path,
    executable: &Path,
    colmap: &Path,
    database: &Path,
    sparse: &Path,
    fixed_rotation: bool,
    control: &JobControl,
) -> Result<(), String> {
    if !executable.is_file() {
        return Err(format!(
            "找不到 orientation BA 執行檔：{}",
            executable.display()
        ));
    }
    if control.cancelled.load(Ordering::SeqCst) {
        return Err("cancelled before external orientation BA capability probe".to_owned());
    }
    let capability_output = silent_command(executable)
        .args(crate::orientation_constraints::external_orientation_ba_capability_args())
        .output()
        .map_err(|error| format!("無法執行 orientation BA capability probe：{error}"))?;
    if !capability_output.status.success() {
        return Err(format!(
            "orientation BA capability probe 失敗：{}",
            String::from_utf8_lossy(&capability_output.stderr).trim()
        ));
    }
    let capability = crate::orientation_constraints::parse_external_orientation_ba_capability(
        &String::from_utf8_lossy(&capability_output.stdout),
    )?;
    capability.validate()?;
    let manifest_path = root.join("metadata/orientation_priors.json");
    crate::orientation_constraints::OrientationPriorManifest::read_json(&manifest_path)?;
    let candidate = root.join("sparse_orientation_candidate");
    let output_database = root.join("metadata/orientation-ba-database.db");
    remove_align_artifact(&candidate)?;
    remove_colmap_database_artifacts(&output_database)?;
    fs::create_dir_all(candidate.join("0")).map_err(|error| error.to_string())?;
    let args = crate::orientation_constraints::external_orientation_ba_model_args(
        &manifest_path,
        database,
        &output_database,
        &sparse.join("0"),
        &candidate.join("0"),
        fixed_rotation,
    )?;
    run_child(app, id, executable, &args, control)?;
    if !sparse_rig_model_exists(&candidate) {
        return Err("orientation BA 未產生可驗證的 rig sparse model".to_owned());
    }
    validate_colmap_configured_rig_model(app, id, colmap, root, &candidate.join("0"), control)?;
    let visual_backup = root.join("sparse_visual_backup");
    remove_align_artifact(&visual_backup)?;
    fs::rename(sparse, &visual_backup)
        .map_err(|error| format!("無法保存 visual sparse model：{error}"))?;
    if let Err(error) = fs::rename(&candidate, sparse) {
        fs::rename(&visual_backup, sparse).map_err(|restore_error| {
            format!("orientation BA 模型提交失敗：{error}；visual 模型還原也失敗：{restore_error}")
        })?;
        return Err(format!("orientation BA 模型提交失敗：{error}"));
    }
    Ok(())
}

fn has_valid_global_mapper_priors(root: &Path, require_gravity: bool) -> bool {
    // Treat only the marker emitted after database round-trip validation as
    // authoritative. This prevents an interrupted calibration or the default
    // focal 0.3 initialization from enabling global mapping on a later run.
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
    fixed_rotation_ba: bool,
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
    if fixed_rotation_ba && !capabilities.global_mapper_fixed_rotation_ba {
        return Some("指定的 global_mapper 缺少 fixed-rotation/joint BA stage 選項".to_owned());
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
) -> Result<StageRunOutput, String> {
    let align_started = Instant::now();
    let mut phase_durations_ms = BTreeMap::<String, f64>::new();
    let colmap = crate::doctor::resolve_colmap(custom_colmap_path)?;
    let root = PathBuf::from(&manifest.output_path);
    let gpu_index = parse_gpu_index(&manifest.settings)?;
    let requested_gpu = setting_bool(&manifest.settings, "/align/useGpu", true);
    let requested_mapper_mode = mapper_mode(&manifest.settings)?;
    let quality_profile = colmap_quality_profile(&manifest.settings)?;
    let use_gravity_prior = setting_bool(&manifest.settings, "/align/useGravityPrior", false);
    let fixed_rotation_ba = setting_bool(&manifest.settings, "/align/fixedRotationBa", false);
    let use_visual_retrieval = setting_bool(&manifest.settings, "/align/useVisualRetrieval", true);
    let use_calibrated_fov_pairs =
        setting_bool(&manifest.settings, "/align/useCalibratedFovPairs", true);
    let pair_graph_started = Instant::now();
    let rig_frame_count =
        write_rig_and_pairs_with_options(
            &root,
            use_visual_retrieval,
            use_calibrated_fov_pairs,
            true,
        )?;
    phase_durations_ms.insert(
        "pairGraph".to_owned(),
        pair_graph_started.elapsed().as_secs_f64() * 1000.0,
    );
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
    let mut rig_configs = serde_json::from_slice::<Vec<RigBootstrapConfig>>(
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
    let mut mapper_mode = match requested_mapper_mode {
        MapperMode::Incremental => MapperMode::Incremental,
        MapperMode::Global => {
            if let Some(error) = global_mapper_prerequisite_error(
                &root,
                rig_preconfigured,
                &colmap_capabilities,
                use_gravity_prior,
                fixed_rotation_ba,
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
                fixed_rotation_ba,
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
    let feature_fingerprint =
        build_feature_fingerprint(&root, &colmap_version, use_masks, quality_profile)?;
    let checkpoint_present = checkpoint_path.exists();
    let checkpoint = load_align_checkpoint(&checkpoint_path);
    let external_orientation_requested = manifest
        .settings
        .pointer("/align/orientationPriorExecutable")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let expected_effective_mapper = if external_orientation_requested {
        "external_orientation_ba"
    } else if mapper_mode == MapperMode::Global {
        "global_mapper"
    } else {
        "final_mapper"
    };
    let checkpoint_matches = !force_rebuild
        && checkpoint.as_ref().is_some_and(|value| {
            value.fingerprint == fingerprint
                && effective_mapper_matches_checkpoint(
                    mapper_mode,
                    external_orientation_requested,
                    value.effective_mapper.as_deref(),
                )
        });
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
        let reused_effective_mapper = checkpoint
            .as_ref()
            .and_then(|value| value.effective_mapper.as_deref())
            .unwrap_or(expected_effective_mapper);
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
        return Ok(StageRunOutput {
            artifacts: vec![
                db.to_string_lossy().into_owned(),
                root.join("rig_config.json").to_string_lossy().into_owned(),
                sparse.to_string_lossy().into_owned(),
            ],
            registration: registration_summary_from_text_model(&root, rig_frame_count),
            capability_updates: BTreeMap::from([(
                "imuApplied".to_owned(),
                (reused_effective_mapper == "global_mapper"
                    && use_gravity_prior
                    && has_valid_global_mapper_priors(&root, true))
                    || reused_effective_mapper == "external_orientation_ba",
            )]),
        });
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
        if !checkpoint_matches {
            // Calibration products depend on telemetry, rig configuration,
            // image timing, pair policy, and parser revisions in the align
            // fingerprint. Never carry them across a fingerprint change even
            // when the SIFT feature cache itself remains reusable.
            invalidate_calibrated_prior_artifacts(&root)?;
        }
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
        } else if preserve_database && rig_mapping_plan == RigMappingPlan::BootstrapThenConfigure {
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
        write_align_checkpoint(
            &checkpoint_path,
            &fingerprint,
            &feature_fingerprint,
            false,
            None,
        )?;
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
    let feature_started = Instant::now();
    let feature_gpu_args =
        feature_extractor_args(&root, &db, true, &gpu_index, use_masks, quality_profile);
    let feature_cpu_args =
        feature_extractor_args(&root, &db, false, &gpu_index, use_masks, quality_profile);
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
                invalidate_calibrated_prior_artifacts(&root)?;
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
    phase_durations_ms.insert(
        "featureExtraction".to_owned(),
        feature_started.elapsed().as_secs_f64() * 1000.0,
    );
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
    let matching_started = Instant::now();
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
    let matching_gpu_args =
        matches_importer_args(&root, &db, true, &gpu_index, quality_profile);
    let matching_cpu_args =
        matches_importer_args(&root, &db, false, &gpu_index, quality_profile);
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
    phase_durations_ms.insert(
        "matching".to_owned(),
        matching_started.elapsed().as_secs_f64() * 1000.0,
    );
    let calibrate_focal_prior =
        setting_bool(&manifest.settings, "/align/calibrateFocalPrior", true);
    if calibrate_focal_prior && !has_valid_global_mapper_priors(&root, false) {
        if colmap_capabilities.view_graph_calibrator {
            let calibration_backup = root.join("metadata/.align-focal-calibration.backup");
            remove_align_artifact(&calibration_backup)?;
            create_colmap_database_backup(&db, &calibration_backup)?;
            emit_log(
                app,
                id,
                "info",
                "正在以 view graph 校正 focal length；只有成功結果才會標記為 focal prior",
            );
            let result = run_child(app, id, &colmap, &view_graph_calibrator_args(&db), control);
            match result {
                Ok(()) => {
                    match crate::colmap_priors::read_focal_prior_report(
                        &db,
                        "view_graph_calibrator",
                    ) {
                        Ok(report) if report.focal_prior_valid => {
                            remove_align_artifact(&calibration_backup)?;
                            emit_log(
                                app,
                                id,
                                "info",
                                format!(
                                    "view graph focal calibration 已驗證（{:.1}% cameras）",
                                    report.focal_coverage_ratio * 100.0
                                ),
                            );
                        }
                        Ok(report) => {
                            restore_colmap_database_backup(&db, &calibration_backup)?;
                            remove_align_artifact(&calibration_backup)?;
                            emit_log(
                                app,
                                id,
                                "warning",
                                format!(
                                    "view graph focal coverage 只有 {:.1}%，已還原資料庫",
                                    report.focal_coverage_ratio * 100.0
                                ),
                            );
                        }
                        Err(error) => {
                            restore_colmap_database_backup(&db, &calibration_backup)?;
                            remove_align_artifact(&calibration_backup)?;
                            emit_log(
                                app,
                                id,
                                "warning",
                                format!(
                                    "view graph focal round-trip 驗證失敗，已還原資料庫：{error}"
                                ),
                            );
                        }
                    }
                }
                Err(error) => {
                    restore_colmap_database_backup(&db, &calibration_backup)?;
                    remove_align_artifact(&calibration_backup)?;
                    emit_log(
                        app,
                        id,
                        "warning",
                        format!(
                            "view graph focal calibration 失敗，已還原資料庫並繼續 incremental：{error}"
                        ),
                    );
                }
            }
        } else {
            emit_log(
                app,
                id,
                "warning",
                "目前 COLMAP 缺少 view_graph_calibrator，無法自動建立可信 focal prior",
            );
        }
    }
    if mapper_mode == MapperMode::Global
        && !has_valid_global_mapper_priors(&root, use_gravity_prior)
    {
        let reason = "特徵資料庫重建後已驗證的 global prior marker 不再有效";
        if requested_mapper_mode == MapperMode::Global {
            return Err(format!(
                "align.mapperMode=global 的前提在 mapping 前失效：{reason}；請先用 auto 建立新的 calibration seed"
            ));
        }
        mapper_mode = MapperMode::Incremental;
        emit_log(
            app,
            id,
            "warning",
            format!("mapperMode=auto：{reason}，先建立 incremental calibration seed"),
        );
    }
    let mapping_started = Instant::now();
    if rig_mapping_plan == RigMappingPlan::BootstrapThenConfigure {
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
                MapperOptions {
                    multiple_models: true,
                    initial_image_pair: None,
                    disable_sensor_refinement: false,
                },
            );
            let bootstrap_cpu_args = mapper_args(
                &db,
                &root.join("images"),
                &bootstrap,
                false,
                &mapper_gpu_index,
                MapperOptions {
                    multiple_models: true,
                    initial_image_pair: None,
                    disable_sensor_refinement: false,
                },
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
        let selected_bootstrap = match select_colmap_bootstrap_for_rig(
            app, id, &colmap, &root, &bootstrap, control,
        ) {
            Ok(model) => model,
            Err(initial_error) => {
                let initial_pairs = verified_bootstrap_initial_pairs(&db, &rig_configs)?;
                if initial_pairs.is_empty() {
                    return Err(format!(
                        "{initial_error}；資料庫中沒有達到 {} 個幾何 inliers 的跨鏡 pair 可供安全重試",
                        MIN_BOOTSTRAP_INITIAL_PAIR_INLIERS
                    ));
                }
                emit_log(
                    app,
                    id,
                    "warning",
                    format!(
                        "自動 bootstrap 子模型皆無共同影格；將依序嘗試 {} 組已驗證的跨鏡初始 pair",
                        initial_pairs.len()
                    ),
                );
                let retry_root = root.join("sparse_bootstrap_retry");
                let mut retry_failures = Vec::new();
                let mut selected = None;
                for pair in initial_pairs {
                    remove_align_artifact(&retry_root)?;
                    fs::create_dir_all(&retry_root)
                        .map_err(|error| format!("無法建立 bootstrap 重試資料夾：{error}"))?;
                    emit_log(
                        app,
                        id,
                        "info",
                        format!(
                            "bootstrap 重試跨鏡影像 {}（image {} ↔ {}，{} inliers）",
                            pair.image_names, pair.image_id1, pair.image_id2, pair.inlier_count
                        ),
                    );
                    let retry_options = MapperOptions {
                        multiple_models: false,
                        initial_image_pair: Some((pair.image_id1, pair.image_id2)),
                        disable_sensor_refinement: false,
                    };
                    let retry_gpu_args = mapper_args(
                        &db,
                        &root.join("images"),
                        &retry_root,
                        mapper_gpu,
                        &mapper_gpu_index,
                        retry_options,
                    );
                    let retry_cpu_args = mapper_args(
                        &db,
                        &root.join("images"),
                        &retry_root,
                        false,
                        &mapper_gpu_index,
                        retry_options,
                    );
                    let run_result = run_mapper_with_gpu_fallback(
                        app,
                        id,
                        &colmap,
                        &retry_root,
                        &retry_gpu_args,
                        &retry_cpu_args,
                        mapper_gpu,
                        "bootstrap_mapper_retry",
                        control,
                        || {},
                        |_| {},
                    );
                    let attempt = run_result.and_then(|()| {
                        select_colmap_bootstrap_for_rig(
                            app,
                            id,
                            &colmap,
                            &root,
                            &retry_root,
                            control,
                        )
                    });
                    match attempt {
                        Ok(model) => {
                            selected = Some(model);
                            break;
                        }
                        Err(error) if cancelled_error(&error, control) => return Err(error),
                        Err(error) => retry_failures.push(format!("{}：{error}", pair.image_names)),
                    }
                }
                selected.ok_or_else(|| {
                    format!(
                        "{initial_error}；已驗證跨鏡初始 pair 的有界重試仍失敗：{}",
                        retry_failures.join("；")
                    )
                })?
            }
        };
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
                selected_bootstrap.to_string_lossy().into_owned(),
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
        let calibrated_sensor_count = match configure_result {
            Ok(count) => {
                commit_configured_rig_model(&configured_bootstrap, &sparse)?;
                count
            }
            Err(error) => {
                let configured_cleanup_detail = remove_align_artifact(&configured_bootstrap)
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
        };
        rig_configs = persist_rig_config_from_database(&root, &db)?;
        let post_rig_focal_report = if calibrate_focal_prior {
            crate::colmap_priors::read_focal_prior_report(
                &db,
                "view_graph_calibrator",
            )
            .ok()
            .filter(|report| report.focal_prior_valid)
        } else {
            None
        };
        if let Some(report) = post_rig_focal_report {
            emit_log(
                app,
                id,
                "info",
                format!(
                    "rig configurator 後仍保有 {:.1}% verified focal priors",
                    report.focal_coverage_ratio * 100.0
                ),
            );
        } else if calibrate_focal_prior && colmap_capabilities.view_graph_calibrator {
            // Some COLMAP builds can leave an otherwise calibrated database
            // unmarked before bootstrap. Retry once after rig configuration,
            // then accept only the same strict DB round-trip validation.
            let retry_backup = root.join("metadata/.align-post-rig-focal.backup");
            remove_align_artifact(&retry_backup)?;
            create_colmap_database_backup(&db, &retry_backup)?;
            emit_log(
                app,
                id,
                "info",
                "正在 rig configurator 後有界重試 view graph focal calibration",
            );
            let retry_result = run_child(
                app,
                id,
                &colmap,
                &view_graph_calibrator_retry_args(&db),
                control,
            )
            .and_then(|()| {
                crate::colmap_priors::read_focal_prior_report(
                    &db,
                    "view_graph_calibrator",
                )
            });
            match retry_result {
                Ok(report) if report.focal_prior_valid => {
                    remove_align_artifact(&retry_backup)?;
                    emit_log(
                        app,
                        id,
                        "info",
                        format!(
                            "post-rig focal calibration 已驗證（{:.1}% cameras）",
                            report.focal_coverage_ratio * 100.0
                        ),
                    );
                }
                Ok(report) => {
                    restore_colmap_database_backup(&db, &retry_backup)?;
                    remove_align_artifact(&retry_backup)?;
                    emit_log(
                        app,
                        id,
                        "warning",
                        format!(
                            "post-rig focal coverage 只有 {:.1}%，已還原資料庫",
                            report.focal_coverage_ratio * 100.0
                        ),
                    );
                }
                Err(error) if cancelled_error(&error, control) => return Err(error),
                Err(error) => {
                    restore_colmap_database_backup(&db, &retry_backup)?;
                    remove_align_artifact(&retry_backup)?;
                    emit_log(
                        app,
                        id,
                        "warning",
                        format!(
                            "post-rig focal round-trip 驗證失敗，已還原資料庫：{error}"
                        ),
                    );
                }
            }
        }
        if !rig_config_has_complete_sensor_poses(&rig_configs) {
            return Err("rig_configurator 完成但無法持久化完整 sensor_from_rig 外參".to_owned());
        }
        emit_log(
            app,
            id,
            "info",
            format!("已驗證 {calibrated_sensor_count} 個 non-reference sensor 的相機組外參"),
        );
        emit_log(
            app,
            id,
            "info",
            "相機組外參完成後直接沿用已驗證模型；不再改寫 matching graph 或啟動第二次 mapper",
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
    let mut effective_final_mapper_component = "bootstrap_mapper";
    if mapper_still_required_after_rig_setup(rig_mapping_plan) {
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
            GlobalMapperOptions {
                use_gravity_prior,
                fixed_rotation_ba,
                disable_sensor_refinement: rig_preconfigured,
                quality_refinement: quality_profile == ColmapQualityProfile::Tuned,
            },
        )
    } else {
        mapper_args(
            &db,
            &root.join("images"),
            &sparse,
            final_mapper_gpu,
            &mapper_gpu_index,
            MapperOptions {
                multiple_models: false,
                initial_image_pair: None,
                disable_sensor_refinement: rig_preconfigured,
            },
        )
    };
    let final_cpu_args = if mapper_mode == MapperMode::Global {
        global_mapper_args(
            &db,
            &root.join("images"),
            &sparse,
            false,
            &gpu_index,
            GlobalMapperOptions {
                use_gravity_prior,
                fixed_rotation_ba,
                disable_sensor_refinement: rig_preconfigured,
                quality_refinement: quality_profile == ColmapQualityProfile::Tuned,
            },
        )
    } else {
        mapper_args(
            &db,
            &root.join("images"),
            &sparse,
            false,
            &mapper_gpu_index,
            MapperOptions {
                multiple_models: false,
                initial_image_pair: None,
                disable_sensor_refinement: rig_preconfigured,
            },
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
        effective_final_mapper_component = final_mapper_component;
    } else {
        if !sparse_rig_model_exists(&sparse) {
            return Err(
                "COLMAP rig configurator 完成但未提交含 rigs/frames 的 sparse model".to_owned(),
            );
        }
        emit_progress_detailed(
            app,
            id,
            &StageName::Align,
            "final-mapping",
            4.0 / 5.0,
            "沿用第一次 mapper 的 rig 模型",
            "running",
            false,
            None,
            None,
            Some("bootstrap_mapper".to_owned()),
            None,
        );
        emit_log(
            app,
            id,
            "info",
            "已直接提交 rig configurator 轉換後的 bootstrap 模型；不再從零執行第二次 mapper",
        );
    }
    let rig_extrinsics_ready = rig_config_has_complete_sensor_poses(&rig_configs);
    let auto_calibrate_telemetry =
        setting_bool(&manifest.settings, "/align/autoCalibrateTelemetry", true);
    if requested_mapper_mode == MapperMode::Auto
        && mapper_mode == MapperMode::Incremental
        && auto_calibrate_telemetry
        && rig_extrinsics_ready
        && colmap_capabilities.global_mapper
        && colmap_capabilities.global_mapper_gravity
    {
        emit_log(
            app,
            id,
            "info",
            "正在從 incremental 種子模型估計 IMU 時間偏移與 rotational hand-eye calibration",
        );
        let candidate_audit = root.join("metadata/global_mapper_candidate.json");
        write_json_atomic(
            &candidate_audit,
            &json!({
                "schemaVersion": 1,
                "attempted": false,
                "status": "calibrating_priors",
                "gravityPriorRequested": use_gravity_prior,
            }),
        )?;
        match prepare_global_mapper_priors_from_seed(
            app,
            id,
            &root,
            &colmap,
            &sparse.join("0"),
            &db,
            &rig_configs,
            setting_bool(
                &manifest.settings,
                "/align/exportRollingShutterTrajectory",
                false,
            ),
            control,
        ) {
            Ok(report) => {
                emit_log(
                    app,
                    id,
                    "info",
                    format!(
                        "IMU/focal prior 驗證通過：gravity coverage {:.1}%、focal coverage {:.1}%",
                        report.marker.gravity_coverage_ratio * 100.0,
                        report.marker.focal_coverage_ratio * 100.0
                    ),
                );
                match refresh_calibrated_pair_matches(
                    app,
                    id,
                    &root,
                    &colmap,
                    &db,
                    &gpu_index,
                    feature_matching_gpu,
                    use_calibrated_fov_pairs,
                    quality_profile,
                    control,
                ) {
                    Ok(true) => emit_log(
                        app,
                        id,
                        "info",
                        "已依 calibrated FOV overlap 更新 pairs 並重新 matching",
                    ),
                    Ok(false) => {}
                    Err(error) if calibrated_pair_refresh_requires_fail_closed(&error) => {
                        write_json_atomic(
                            &candidate_audit,
                            &json!({
                                "schemaVersion": 1,
                                "attempted": false,
                                "status": "pair_refresh_rollback_failed",
                                "gravityPriorRequested": use_gravity_prior,
                                "effectiveMapper": "bootstrap_mapper",
                                "error": &error,
                            }),
                        )?;
                        return Err(format!(
                            "calibrated pair graph rollback failed; refusing to run global mapper: {error}"
                        ));
                    }
                    Err(error) => emit_log(
                        app,
                        id,
                        "warning",
                        format!(
                            "calibrated FOV pair refresh 失敗，已還原原 matching graph：{error}"
                        ),
                    ),
                }
                let fixed_validation =
                    crate::orientation_constraints::validate_fixed_rotation_global_ba(
                        crate::orientation_constraints::FixedRotationBackend::StockGlobalMapperGravity,
                        &crate::orientation_constraints::FixedRotationGlobalBaPrerequisites {
                            global_mapper_available: colmap_capabilities.global_mapper,
                            fixed_rotation_option_available: colmap_capabilities
                                .global_mapper_fixed_rotation_ba,
                            rig_extrinsics_complete: rig_extrinsics_ready,
                            focal_prior_valid: report.marker.focal_prior_valid,
                            orientation_manifest_valid: false,
                            gravity_prior_valid: report.marker.gravity_prior_valid,
                            gravity_coverage_ratio: Some(
                                report.marker.gravity_coverage_ratio,
                            ),
                            external_capability: None,
                        },
                    );
                let enable_fixed_rotation = fixed_rotation_ba && fixed_validation.valid;
                if fixed_rotation_ba && !enable_fixed_rotation {
                    emit_log(
                        app,
                        id,
                        "warning",
                        format!(
                            "固定旋轉 BA 前提不完整，維持 joint optimization：{}",
                            fixed_validation.issues.join("；")
                        ),
                    );
                }
                let global_candidate = root.join("sparse_global_candidate");
                let seed_quality = global_candidate_quality_metrics(
                    &root.join("metadata/final-model-text"),
                )?;
                let seed_complete_rigs = seed_quality.complete_registered_rig_frames;
                write_json_atomic(
                    &candidate_audit,
                    &json!({
                        "schemaVersion": 1,
                        "attempted": true,
                        "status": "running",
                        "gravityPriorRequested": use_gravity_prior,
                        "seedCompleteRegisteredRigFrames": seed_complete_rigs,
                        "seedQuality": &seed_quality,
                    }),
                )?;
                remove_align_artifact(&global_candidate)?;
                fs::create_dir_all(&global_candidate).map_err(|error| error.to_string())?;
                let global_gpu = requested_gpu
                    && colmap_capabilities.cuda_build
                    && colmap_capabilities.global_mapper_gp_gpu
                    && colmap_capabilities.global_mapper_ba_gpu;
                let gpu_args = global_mapper_args(
                    &db,
                    &root.join("images"),
                    &global_candidate,
                    true,
                    &gpu_index,
                    GlobalMapperOptions {
                        use_gravity_prior,
                        fixed_rotation_ba: enable_fixed_rotation,
                        disable_sensor_refinement: rig_extrinsics_ready,
                        quality_refinement: quality_profile == ColmapQualityProfile::Tuned,
                    },
                );
                let cpu_args = global_mapper_args(
                    &db,
                    &root.join("images"),
                    &global_candidate,
                    false,
                    &gpu_index,
                    GlobalMapperOptions {
                        use_gravity_prior,
                        fixed_rotation_ba: enable_fixed_rotation,
                        disable_sensor_refinement: rig_extrinsics_ready,
                        quality_refinement: quality_profile == ColmapQualityProfile::Tuned,
                    },
                );
                let mut quality_gate_audit = None;
                let global_result = run_mapper_with_gpu_fallback(
                    app,
                    id,
                    &colmap,
                    &global_candidate,
                    &gpu_args,
                    &cpu_args,
                    global_gpu,
                    "global_mapper",
                    control,
                    || {},
                    |_| {},
                )
                .and_then(|()| {
                    if !sparse_rig_model_exists(&global_candidate) {
                        return Err("global_mapper 未產生含 rigs/frames 的候選模型".to_owned());
                    }
                    validate_colmap_configured_rig_model(
                        app,
                        id,
                        &colmap,
                        &root,
                        &global_candidate.join("0"),
                        control,
                    )?;
                    let candidate_text = root.join("metadata/global-candidate-text");
                    export_colmap_text_model(
                        app,
                        id,
                        &colmap,
                        &global_candidate.join("0"),
                        &candidate_text,
                        control,
                    )?;
                    let candidate_quality = global_candidate_quality_metrics(&candidate_text)?;
                    let quality_gate = evaluate_global_candidate_quality(
                        seed_quality.clone(),
                        candidate_quality,
                    );
                    let accepted = quality_gate.accepted;
                    let issues = quality_gate.issues.clone();
                    quality_gate_audit = Some(quality_gate.clone());
                    if !accepted {
                        return Err(format!(
                            "global candidate quality gate rejected: {}",
                            issues.join("; ")
                        ));
                    }
                    Ok(quality_gate)
                });
                match global_result {
                    Ok(quality_gate) => {
                        let candidate_complete_rigs =
                            quality_gate.candidate.complete_registered_rig_frames;
                        let incremental_seed = root.join("sparse_incremental_seed");
                        remove_align_artifact(&incremental_seed)?;
                        fs::rename(&sparse, &incremental_seed)
                            .map_err(|error| format!("無法保存 incremental 種子模型：{error}"))?;
                        if let Err(error) = fs::rename(&global_candidate, &sparse) {
                            fs::rename(&incremental_seed, &sparse).map_err(|restore_error| {
                                format!(
                                    "global 模型提交失敗：{error}；incremental 種子還原也失敗：{restore_error}"
                                )
                            })?;
                            return Err(format!("global 模型提交失敗：{error}"));
                        }
                        effective_final_mapper_component = "global_mapper";
                        write_json_atomic(
                            &candidate_audit,
                            &json!({
                                "schemaVersion": 1,
                                "attempted": true,
                                "status": "accepted",
                                "gravityPriorRequested": use_gravity_prior,
                                "seedCompleteRegisteredRigFrames": seed_complete_rigs,
                                "candidateCompleteRegisteredRigFrames": candidate_complete_rigs,
                                "qualityGate": &quality_gate,
                                "effectiveMapper": "global_mapper",
                            }),
                        )?;
                        emit_log(
                            app,
                            id,
                            "info",
                            "global_mapper 候選模型驗證通過，已保留 incremental seed 並提交 global 結果",
                        );
                    }
                    Err(error) if cancelled_error(&error, control) => return Err(error),
                    Err(error) => {
                        remove_align_artifact(&global_candidate)?;
                        write_json_atomic(
                            &candidate_audit,
                            &json!({
                                "schemaVersion": 1,
                                "attempted": true,
                                "status": "rejected",
                                "gravityPriorRequested": use_gravity_prior,
                                "seedCompleteRegisteredRigFrames": seed_complete_rigs,
                                "qualityGate": &quality_gate_audit,
                                "effectiveMapper": "bootstrap_mapper",
                                "error": &error,
                            }),
                        )?;
                        emit_log(
                            app,
                            id,
                            "warning",
                            format!("global_mapper 候選失敗，保留已驗證 incremental 結果：{error}"),
                        );
                    }
                }
            }
            Err(error) if cancelled_error(&error, control) => return Err(error),
            Err(error) => {
                write_json_atomic(
                    &candidate_audit,
                    &json!({
                        "schemaVersion": 1,
                        "attempted": false,
                        "status": "prior_validation_failed",
                        "gravityPriorRequested": use_gravity_prior,
                        "effectiveMapper": "bootstrap_mapper",
                        "error": &error,
                    }),
                )?;
                emit_log(
                    app,
                    id,
                    "warning",
                    format!("IMU/global 前提未通過，保留 incremental 結果：{error}"),
                );
            }
        }
    }
    if let Some(executable) = manifest
        .settings
        .pointer("/align/orientationPriorExecutable")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        emit_log(
            app,
            id,
            "info",
            "正在驗證並執行外部 orientation-aware BA；不會將 quaternion 傳給 stock COLMAP",
        );
        match run_external_orientation_ba_candidate(
            app,
            id,
            &root,
            &PathBuf::from(executable),
            &colmap,
            &db,
            &sparse,
            fixed_rotation_ba,
            control,
        ) {
            Ok(()) => {
                effective_final_mapper_component = "external_orientation_ba";
                emit_log(app, id, "info", "外部 orientation BA 候選已驗證並提交");
            }
            Err(error) if cancelled_error(&error, control) => return Err(error),
            Err(error) => {
                remove_align_artifact(&root.join("sparse_orientation_candidate"))?;
                emit_log(
                    app,
                    id,
                    "warning",
                    format!("外部 orientation BA 未套用，保留 visual 模型：{error}"),
                );
            }
        }
    }
    phase_durations_ms.insert(
        "mapping".to_owned(),
        mapping_started.elapsed().as_secs_f64() * 1000.0,
    );
    phase_durations_ms.insert(
        "alignTotal".to_owned(),
        align_started.elapsed().as_secs_f64() * 1000.0,
    );
    let timing_path = root.join("metadata/align_timings.json");
    write_json_atomic(
        &timing_path,
        &json!({
            "schemaVersion": 1,
            "phaseDurationsMs": phase_durations_ms,
            "effectiveMapper": effective_final_mapper_component,
        }),
    )?;
    let final_fingerprint =
        build_align_fingerprint(&root, &manifest.settings, &colmap_version, use_masks)?;
    write_align_checkpoint(
        &checkpoint_path,
        &final_fingerprint,
        &feature_fingerprint,
        true,
        Some(effective_final_mapper_component),
    )?;
    let final_text_model = root.join("metadata/final-model-text");
    export_colmap_text_model(
        app,
        id,
        &colmap,
        &sparse.join("0"),
        &final_text_model,
        control,
    )?;
    let benchmark_variant = if effective_final_mapper_component == "global_mapper" {
        crate::reconstruction_benchmark::BenchmarkVariant::CGlobal
    } else if setting_bool(&manifest.settings, "/extract/keyframePruning", true) {
        crate::reconstruction_benchmark::BenchmarkVariant::BImuPruning
    } else {
        crate::reconstruction_benchmark::BenchmarkVariant::ACurrent
    };
    let benchmark_request = crate::reconstruction_benchmark::BenchmarkRequest {
        variant: benchmark_variant,
        model_dir: Some(final_text_model),
        timing_path: Some(timing_path),
    };
    let benchmark_path = root
        .join("metadata")
        .join(format!("benchmark_{}.json", benchmark_variant.as_str()));
    let benchmark = crate::reconstruction_benchmark::write_benchmark_report(
        &root,
        &benchmark_request,
        &benchmark_path,
    )?;
    emit_log(
        app,
        id,
        "info",
        format!(
            "已輸出 {} benchmark report（{}）",
            benchmark_variant.as_str(),
            if benchmark.partial {
                "partial"
            } else {
                "complete"
            }
        ),
    );
    remove_align_artifact(&unconfigured_database_backup)?;
    emit_colmap_step_completed(
        app,
        id,
        "final-mapping",
        4,
        "最終模型重建完成",
        effective_final_mapper_component,
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
    let registration = benchmark
        .colmap
        .complete_registered_rig_frame_count
        .filter(|_| rig_frame_count > 0)
        .map(|registered| RegistrationSummary {
            registered: registered.min(rig_frame_count),
            total: rig_frame_count,
        });
    let gravity_prior_applied = effective_final_mapper_component == "global_mapper"
        && use_gravity_prior
        && has_valid_global_mapper_priors(&root, true);
    Ok(StageRunOutput {
        artifacts: vec![
            db.to_string_lossy().into_owned(),
            root.join("rig_config.json").to_string_lossy().into_owned(),
            sparse.to_string_lossy().into_owned(),
            benchmark_path.to_string_lossy().into_owned(),
        ],
        registration,
        capability_updates: BTreeMap::from([(
            "imuApplied".to_owned(),
            gravity_prior_applied
                || effective_final_mapper_component == "external_orientation_ba",
        )]),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        balanced_select_expression, build_align_fingerprint, build_feature_fingerprint,
        can_reuse_align_result, can_reuse_feature_database, candidate_ffmpeg_args,
        candidate_image_names, cleanup_align_artifacts, cleanup_obsolete_candidate_cache,
        cleanup_stale_full_res_dirs, colmap_image_pair_id, colmap_step_progress,
        commit_configured_rig_model, complete_registered_dual_fisheye_frames,
        create_colmap_database_backup,
        cross_source_pair_lines,
        colmap_quality_profile, ColmapQualityProfile,
        dual_fisheye_registration_totals, expected_candidate_frames, extract_frame_settings,
        evaluate_global_candidate_quality, extraction_completed_count, feature_extractor_args,
        global_mapper_args,
        global_mapper_prerequisite_error, has_valid_global_mapper_priors,
        calibrated_pair_refresh_requires_fail_closed, effective_mapper_matches_checkpoint,
        invalidate_calibrated_prior_artifacts, is_mapper_gpu_cpu_fallback_line,
        is_rig_pose_derivation_failure_line, keyframe_pruning_settings,
        load_candidate_selection_checkpoint, map_full_res_candidates, mapper_args,
        mapper_gpu_index, mapper_mode, mapper_still_required_after_rig_setup, mask_classes,
        mask_confidence, mask_enabled,
        matches_importer_args, merge_pair_lists, parse_feature_name, parse_feature_progress,
        parse_gpu_index,
        parse_mapper_registration, parse_matching_progress, parse_showinfo_timestamp_ms,
        probe_duration_seconds, read_raw_frames, registered_rig_image_names,
        reset_capabilities_for_stage_start, restore_colmap_database_backup,
        rollback_calibrated_pair_transaction,
        rig_bootstrap_shared_frame_count, rig_camera_rotations,
        rig_configs_from_camera_extrinsics,
        rig_config_has_complete_sensor_poses, rig_mapping_plan, select_best_bootstrap_candidate,
        selected_ffmpeg_args, setting_bool, source_stage_progress, sparse_model_directories,
        synchronized_candidate_count, validate_rig_bootstrap_registration,
        validate_rigs_text_sensor_poses, verified_bootstrap_initial_pairs,
        view_graph_calibrator_retry_args, with_hwaccel_auto, write_candidate_selection_checkpoint,
        write_rig_and_pairs, AlignCheckpoint, ColmapFraction, GlobalCandidateQualityMetrics,
        ExtractionStage, GlobalMapperOptions, JobControl, JobManager, LogEvent, MapperMode,
        MapperOptions, ProgressEvent, RawFrameMessage, RegistrationSummary, RigBootstrapCamera,
        RigBootstrapConfig, RigBootstrapModelCandidate, RigMappingPlan, StageName,
        StartStageRequest, StreamingCandidateSelector, CANDIDATE_FRAME_BYTES,
        CANDIDATE_IMAGE_FORMAT, CANDIDATE_PROXY_SIZE, CANDIDATE_STREAM_WIDTH,
    };
    use crate::doctor::ColmapCapabilities;
    use crate::masking::CancelToken;
    use serde_json::json;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn starting_align_clears_stale_imu_capability() {
        let mut manifest: crate::project::ProjectManifest =
            serde_json::from_value(json!({})).unwrap();
        manifest.capabilities.insert("imuApplied".to_owned(), true);

        reset_capabilities_for_stage_start(&mut manifest, &StageName::Align);

        assert_eq!(manifest.capabilities.get("imuApplied"), Some(&false));
    }

    #[test]
    fn incremental_checkpoint_accepts_unknown_rig_bootstrap_mapper() {
        assert!(effective_mapper_matches_checkpoint(
            MapperMode::Incremental,
            false,
            Some("bootstrap_mapper")
        ));
        assert!(effective_mapper_matches_checkpoint(
            MapperMode::Incremental,
            false,
            Some("final_mapper")
        ));
        assert!(!effective_mapper_matches_checkpoint(
            MapperMode::Global,
            false,
            Some("bootstrap_mapper")
        ));
    }

    #[test]
    fn pair_rollback_failure_is_a_fail_closed_error() {
        assert!(calibrated_pair_refresh_requires_fail_closed(
            "matching failed; calibrated pair transaction rollback failed: database locked"
        ));
        assert!(!calibrated_pair_refresh_requires_fail_closed(
            "matching failed; original graph restored"
        ));
    }

    #[test]
    fn global_candidate_gate_rejects_fragmentation_and_support_regression() {
        let seed = GlobalCandidateQualityMetrics {
            complete_registered_rig_frames: 100,
            model: crate::reconstruction_benchmark::ColmapModelQualityMetrics {
                registered_image_count: 200,
                points3d_count: 10_000,
                median_track_length: Some(4.0),
                median_reprojection_error_px: Some(0.7),
                connected_component_count: 1,
                largest_connected_component_image_count: 200,
                largest_component_coverage_ratio: 1.0,
            },
        };
        let candidate = GlobalCandidateQualityMetrics {
            complete_registered_rig_frames: 120,
            model: crate::reconstruction_benchmark::ColmapModelQualityMetrics {
                registered_image_count: 240,
                points3d_count: 2_000,
                median_track_length: Some(2.0),
                median_reprojection_error_px: Some(1.2),
                connected_component_count: 5,
                largest_connected_component_image_count: 120,
                largest_component_coverage_ratio: 0.5,
            },
        };

        let gate = evaluate_global_candidate_quality(seed, candidate);

        assert!(!gate.accepted);
        assert!(gate.issues.iter().any(|issue| issue.contains("point support")));
        assert!(gate
            .issues
            .iter()
            .any(|issue| issue.contains("largest-component")));
        assert!(gate
            .issues
            .iter()
            .any(|issue| issue.contains("connected components")));
    }

    #[test]
    fn post_rig_focal_retry_is_bounded_and_deterministic() {
        let args = view_graph_calibrator_retry_args(Path::new("database.db"));
        assert!(args
            .windows(2)
            .any(|values| values == ["--min_calibrated_pair_ratio", "0.25"]));
        assert!(args
            .windows(2)
            .any(|values| values == ["--default_random_seed", "0"]));
    }

    #[test]
    fn calibrated_pair_transaction_rolls_back_database_and_pairs_together() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("database.db");
        let backup = temp.path().join("database.backup");
        let pairs = temp.path().join("pairs.txt");
        fs::write(&database, b"changed database").unwrap();
        fs::write(&pairs, b"changed pairs\n").unwrap();
        fs::create_dir(&backup).unwrap();
        fs::write(backup.join("database.db"), b"original database").unwrap();

        rollback_calibrated_pair_transaction(
            &database,
            &backup,
            &pairs,
            b"original pairs\n",
        )
        .unwrap();

        assert_eq!(fs::read(database).unwrap(), b"original database");
        assert_eq!(fs::read(pairs).unwrap(), b"original pairs\n");
        assert!(!backup.exists());
    }

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
    fn gpu_is_requested_by_default_but_respects_an_explicit_choice() {
        assert!(setting_bool(&json!({}), "/align/useGpu", true));
        assert!(setting_bool(
            &json!({"align": {"useGpu": true}}),
            "/align/useGpu",
            true
        ));
        assert!(!setting_bool(
            &json!({"align": {"useGpu": false}}),
            "/align/useGpu",
            true
        ));
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
        let feature = feature_extractor_args(
            root,
            &db,
            true,
            "0,1",
            true,
            ColmapQualityProfile::Tuned,
        );
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
            .any(|args| { args == ["--SiftExtraction.max_num_features", "10240"] }));
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--SiftExtraction.peak_threshold", "0.006"] }));
        assert!(feature
            .windows(2)
            .any(|args| { args == ["--ImageReader.camera_model", "OPENCV_FISHEYE"] }));
        assert!(feature.contains(&"--ImageReader.mask_path".to_owned()));

        let baseline_feature = feature_extractor_args(
            root,
            &db,
            true,
            "0",
            false,
            ColmapQualityProfile::Baseline,
        );
        assert!(baseline_feature
            .windows(2)
            .any(|args| args == ["--SiftExtraction.max_num_features", "8192"]));
        assert!(!baseline_feature
            .iter()
            .any(|arg| arg == "--SiftExtraction.peak_threshold"));

        let matching = matches_importer_args(
            root,
            &db,
            true,
            "0,1",
            ColmapQualityProfile::Tuned,
        );
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--FeatureMatching.use_gpu", "1"] }));
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--FeatureMatching.gpu_index", "0,1"] }));
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--FeatureMatching.max_num_matches", "10240"] }));
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--TwoViewGeometry.confidence", "0.999"] }));
        assert!(matching
            .windows(2)
            .any(|args| { args == ["--TwoViewGeometry.max_num_trials", "15000"] }));
        let baseline_matching = matches_importer_args(
            root,
            &db,
            true,
            "0",
            ColmapQualityProfile::Baseline,
        );
        assert!(baseline_matching
            .windows(2)
            .any(|args| args == ["--FeatureMatching.max_num_matches", "8192"]));
        assert!(!baseline_matching
            .iter()
            .any(|arg| arg == "--FeatureMatching.guided_matching"));

        let mapper_index = mapper_gpu_index("0,1");
        let mapper = mapper_args(
            &db,
            &images,
            &sparse,
            true,
            mapper_index,
            MapperOptions {
                multiple_models: false,
                initial_image_pair: None,
                disable_sensor_refinement: true,
            },
        );
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
            .any(|args| { args == ["--Mapper.ba_refine_sensor_from_rig", "0"] }));
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
            MapperOptions {
                multiple_models: true,
                initial_image_pair: Some((7, 9)),
                disable_sensor_refinement: false,
            },
        );
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.multiple_models", "1"]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.min_model_size", "2"]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.init_image_id1", "7"]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.init_image_id2", "9"]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.init_num_trials", "1"]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.ba_local_backend", "CERES"]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--Mapper.ba_global_backend", "CERES"]));
        assert!(!args
            .iter()
            .any(|arg| arg == "--Mapper.ba_local_num_images"));
        assert!(!args
            .iter()
            .any(|arg| arg == "--Mapper.filter_max_reproj_error"));
        assert!(!args
            .iter()
            .any(|arg| arg == "--Mapper.ba_refine_sensor_from_rig"));
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
            RigMappingPlan::BootstrapThenConfigure,
            "unknown sensor extrinsics require one mapper followed by rig conversion"
        );
        assert!(!mapper_still_required_after_rig_setup(
            RigMappingPlan::BootstrapThenConfigure
        ));
        assert!(mapper_still_required_after_rig_setup(
            RigMappingPlan::PreconfiguredSinglePass
        ));
        let registered = registered_rig_image_names(
            "# images\n1 1 0 0 0 0 0 0 1 lens0/frame one.png\n0 0 -1\n2 1 0 0 0 0 0 0 2 lens1/frame one.png\n0 0 -1\n",
            &prefixes,
        );
        let summary = validate_rig_bootstrap_registration(&[config], &registered).unwrap();
        assert!(summary.contains("1 組共同影格"));
    }

    #[test]
    fn configured_bootstrap_model_is_committed_without_a_second_mapper() {
        let temp = TempDir::new().unwrap();
        let configured = temp.path().join("configured");
        let sparse = temp.path().join("sparse");
        fs::create_dir_all(&configured).unwrap();
        fs::create_dir_all(sparse.join("old")).unwrap();
        fs::write(sparse.join("old/stale.bin"), b"stale").unwrap();
        for name in ["rigs", "cameras", "frames", "images", "points3D"] {
            fs::write(configured.join(format!("{name}.bin")), name.as_bytes()).unwrap();
        }

        commit_configured_rig_model(&configured, &sparse).unwrap();

        assert!(!configured.exists());
        assert!(!sparse.join("old").exists());
        for name in ["rigs", "cameras", "frames", "images", "points3D"] {
            assert_eq!(
                fs::read(sparse.join("0").join(format!("{name}.bin"))).unwrap(),
                name.as_bytes()
            );
        }
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
    fn bootstrap_candidate_counts_shared_frames_across_lenses() {
        let configs = serde_json::from_value::<Vec<RigBootstrapConfig>>(json!([{
            "cameras": [
                {"image_prefix": "lens0/", "ref_sensor": true},
                {"image_prefix": "lens1/"}
            ]
        }]))
        .unwrap();
        let registered = BTreeSet::from([
            "lens0/frame0001.png".to_owned(),
            "lens0/frame0002.png".to_owned(),
            "lens0/frame0003.png".to_owned(),
            "lens1/frame0002.png".to_owned(),
            "lens1/frame0003.png".to_owned(),
            "lens1/frame0004.png".to_owned(),
        ]);
        assert_eq!(rig_bootstrap_shared_frame_count(&configs, &registered), 2);
    }

    #[test]
    fn bootstrap_models_are_discovered_in_numeric_order() {
        let temp = tempfile::tempdir().unwrap();
        for name in ["10", "2", "0", "not-a-model"] {
            fs::create_dir(temp.path().join(name)).unwrap();
        }
        fs::write(temp.path().join("1"), b"not a directory").unwrap();
        let names = sparse_model_directories(temp.path())
            .into_iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(names, ["0", "2", "10"]);
    }

    #[test]
    fn bootstrap_candidate_balances_model_size_and_shared_calibration_coverage() {
        let candidate = |shared_frame_count, registered_image_count| RigBootstrapModelCandidate {
            path: PathBuf::new(),
            shared_frame_count,
            registered_image_count,
            summary: String::new(),
        };
        let selected =
            select_best_bootstrap_candidate(vec![candidate(2, 100), candidate(3, 10)]).unwrap();
        assert_eq!(
            (selected.shared_frame_count, selected.registered_image_count),
            (3, 10)
        );

        let selected =
            select_best_bootstrap_candidate(vec![candidate(35, 89), candidate(40, 80)]).unwrap();
        assert_eq!(
            (selected.shared_frame_count, selected.registered_image_count),
            (40, 80)
        );

        let selected =
            select_best_bootstrap_candidate(vec![candidate(3, 100), candidate(100, 99)]).unwrap();
        assert_eq!(
            (selected.shared_frame_count, selected.registered_image_count),
            (100, 99)
        );
    }

    #[test]
    fn bootstrap_retry_uses_only_strong_verified_cross_lens_geometry() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("database.db");
        let connection = rusqlite::Connection::open(&database).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
                 CREATE TABLE two_view_geometries (
                    pair_id INTEGER PRIMARY KEY,
                    rows INTEGER NOT NULL,
                    config INTEGER NOT NULL
                 );",
            )
            .unwrap();
        for (image_id, name) in [
            (1_i64, "lens0/frame0001.png"),
            (2, "lens1/frame0001.png"),
            (3, "lens0/frame0002.png"),
            (4, "lens1/frame0003.png"),
            (5, "lens0/degenerate.png"),
            (6, "lens1/degenerate.png"),
            (2_147_483_647, "lens0/malformed-id.png"),
        ] {
            connection
                .execute(
                    "INSERT INTO images(image_id, name) VALUES (?1, ?2)",
                    rusqlite::params![image_id, name],
                )
                .unwrap();
        }
        connection
            .execute(
                "INSERT INTO two_view_geometries(pair_id, rows, config) VALUES (?1, 120, 3)",
                [colmap_image_pair_id(1, 2)],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO two_view_geometries(pair_id, rows, config) VALUES (?1, 130, 4)",
                [colmap_image_pair_id(3, 4)],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO two_view_geometries(pair_id, rows, config) VALUES (?1, 99, 3)",
                [colmap_image_pair_id(1, 4)],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO two_view_geometries(pair_id, rows, config) VALUES (?1, 200, 1)",
                [colmap_image_pair_id(5, 6)],
            )
            .unwrap();
        drop(connection);

        let configs = serde_json::from_value::<Vec<RigBootstrapConfig>>(json!([{
            "cameras": [
                {"image_prefix": "lens0/", "ref_sensor": true},
                {"image_prefix": "lens1/"}
            ]
        }]))
        .unwrap();
        let pairs = verified_bootstrap_initial_pairs(&database, &configs).unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs[0].same_frame);
        assert_eq!((pairs[0].image_id1, pairs[0].image_id2), (1, 2));
        assert_eq!(pairs[1].inlier_count, 130);
        assert!(!pairs[1].same_frame);
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
            rig_mapping_plan(&configs),
            RigMappingPlan::PreconfiguredSinglePass
        );
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
        let baseline = build_feature_fingerprint(
            temp.path(),
            "COLMAP 4.1.1",
            false,
            ColmapQualityProfile::Baseline,
        )
        .unwrap();
        fs::create_dir_all(temp.path().join("metadata")).unwrap();
        fs::write(temp.path().join("metadata/pairs.txt"), b"changed pairs").unwrap();
        fs::write(temp.path().join("rig_config.json"), b"changed rig").unwrap();
        assert_eq!(
            baseline,
            build_feature_fingerprint(
                temp.path(),
                "COLMAP 4.1.1",
                false,
                ColmapQualityProfile::Baseline,
            )
            .unwrap()
        );
        assert_ne!(
            baseline,
            build_feature_fingerprint(
                temp.path(),
                "COLMAP 4.2.0",
                false,
                ColmapQualityProfile::Baseline,
            )
            .unwrap()
        );
        fs::create_dir_all(temp.path().join("masks_colmap/lens0")).unwrap();
        fs::write(
            temp.path().join("masks_colmap/lens0/frame0001.png"),
            b"mask",
        )
        .unwrap();
        assert_ne!(
            baseline,
            build_feature_fingerprint(
                temp.path(),
                "COLMAP 4.1.1",
                true,
                ColmapQualityProfile::Baseline,
            )
            .unwrap()
        );
        assert_ne!(
            baseline,
            build_feature_fingerprint(
                temp.path(),
                "COLMAP 4.1.1",
                false,
                ColmapQualityProfile::Tuned,
            )
            .unwrap()
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
        assert_eq!(checkpoint.effective_mapper, None);
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
    fn completed_alignment_reports_complete_rig_frames_and_percentage() {
        let images = concat!(
            "# Image list\n",
            "1 1 0 0 0 0 0 0 1 lens0/frame-a.jpg\n\n",
            "2 1 0 0 0 0 0 0 2 lens1/frame-a.jpg\n\n",
            "3 1 0 0 0 0 0 0 1 lens0/frame-b.jpg\n\n",
        );
        assert_eq!(complete_registered_dual_fisheye_frames(images), 1);
        assert_eq!(
            RegistrationSummary {
                registered: 149,
                total: 257,
            }
            .completion_message(),
            "對齊處理完成：已註冊 149 / 257 組相機組影格（58.0%）"
        );
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
    fn temporal_rescue_links_skip_beyond_the_local_matching_window() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for sequence in 1..=18 {
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
        let frames = (1..=18)
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
        let pairs_path = temp.path().join("metadata/pairs.txt");
        let pairs = fs::read_to_string(&pairs_path).unwrap();
        assert!(!pairs.contains("lens0/source000_00000001.png lens0/source000_00000009.png"));

        fs::write(
            temp.path().join("rig_config.json"),
            serde_json::to_vec(&json!([{"cameras": [
                {"image_prefix": "lens0/", "ref_sensor": true},
                {
                    "image_prefix": "lens1/",
                    "cam_from_rig_rotation": [1.0, 0.0, 0.0, 0.0],
                    "cam_from_rig_translation": [0.0, 0.0, 0.0]
                }
            ]}]))
            .unwrap(),
        )
        .unwrap();
        write_rig_and_pairs(temp.path()).unwrap();
        let pairs = fs::read_to_string(&pairs_path).unwrap();
        for rescue in [9, 13, 17] {
            assert!(pairs.contains(&format!(
                "lens0/source000_00000001.png lens0/source000_{rescue:08}.png"
            )));
            assert!(pairs.contains(&format!(
                "lens1/source000_00000001.png lens1/source000_{rescue:08}.png"
            )));
        }
        // Without calibrated per-frame FOV overlap, long rescue edges stay on
        // the same physical sensor to avoid speculative cross-lens matches.
        assert!(!pairs.contains("lens0/source000_00000001.png lens1/source000_00000009.png"));
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
    fn colmap_quality_profile_defaults_to_baseline_and_supports_tuned() {
        assert_eq!(
            colmap_quality_profile(&json!({})).unwrap(),
            ColmapQualityProfile::Baseline
        );
        assert_eq!(
            colmap_quality_profile(&json!({"align": {"colmapQualityProfile": "tuned"}}))
                .unwrap(),
            ColmapQualityProfile::Tuned
        );
        assert!(colmap_quality_profile(
            &json!({"align": {"colmapQualityProfile": "aggressive"}})
        )
        .is_err());
    }

    #[test]
    fn refreshed_pair_graph_preserves_only_old_cross_source_retrieval_pairs() {
        let original = b"lens0/source000_a.png lens1/source000_b.png\nlens0/source000_c.png lens1/source001_c.png\n";
        let preserved = cross_source_pair_lines(original).unwrap();
        let merged = merge_pair_lists(
            &preserved,
            b"lens0/source000_a.png lens0/source000_d.png\n",
        )
        .unwrap();
        assert_eq!(
            std::str::from_utf8(&merged).unwrap(),
            "lens0/source000_a.png lens0/source000_d.png\nlens0/source000_c.png lens1/source001_c.png\n"
        );
    }

    #[test]
    fn database_rig_groups_remain_separate_in_json_config() {
        let camera = |rig_id, prefix: &str, reference| {
            crate::colmap_priors::RigCameraExtrinsic {
                rig_id,
                image_prefix: prefix.to_owned(),
                ref_sensor: reference,
                cam_from_rig_rotation: [1.0, 0.0, 0.0, 0.0],
                cam_from_rig_translation: [0.0, 0.0, 0.0],
            }
        };
        let configs = rig_configs_from_camera_extrinsics(vec![
            camera(2, "rig2/lens1/", false),
            camera(1, "rig1/lens0/", true),
            camera(2, "rig2/lens0/", true),
            camera(1, "rig1/lens1/", false),
        ]);
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].cameras.len(), 2);
        assert!(configs[0]
            .cameras
            .iter()
            .all(|camera| camera.image_prefix.starts_with("rig1/")));
        assert!(configs[1]
            .cameras
            .iter()
            .all(|camera| camera.image_prefix.starts_with("rig2/")));
        assert!(rig_camera_rotations(&configs).is_err());
    }

    #[test]
    fn global_mapper_args_use_colmap_4_1_1_option_names() {
        let args = global_mapper_args(
            Path::new("database.db"),
            Path::new("images"),
            Path::new("sparse"),
            true,
            "2,3",
            GlobalMapperOptions {
                use_gravity_prior: true,
                fixed_rotation_ba: false,
                disable_sensor_refinement: true,
                quality_refinement: true,
            },
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
        assert!(pairs.contains(&("--GlobalMapper.gp_max_num_iterations", "120")));
        assert!(pairs.contains(&("--GlobalMapper.tri_complete_max_reproj_error", "8")));
        assert!(pairs.contains(&("--GlobalMapper.tri_merge_max_reproj_error", "8")));
        assert!(pairs.contains(&("--GlobalMapper.max_normalized_reproj_error", "0.008")));

        let without_gravity = global_mapper_args(
            Path::new("database.db"),
            Path::new("images"),
            Path::new("sparse"),
            false,
            "-1",
            GlobalMapperOptions {
                use_gravity_prior: false,
                fixed_rotation_ba: false,
                disable_sensor_refinement: false,
                quality_refinement: false,
            },
        );
        let pairs = without_gravity
            .windows(2)
            .map(|pair| (pair[0].as_str(), pair[1].as_str()))
            .collect::<BTreeSet<_>>();
        assert!(pairs.contains(&("--GlobalMapper.ra_use_gravity", "0")));
        assert!(pairs.contains(&("--GlobalMapper.ra_use_stratified", "0")));
        assert!(pairs.contains(&("--GlobalMapper.refine_sensor_from_rig", "1")));
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
        let error = global_mapper_prerequisite_error(temp.path(), true, &capabilities, true, false)
            .expect("missing global mapper priors must block explicit mode");
        assert!(error.contains("focal prior"));
        assert!(error.contains("global_mapper_priors.json"));
    }

    #[test]
    fn invalidating_priors_removes_only_calibration_derived_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let metadata = temp.path().join("metadata");
        fs::create_dir_all(&metadata).unwrap();
        for name in [
            "global_mapper_priors.json",
            "imu_calibration.json",
            "orientation_priors.json",
            "orientation_priors_source000.json",
            "rolling_shutter_source000.json",
        ] {
            fs::write(metadata.join(name), b"derived").unwrap();
        }
        fs::write(metadata.join("source000_telemetry.json"), b"source").unwrap();
        fs::write(metadata.join("pairs.txt"), b"pairs").unwrap();

        invalidate_calibrated_prior_artifacts(temp.path()).unwrap();

        assert!(metadata.join("source000_telemetry.json").is_file());
        assert!(metadata.join("pairs.txt").is_file());
        assert!(!metadata.join("global_mapper_priors.json").exists());
        assert!(!metadata.join("imu_calibration.json").exists());
        assert!(!metadata.join("orientation_priors.json").exists());
        assert!(!metadata.join("orientation_priors_source000.json").exists());
        assert!(!metadata.join("rolling_shutter_source000.json").exists());
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
    fn default_rig_has_unknown_extrinsics_and_migrates_colocated_default() {
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
            br#"[{"cameras":[{"image_prefix":"lens0/","ref_sensor":true},{"image_prefix":"lens1/","cam_from_rig_rotation":[0.0,0.0,1.0,0.0],"cam_from_rig_translation":[0.0,0.0,0.0]}]}]"#,
        )
        .unwrap();

        write_rig_and_pairs(temp.path()).unwrap();

        let configs =
            serde_json::from_slice::<Vec<RigBootstrapConfig>>(&fs::read(&rig_path).unwrap())
                .unwrap();
        assert!(!rig_config_has_complete_sensor_poses(&configs));
        assert_eq!(
            rig_mapping_plan(&configs),
            RigMappingPlan::BootstrapThenConfigure,
            "physical lenses without measured extrinsics require calibration"
        );
        assert_eq!(configs[0].cameras[1].cam_from_rig_rotation, None);
        assert_eq!(configs[0].cameras[1].cam_from_rig_translation, None);
    }

    #[test]
    fn missing_rig_config_creates_unknown_extrinsics_default() {
        let temp = tempfile::tempdir().unwrap();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(temp.path().join("images").join(lens)).unwrap();
            for name in ["frame0001.png", "frame0002.png"] {
                fs::write(temp.path().join("images").join(lens).join(name), b"frame").unwrap();
            }
        }

        write_rig_and_pairs(temp.path()).unwrap();

        let configs = serde_json::from_slice::<Vec<RigBootstrapConfig>>(
            &fs::read(temp.path().join("rig_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            rig_mapping_plan(&configs),
            RigMappingPlan::BootstrapThenConfigure
        );
        assert!(!configs[0].cameras[1].has_explicit_pose());
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
