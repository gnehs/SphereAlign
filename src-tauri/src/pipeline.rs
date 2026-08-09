//! Resumable stage orchestration around system FFmpeg, the native mask engine,
//! and COLMAP. External commands are always passed as argument arrays (never a
//! shell string), so paths containing spaces cannot inject commands.

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
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError};
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
const CANDIDATE_SELECTION_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
const CANDIDATE_PROGRESS_INTERVAL: Duration = Duration::from_millis(500);
const CANDIDATE_SELECTION_PROGRESS_SHARE: f32 = 0.7;
const FULL_RESOLUTION_PROGRESS_SHARE: f32 = 0.2;

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
    image_format: String,
    selection: SelectionMetadata,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartStageRequest {
    pub project_path: String,
    pub stage: StageName,
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
    let colmap_path = request.colmap_path.clone();
    let response = StartStageResponse { job_id: id.clone() };
    thread::spawn(move || {
        let stage_started_at = Instant::now();
        let _ = project::update_stage_timed(
            &mut manifest,
            &stage,
            StageStatus::Running,
            0.0,
            "處理階段已開始",
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
            "正在準備工作",
            "running",
            false,
        );
        let result = match stage {
            StageName::Extract => run_extract(&app, &id, &manifest, &control),
            StageName::Mask => run_mask(&app, &id, &manifest, &control),
            StageName::Align => run_align(&app, &id, &manifest, colmap_path.as_deref(), &control),
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
                    let _ = project::update_stage_timed(
                        &mut manifest,
                        &stage,
                        StageStatus::Completed,
                        1.0,
                        "處理階段已完成",
                        artifacts,
                        Vec::new(),
                        Some(stage_started_at),
                    );
                    emit_progress_detailed(
                        &app,
                        &id,
                        &stage,
                        "completed",
                        1.0,
                        "處理完成",
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

/// Build the software-decoder FFmpeg command used to stream both candidate
/// fisheye lenses as one fixed-size grayscale frame. No candidate image is
/// encoded or written to disk: stdout contains consecutive 1024x512 gray8
/// frames, with lens0 on the left and lens1 on the right.
fn candidate_ffmpeg_args(
    input: &Path,
    stream0: usize,
    stream1: usize,
    candidate_fps: f64,
) -> Vec<String> {
    let lens_filter = |stream: usize, label: &str| {
        format!(
            "[0:{stream}]fps={candidate_fps},setpts=N/({candidate_fps}*TB),scale=w='min(512,iw)':h='min(512,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,pad=512:512:(ow-iw)/2:(oh-ih)/2,format=gray[{label}]"
        )
    };
    let filter = format!(
        "{};{};[lens0][lens1]hstack=inputs=2:shortest=1[out]",
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
}

impl StreamingCandidateSelector {
    fn new(base_fps: f64, candidate_fps: f64, score_candidates: bool) -> Self {
        Self {
            base_fps,
            candidate_fps,
            score_candidates,
            records: Vec::new(),
            best_by_interval: BTreeMap::new(),
        }
    }

    fn push(&mut self, frame: &[u8]) -> Result<(), String> {
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
            output_lens0: None,
            output_lens1: None,
        });
        Ok(())
    }

    fn finish(self) -> Result<Vec<SelectionRecord>, String> {
        if self.records.is_empty() {
            return Err("FFmpeg 未產生任何記憶體候選影格".to_owned());
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

fn stderr_detail(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .lines()
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
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
    let Some(mut stderr) = child.stderr.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("無法讀取 FFmpeg 錯誤輸出".to_owned());
    };
    let (sender, receiver) = sync_channel(2);
    let stdout_reader = thread::spawn(move || read_raw_frames(stdout, sender));
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.read_to_end(&mut bytes);
        bytes
    });
    let mut selector = StreamingCandidateSelector::new(base_fps, candidate_fps, score_candidates);
    let mut stream_error = None;
    loop {
        if control.cancelled.load(Ordering::Acquire) {
            stream_error = Some("cancelled".to_owned());
            break;
        }
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(RawFrameMessage::Frame(frame)) => {
                if let Err(error) = selector.push(&frame) {
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
        let detail = stderr_detail(&stderr);
        return Err(if detail.trim().is_empty() {
            format!("FFmpeg 候選串流結束，狀態碼 {:?}", status.code())
        } else {
            detail
        });
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

fn load_candidate_selection_checkpoint(
    path: &Path,
    input: &Path,
    base_fps: f64,
    candidate_fps: f64,
    dense_fps: f64,
    skip_blurry: bool,
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
        && checkpoint.image_format == CANDIDATE_IMAGE_FORMAT
        && checkpoint.selection.schema_version == 4
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
    let base_fps = setting_f64(&manifest.settings, "/extract/baseFps", 2.0).clamp(0.1, 30.0);
    let dense_fps = setting_f64(&manifest.settings, "/extract/denseFps", 8.0).clamp(base_fps, 60.0);
    let skip_blurry = setting_bool(&manifest.settings, "/extract/skipBlurry", true);
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
        let selection_checkpoint = candidate_root.join("selection.checkpoint.json");
        let selection_metadata = if let Some(selection) = load_candidate_selection_checkpoint(
            &selection_checkpoint,
            &input,
            base_fps,
            candidate_fps,
            dense_fps,
            skip_blurry,
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
                control,
                &mut candidate_progress,
            )?;
            let selected_intervals = selection_records
                .iter()
                .filter(|record| record.selected)
                .count();
            let selection = SelectionMetadata {
                schema_version: 4,
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
        // Release the native-resolution intermediates before telemetry work so
        // long captures do not retain avoidable disk usage for the rest of the
        // source iteration. Errors above still clean through the same guard.
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
        let normalized_path = metadata.join(format!("source{source_index:03}_telemetry.json"));
        match telemetry::parse_and_write(&input, &normalized_path, control.cancelled.clone()) {
            Ok(export) => normalized_telemetry.push(json!({
                "sourceIndex": source_index,
                "path": export.path.to_string_lossy(),
                "cameraModel": export.camera_model,
                "normalizedImuSampleCount": export.normalized_imu_sample_count,
                "fusedAttitudeSampleCount": export.fused_attitude_sample_count,
                "appliedToColmap": false
            })),
            Err(error) if !control.cancelled.load(Ordering::Acquire) => emit_log(
                app,
                id,
                "warning",
                format!(
                    "無法解析來源 {} 的標準化 telemetry：{error}",
                    source_index + 1
                ),
            ),
            Err(error) => return Err(error),
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
        "schemaVersion": 4, "canonicalProjection": "native_fisheye", "sources": manifest.input_paths,
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
        "motionAdaptiveCadence": false,
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
    let root = PathBuf::from(&manifest.output_path);
    let classes = manifest
        .settings
        .pointer("/mask/classes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_else(Vec::new);
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

fn write_rig_and_pairs(root: &Path) -> Result<(), String> {
    fs::write(
        root.join("rig_config.json"),
        serde_json::to_vec_pretty(&json!([{"cameras":[
        {"image_prefix":"lens0/","ref_sensor":true},{"image_prefix":"lens1/"}]}]))
        .unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let lens1 = root.join("images/lens1");
    let mut names: Vec<String> = fs::read_dir(root.join("images/lens0"))
        .map_err(|e| e.to_string())?
        .flatten()
        .filter(|entry| entry.path().is_file())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .filter(|name| lens1.join(name).is_file())
        .collect();
    names.sort();
    if names.len() < 2 {
        return Err("至少需要兩組同名的 lens0/lens1 影格才能對齊".to_owned());
    }
    let mut pairs = BTreeSet::new();
    for (index, name) in names.iter().enumerate() {
        pairs.insert(format!("lens0/{name} lens1/{name}"));
        for neighbor in names.iter().skip(index + 1).take(5) {
            pairs.insert(format!("lens0/{name} lens0/{neighbor}"));
            pairs.insert(format!("lens1/{name} lens1/{neighbor}"));
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
    .map_err(|e| e.to_string())
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

fn run_align(
    app: &AppHandle,
    id: &str,
    manifest: &ProjectManifest,
    custom_colmap_path: Option<&str>,
    control: &JobControl,
) -> Result<Vec<String>, String> {
    let colmap = crate::doctor::resolve_colmap(custom_colmap_path)?;
    let root = PathBuf::from(&manifest.output_path);
    write_rig_and_pairs(&root)?;
    let db = root.join("database.db");
    let sparse = root.join("sparse");
    if db.is_file() && sparse_model_exists(&sparse) {
        emit_log(app, id, "info", "已驗證現有 COLMAP 重建結果，略過重算");
        emit_progress_detailed(
            app,
            id,
            &StageName::Align,
            "completed",
            1.0,
            "已沿用現有 COLMAP 重建結果",
            "running",
            false,
            Some(5),
            Some(5),
            Some("existing sparse model".to_owned()),
            None,
        );
        return Ok(vec![
            db.to_string_lossy().into_owned(),
            root.join("rig_config.json").to_string_lossy().into_owned(),
            sparse.to_string_lossy().into_owned(),
        ]);
    }
    let cuda_available = crate::doctor::report(custom_colmap_path)
        .accelerators
        .into_iter()
        .any(|accelerator| accelerator.kind == "cuda" && accelerator.available);
    let gpu = setting_bool(
        &manifest.settings,
        "/align/useGpu",
        cfg!(target_os = "windows"),
    ) && cuda_available;
    let mut args = vec![
        "feature_extractor".into(),
        "--database_path".into(),
        db.to_string_lossy().into_owned(),
        "--image_path".into(),
        root.join("images").to_string_lossy().into_owned(),
        "--ImageReader.single_camera_per_folder".into(),
        "1".into(),
        "--ImageReader.camera_model".into(),
        "OPENCV_FISHEYE".into(),
        "--FeatureExtraction.use_gpu".into(),
        if gpu { "1".into() } else { "0".into() },
    ];
    if matches!(
        manifest.stage(&StageName::Mask).status,
        StageStatus::Completed
    ) {
        args.push("--ImageReader.mask_path".into());
        args.push(root.join("masks_colmap").to_string_lossy().into_owned());
    }
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "feature-extraction",
        0.0,
        "正在擷取影像特徵",
        "running",
        false,
        Some(0),
        Some(5),
        Some("feature_extractor".to_owned()),
        None,
    );
    run_child(app, id, &colmap, &args, control)?;
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "matching",
        1.0 / 5.0,
        "正在匯入受限影像配對",
        "running",
        false,
        Some(1),
        Some(5),
        Some("matches_importer".to_owned()),
        None,
    );
    run_child(
        app,
        id,
        &colmap,
        &[
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
            if gpu { "1".into() } else { "0".into() },
        ],
        control,
    )?;
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "bootstrap",
        2.0 / 5.0,
        "正在建立初始模型",
        "running",
        false,
        Some(2),
        Some(5),
        Some("bootstrap_mapper".to_owned()),
        None,
    );
    let bootstrap = root.join("sparse_bootstrap");
    if !sparse_model_exists(&bootstrap) {
        if bootstrap.exists() {
            fs::remove_dir_all(&bootstrap).map_err(|e| e.to_string())?;
        }
        fs::create_dir_all(&bootstrap).map_err(|e| e.to_string())?;
        run_child(
            app,
            id,
            &colmap,
            &[
                "mapper".into(),
                "--database_path".into(),
                db.to_string_lossy().into_owned(),
                "--image_path".into(),
                root.join("images").to_string_lossy().into_owned(),
                "--output_path".into(),
                bootstrap.to_string_lossy().into_owned(),
            ],
            control,
        )?;
    }
    let boot0 = bootstrap.join("0");
    if !boot0.is_dir() {
        return Err("COLMAP 初始建模未產生 sparse/0".into());
    }
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "rig",
        3.0 / 5.0,
        "正在估計雙鏡頭相機組",
        "running",
        false,
        Some(3),
        Some(5),
        Some("rig_configurator".to_owned()),
        None,
    );
    run_child(
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
        ],
        control,
    )?;
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "final-mapping",
        4.0 / 5.0,
        "正在重建最終模型",
        "running",
        false,
        Some(4),
        Some(5),
        Some("final_mapper".to_owned()),
        None,
    );
    if sparse.exists() {
        fs::remove_dir_all(&sparse).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(&sparse).map_err(|e| e.to_string())?;
    run_child(
        app,
        id,
        &colmap,
        &[
            "mapper".into(),
            "--database_path".into(),
            db.to_string_lossy().into_owned(),
            "--image_path".into(),
            root.join("images").to_string_lossy().into_owned(),
            "--output_path".into(),
            sparse.to_string_lossy().into_owned(),
            "--Mapper.ba_refine_sensor_from_rig".into(),
            "0".into(),
        ],
        control,
    )?;
    emit_progress_detailed(
        app,
        id,
        &StageName::Align,
        "completed",
        1.0,
        "對齊處理完成",
        "running",
        false,
        Some(5),
        Some(5),
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
        balanced_select_expression, candidate_ffmpeg_args, candidate_image_names,
        cleanup_obsolete_candidate_cache, cleanup_stale_full_res_dirs, expected_candidate_frames,
        extraction_completed_count, load_candidate_selection_checkpoint, map_full_res_candidates,
        mask_confidence, probe_duration_seconds, read_raw_frames, selected_ffmpeg_args,
        source_stage_progress, synchronized_candidate_count, with_hwaccel_auto,
        write_candidate_selection_checkpoint, write_rig_and_pairs, ExtractionStage, JobControl,
        JobManager, LogEvent, ProgressEvent, RawFrameMessage, StageName, StartStageRequest,
        StreamingCandidateSelector, CANDIDATE_FRAME_BYTES, CANDIDATE_IMAGE_FORMAT,
        CANDIDATE_PROXY_SIZE, CANDIDATE_STREAM_WIDTH,
    };
    use crate::masking::CancelToken;
    use serde_json::json;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::Arc;
    use tempfile::TempDir;

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
            "colmapPath": "C:\\COLMAP portable\\COLMAP.bat"
        }))
        .unwrap();

        assert_eq!(
            request.colmap_path.as_deref(),
            Some("C:\\COLMAP portable\\COLMAP.bat")
        );
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
        }

        write_rig_and_pairs(temp.path()).unwrap();

        let pairs = fs::read_to_string(temp.path().join("metadata/pairs.txt")).unwrap();
        assert!(pairs.contains("lens0/source000_00000001.png lens0/source001_00000001.png"));
        assert!(pairs.contains("lens1/source000_00000001.png lens1/source001_00000001.png"));
        assert!(pairs.contains("lens0/source000_00000001.png lens1/source000_00000001.png"));
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
        assert_eq!(filter.matches("setpts=N/(8*TB)").count(), 2);
        assert_eq!(filter.matches("scale=").count(), 2);
        assert_eq!(filter.matches("pad=512:512").count(), 2);
        assert!(filter.contains("hstack=inputs=2:shortest=1[out]"));
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
            schema_version: 4,
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
        write_candidate_selection_checkpoint(&checkpoint, &input, 2.0, 8.0, 8.0, false, &selection)
            .unwrap();

        assert!(
            load_candidate_selection_checkpoint(&checkpoint, &input, 2.0, 8.0, 8.0, false)
                .is_some()
        );
        assert!(
            load_candidate_selection_checkpoint(&checkpoint, &input, 1.0, 8.0, 8.0, false)
                .is_none()
        );
        fs::write(&input, b"video-v2-longer").unwrap();
        assert!(
            load_candidate_selection_checkpoint(&checkpoint, &input, 2.0, 8.0, 8.0, false)
                .is_none()
        );
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
