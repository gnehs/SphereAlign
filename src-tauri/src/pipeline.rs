//! Resumable stage orchestration around system FFmpeg, the native mask engine,
//! and COLMAP. External commands are always passed as argument arrays (never a
//! shell string), so paths containing spaces cannot inject commands.

use crate::doctor::find_executable;
use crate::extraction::{self, ExtractionRequest};
use crate::masking::{self, CancelToken, MaskRequest};
use crate::project::{self, ProjectManifest, StageName, StageStatus};
use crate::telemetry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

static JOB_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartStageRequest {
    pub project_path: String,
    pub stage: StageName,
    #[serde(default)]
    pub settings: Option<Value>,
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
    done: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LogEvent {
    job_id: String,
    level: String,
    message: String,
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
        if jobs
            .values()
            .any(|job| !job.cancelled.load(Ordering::Acquire))
        {
            return Err("Another pipeline stage is already running".to_string());
        }
        jobs.retain(|_, job| !job.cancelled.load(Ordering::Acquire));
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
    let _ = app.emit(
        "pipeline-progress",
        ProgressEvent {
            job_id: id.to_string(),
            stage: stage.clone(),
            phase: phase.to_string(),
            progress: progress.clamp(0.0, 1.0),
            message: message.into(),
            completed: None,
            total: None,
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
        },
    );
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
    let response = StartStageResponse { job_id: id.clone() };
    thread::spawn(move || {
        let _ = project::update_stage(
            &mut manifest,
            &stage,
            StageStatus::Running,
            0.0,
            "Stage started",
            Vec::new(),
            Vec::new(),
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
            StageName::Align => run_align(&app, &id, &manifest, &control),
        };
        let cancelled = control.cancelled.load(Ordering::Acquire);
        if cancelled {
            let _ = project::update_stage(
                &mut manifest,
                &stage,
                StageStatus::Cancelled,
                0.0,
                "Stage cancelled; committed artifacts are resumable",
                Vec::new(),
                Vec::new(),
            );
            emit_progress(
                &app,
                &id,
                &stage,
                "cancelled",
                0.0,
                "工作已取消，可稍後續作",
                "cancelled",
                true,
            );
        } else {
            match result {
                Ok(artifacts) => {
                    let _ = project::update_stage(
                        &mut manifest,
                        &stage,
                        StageStatus::Completed,
                        1.0,
                        "Stage completed",
                        artifacts,
                        Vec::new(),
                    );
                    emit_progress(
                        &app,
                        &id,
                        &stage,
                        "completed",
                        1.0,
                        "處理完成",
                        "completed",
                        true,
                    );
                }
                Err(error) => {
                    emit_log(&app, &id, "error", &error);
                    let _ = project::update_stage(
                        &mut manifest,
                        &stage,
                        StageStatus::Failed,
                        0.0,
                        error.clone(),
                        Vec::new(),
                        vec![error.clone()],
                    );
                    emit_progress(&app, &id, &stage, "failed", 0.0, error, "failed", true);
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

fn run_child(
    app: &AppHandle,
    id: &str,
    program: &Path,
    args: &[String],
    control: &JobControl,
) -> Result<(), String> {
    emit_log(
        app,
        id,
        "info",
        format!(
            "執行 {}",
            program.file_name().unwrap_or_default().to_string_lossy()
        ),
    );
    let mut child = Command::new(program)
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

fn probe_streams(ffprobe: &Path, input: &Path) -> Result<Value, String> {
    let output = Command::new(ffprobe).args(["-v", "error",
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

fn has_candidate_image(path: &Path) -> bool {
    fs::read_dir(path)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry.path().is_file()
                && matches!(
                    entry
                        .path()
                        .extension()
                        .and_then(|value| value.to_str())
                        .map(|value| value.to_ascii_lowercase())
                        .as_deref(),
                    Some("jpg") | Some("jpeg") | Some("png")
                )
        })
}

fn candidate_checkpoint_valid(
    checkpoint: &Path,
    input: &Path,
    candidate_fps: f64,
    lens0: &Path,
    lens1: &Path,
) -> bool {
    let Ok(bytes) = fs::read(checkpoint) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    let Ok(metadata) = fs::metadata(input) else {
        return false;
    };
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().to_string());
    value.get("sourcePath").and_then(Value::as_str) == Some(input.to_string_lossy().as_ref())
        && value.get("sourceSize").and_then(Value::as_u64) == Some(metadata.len())
        && value.get("sourceModifiedNanos").and_then(Value::as_str) == modified.as_deref()
        && value.get("imageFormat").and_then(Value::as_str) == Some("jpeg-q2")
        && value
            .get("candidateFps")
            .and_then(Value::as_f64)
            .is_some_and(|value| (value - candidate_fps).abs() < 1e-6)
        && has_candidate_image(lens0)
        && has_candidate_image(lens1)
}

fn run_extract(
    app: &AppHandle,
    id: &str,
    manifest: &ProjectManifest,
    control: &JobControl,
) -> Result<Vec<String>, String> {
    if manifest.input_paths.is_empty() {
        return Err("此資料夾沒有原始影片；可直接執行現有影格適用的 Mask 或對齊階段".to_owned());
    }
    let ffmpeg = find_executable("ffmpeg").ok_or("System ffmpeg was not found on PATH")?;
    let ffprobe = find_executable("ffprobe").ok_or("System ffprobe was not found on PATH")?;
    let output = PathBuf::from(&manifest.output_path);
    let base_fps = setting_f64(&manifest.settings, "/extract/baseFps", 2.0).clamp(0.1, 30.0);
    let dense_fps = setting_f64(&manifest.settings, "/extract/denseFps", 8.0).clamp(base_fps, 60.0);
    let skip_blurry = setting_bool(&manifest.settings, "/extract/skipBlurry", true);
    let candidate_fps = if skip_blurry { dense_fps } else { base_fps };
    let total_sources = manifest.input_paths.len().max(1) as f32;
    let mut telemetry_streams = Vec::new();
    let mut normalized_telemetry = Vec::new();
    for (source_index, raw_input) in manifest.input_paths.iter().enumerate() {
        let input = PathBuf::from(raw_input);
        let probe = probe_streams(&ffprobe, &input)?;
        let streams = stream_indices(&probe, "video");
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
        let lens0_candidates = candidate_root.join("lens0");
        let lens1_candidates = candidate_root.join("lens1");
        let checkpoint = candidate_root.join("candidates.complete.json");
        if !candidate_checkpoint_valid(
            &checkpoint,
            &input,
            candidate_fps,
            &lens0_candidates,
            &lens1_candidates,
        ) {
            if candidate_root.exists() {
                fs::remove_dir_all(&candidate_root).map_err(|error| error.to_string())?;
            }
            fs::create_dir_all(&lens0_candidates).map_err(|error| error.to_string())?;
            fs::create_dir_all(&lens1_candidates).map_err(|error| error.to_string())?;
            emit_progress(
                app,
                id,
                &StageName::Extract,
                "decoding",
                source_index as f32 / total_sources,
                format!("正在同步解碼來源 {} 的雙魚眼候選影格", source_index + 1),
                "running",
                false,
            );
            let args = vec![
                "-hide_banner".into(),
                "-nostdin".into(),
                "-y".into(),
                "-i".into(),
                input.to_string_lossy().into_owned(),
                "-map".into(),
                format!("0:{}", streams[0]),
                "-vf".into(),
                format!("fps={candidate_fps}"),
                "-fps_mode".into(),
                "passthrough".into(),
                "-q:v".into(),
                "2".into(),
                lens0_candidates
                    .join("%08d.jpg")
                    .to_string_lossy()
                    .into_owned(),
                "-map".into(),
                format!("0:{}", streams[1]),
                "-vf".into(),
                format!("fps={candidate_fps}"),
                "-fps_mode".into(),
                "passthrough".into(),
                "-q:v".into(),
                "2".into(),
                lens1_candidates
                    .join("%08d.jpg")
                    .to_string_lossy()
                    .into_owned(),
            ];
            run_child(app, id, &ffmpeg, &args, control)?;
            if !has_candidate_image(&lens0_candidates) || !has_candidate_image(&lens1_candidates) {
                return Err(format!(
                    "來源 {} 沒有產生完整雙魚眼候選影格",
                    input.display()
                ));
            }
            let source_size = fs::metadata(&input)
                .map_err(|error| error.to_string())?
                .len();
            let source_modified = fs::metadata(&input)
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos().to_string());
            fs::write(
                &checkpoint,
                serde_json::to_vec_pretty(&json!({
                    "schemaVersion": 2, "sourcePath": input.to_string_lossy(),
                    "sourceSize": source_size, "sourceModifiedNanos": source_modified,
                    "candidateFps": candidate_fps, "imageFormat": "jpeg-q2"
                }))
                .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        }

        let extraction_request = ExtractionRequest {
            lens0_candidates,
            lens1_candidates,
            lens0_output: output.join("images/lens0"),
            lens1_output: output.join("images/lens1"),
            output_prefix: format!("source{source_index:03}_"),
            base_fps,
            candidate_fps,
            dense_fps,
            score_candidates: skip_blurry,
            skip_completed: true,
            metadata_path: Some(metadata.join(format!("source{source_index:03}_selection.json"))),
        };
        let app_clone = app.clone();
        let id_owned = id.to_owned();
        let cancelled = control.cancelled.clone();
        let source_offset = source_index as f32 / total_sources;
        let source_scale = 1.0 / total_sources;
        let summary = extraction::extract_selected_pairs(
            &extraction_request,
            || cancelled.load(Ordering::Acquire),
            move |event| {
                emit_progress(
                    &app_clone,
                    &id_owned,
                    &StageName::Extract,
                    "selecting",
                    source_offset + event.fraction * source_scale,
                    format!("來源 {}：{}", source_index + 1, event.message),
                    "running",
                    false,
                )
            },
        )
        .map_err(|error| error.to_string())?;
        if summary.cancelled {
            return Err("cancelled".to_owned());
        }
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
        "schemaVersion": 1, "canonicalProjection": "native_fisheye", "sources": manifest.input_paths,
        "lensCount": 2, "baseFps": base_fps, "candidateFps": candidate_fps,
        "requestedDenseFps": dense_fps, "skipBlurry": skip_blurry,
        "sharpness": if skip_blurry { "gaussian+laplacian+tenengrad; conservative pair minimum" } else { "disabled" },
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
    let confidence = setting_f64(&manifest.settings, "/mask/confidence", 0.25);
    let request = MaskRequest {
        images_dir: root.join("images"),
        masks_dir: root.join("masks"),
        colmap_masks_dir: root.join("masks_colmap"),
        classes,
        mask_sky: setting_bool(&manifest.settings, "/mask/maskSky", false),
        confidence: if confidence > 1.0 {
            confidence / 100.0
        } else {
            confidence
        }
        .clamp(0.01, 0.99) as f32,
        valid_radius_ratio: 0.497,
        skip_verified: true,
        model_dir: manifest
            .settings
            .pointer("/mask/modelDir")
            .and_then(Value::as_str)
            .filter(|path| !path.trim().is_empty())
            .map(PathBuf::from),
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
        emit_progress(
            &app_clone,
            &id_owned,
            &stage,
            "masking",
            event.fraction,
            event.message,
            "running",
            false,
        );
    })
    .map_err(|error| error.to_string())?;
    if summary.failed > 0 {
        return Err(format!("{} masks failed; see pipeline log", summary.failed));
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
    control: &JobControl,
) -> Result<Vec<String>, String> {
    let colmap = find_executable("colmap").ok_or("COLMAP was not found on PATH")?;
    let root = PathBuf::from(&manifest.output_path);
    write_rig_and_pairs(&root)?;
    let db = root.join("database.db");
    let sparse = root.join("sparse");
    if db.is_file() && sparse_model_exists(&sparse) {
        emit_log(
            app,
            id,
            "info",
            "已驗證現有 COLMAP reconstruction，略過重算",
        );
        return Ok(vec![
            db.to_string_lossy().into_owned(),
            root.join("rig_config.json").to_string_lossy().into_owned(),
            sparse.to_string_lossy().into_owned(),
        ]);
    }
    let cuda_available = crate::doctor::report()
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
    run_child(app, id, &colmap, &args, control)?;
    emit_progress(
        app,
        id,
        &StageName::Align,
        "matching",
        0.3,
        "正在建立受限影像配對",
        "running",
        false,
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
        return Err("COLMAP bootstrap did not produce sparse/0".into());
    }
    emit_progress(
        app,
        id,
        &StageName::Align,
        "rig",
        0.75,
        "正在估計雙鏡頭 rig",
        "running",
        false,
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
    Ok(vec![
        db.to_string_lossy().into_owned(),
        root.join("rig_config.json").to_string_lossy().into_owned(),
        sparse.to_string_lossy().into_owned(),
    ])
}

#[cfg(test)]
mod tests {
    use super::write_rig_and_pairs;
    use std::fs;

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
}
