//! Project discovery and resumable manifest handling.

use crate::doctor;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::process::silent_command;

pub const MANIFEST_FILE: &str = "project.json";
pub const MANIFEST_VERSION: u32 = 1;

static PROJECT_COUNTER: AtomicU64 = AtomicU64::new(1);

fn now_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    seconds.to_string()
}

fn now_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or_default()
}

fn new_project_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    format!(
        "p-{millis}-{}",
        PROJECT_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub enum StageName {
    Extract,
    Mask,
    Align,
}

impl StageName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Extract => "extract",
            Self::Mask => "mask",
            Self::Align => "align",
        }
    }
}

impl std::fmt::Display for StageName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for StageName {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "extract" | "capture" => Ok(Self::Extract),
            "mask" | "masks" => Ok(Self::Mask),
            "align" | "sfm" | "colmap" => Ok(Self::Align),
            other => Err(format!("unknown pipeline stage: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StageStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}

impl Default for StageStatus {
    fn default() -> Self {
        Self::Pending
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StageCheckpoint {
    #[serde(default)]
    pub status: StageStatus,
    #[serde(default)]
    pub progress: f32,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub artifacts: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub updated_at: String,
    /// Wall-clock timestamp (milliseconds since Unix epoch) for the current
    /// run.  Optional so manifests written before timing support remain
    /// readable without migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at_ms: Option<u64>,
    /// Wall-clock timestamp (milliseconds since Unix epoch) when the current
    /// run reached a terminal state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at_ms: Option<u64>,
    /// Monotonic elapsed duration of the run, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl Default for StageCheckpoint {
    fn default() -> Self {
        Self {
            status: StageStatus::Pending,
            progress: 0.0,
            message: String::new(),
            artifacts: Vec::new(),
            warnings: Vec::new(),
            updated_at: now_timestamp(),
            started_at_ms: None,
            finished_at_ms: None,
            duration_ms: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectManifest {
    #[serde(default = "manifest_version")]
    pub manifest_version: u32,
    #[serde(default)]
    pub project_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub root_path: String,
    #[serde(default)]
    pub input_paths: Vec<String>,
    #[serde(default)]
    pub output_path: String,
    #[serde(default)]
    pub settings: Value,
    #[serde(default = "default_stages")]
    pub stages: BTreeMap<String, StageCheckpoint>,
    #[serde(default)]
    pub capabilities: BTreeMap<String, bool>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
}

fn manifest_version() -> u32 {
    MANIFEST_VERSION
}

fn default_stages() -> BTreeMap<String, StageCheckpoint> {
    ["extract", "mask", "align"]
        .into_iter()
        .map(|name| (name.to_owned(), StageCheckpoint::default()))
        .collect()
}

impl ProjectManifest {
    pub fn stage(&self, stage: &StageName) -> StageCheckpoint {
        self.stages.get(stage.as_str()).cloned().unwrap_or_default()
    }

    pub fn set_stage(&mut self, stage: &StageName, checkpoint: StageCheckpoint) {
        self.stages.insert(stage.as_str().to_owned(), checkpoint);
        self.updated_at = now_timestamp();
    }

    pub fn manifest_path(&self) -> PathBuf {
        PathBuf::from(&self.root_path).join(MANIFEST_FILE)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateProjectRequest {
    #[serde(default)]
    pub input_paths: Vec<String>,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub settings: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateQueuedProjectRequest {
    pub project_path: String,
    pub name: String,
    #[serde(default)]
    pub input_paths: Vec<String>,
    pub settings: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceInspection {
    pub path: String,
    pub name: String,
    pub size: u64,
    pub valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lens_count: Option<u32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectInspection {
    pub path: String,
    pub manifest_path: Option<String>,
    pub status: String,
    pub has_manifest: bool,
    pub stages: BTreeMap<String, StageCheckpoint>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectPathsResponse {
    /// `sources`, `project`, or `mixed`.
    pub kind: String,
    pub sources: Vec<SourceInspection>,
    pub project: Option<ProjectInspection>,
    pub suggested_output_path: Option<String>,
    pub warnings: Vec<String>,
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn source_extension(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("osv")
            | Some("mp4")
            | Some("mov")
            | Some("mkv")
            | Some("avi")
            | Some("webm")
            | Some("m4v")
            | Some("mts")
            | Some("m2ts")
            | Some("ts")
    )
}

fn collect_source_paths(path: &Path, output: &mut Vec<PathBuf>) {
    if path.is_file() {
        if source_extension(path) {
            output.push(path.to_path_buf());
        }
        return;
    }
    if !path.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    // A selected source directory is intentionally shallow.  Camera exports
    // may contain unrelated proxy media in nested folders; callers can pass
    // those folders explicitly when they are intended inputs.
    for entry in entries.flatten() {
        let candidate = entry.path();
        if candidate.is_file() && source_extension(&candidate) {
            output.push(candidate);
        }
    }
}

fn parse_fraction(value: Option<&str>) -> Option<f64> {
    let value = value?.trim();
    if value.is_empty() || value == "0/0" || value.eq_ignore_ascii_case("N/A") {
        return None;
    }
    if let Some((numerator, denominator)) = value.split_once('/') {
        let n = numerator.parse::<f64>().ok()?;
        let d = denominator.parse::<f64>().ok()?;
        return (d.abs() > f64::EPSILON).then_some(n / d);
    }
    value.parse::<f64>().ok()
}

fn probe_source(path: &Path) -> io::Result<Value> {
    let Some(ffprobe) = doctor::find_executable("ffprobe") else {
        return Err(io::Error::new(io::ErrorKind::NotFound, "ffprobe not found"));
    };
    let output = silent_command(ffprobe)
        .args([
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
        ])
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn inspect_source(path: &Path) -> SourceInspection {
    let path = absolute_path(path);
    let metadata = fs::metadata(&path).ok();
    let size = metadata.as_ref().map_or(0, |m| m.len());
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("source")
        .to_owned();
    let mut inspection = SourceInspection {
        path: path.to_string_lossy().into_owned(),
        name,
        size,
        valid: metadata.is_some() && path.is_file() && source_extension(&path),
        duration: None,
        fps: None,
        width: None,
        height: None,
        lens_count: None,
        warnings: Vec::new(),
    };
    if !inspection.valid {
        inspection
            .warnings
            .push("Expected a supported video/OSV file".to_owned());
        return inspection;
    }

    match probe_source(&path) {
        Ok(probe) => {
            let streams = probe
                .get("streams")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let videos: Vec<&Value> = streams
                .iter()
                .filter(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"))
                .collect();
            inspection.lens_count = Some(videos.len() as u32);
            inspection.width = videos
                .first()
                .and_then(|s| s.get("width"))
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok());
            inspection.height = videos
                .first()
                .and_then(|s| s.get("height"))
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok());
            inspection.fps = videos.first().and_then(|s| {
                parse_fraction(s.get("avg_frame_rate").and_then(Value::as_str))
                    .or_else(|| parse_fraction(s.get("r_frame_rate").and_then(Value::as_str)))
            });
            inspection.duration = probe
                .get("format")
                .and_then(|v| v.get("duration"))
                .and_then(Value::as_str)
                .and_then(|v| v.parse::<f64>().ok());
            if videos.is_empty() {
                inspection.valid = false;
                inspection
                    .warnings
                    .push("No video stream was found by ffprobe".to_owned());
            } else if path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("osv"))
                && videos.len() < 2
            {
                inspection.warnings.push("OSV input has fewer than two video streams; dual-fisheye extraction may be unavailable".to_owned());
            }
        }
        Err(error) => {
            // Keep the file selectable when ffprobe is not installed.  The
            // extract stage will produce a precise tool error later.
            inspection
                .warnings
                .push(format!("ffprobe metadata unavailable: {error}"));
        }
    }
    inspection
}

fn locate_manifest(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        let file_name = path.file_name()?.to_str()?.to_ascii_lowercase();
        if file_name == MANIFEST_FILE || file_name == "manifest.json" {
            return Some(path.to_path_buf());
        }
    }
    if !path.is_dir() {
        return None;
    }
    let candidates = [
        path.join(MANIFEST_FILE),
        path.join("manifest.json"),
        path.join(".gs360").join(MANIFEST_FILE),
    ];
    candidates.into_iter().find(|candidate| candidate.is_file())
}

fn infer_partial_project(path: &Path) -> Option<ProjectInspection> {
    if !path.is_dir() {
        return None;
    }
    let evidence = ["images", "capture", "metadata", "masks", "rig_config.json"]
        .iter()
        .any(|name| path.join(name).exists());
    evidence.then(|| ProjectInspection {
        path: path.to_string_lossy().into_owned(),
        manifest_path: None,
        status: "partial".to_owned(),
        has_manifest: false,
        stages: infer_stage_checkpoints(path),
    })
}

fn relative_files(path: &Path) -> BTreeSet<PathBuf> {
    fn visit(root: &Path, current: &Path, files: &mut BTreeSet<PathBuf>) {
        let Ok(entries) = fs::read_dir(current) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(root, &path, files);
            } else if path.is_file() {
                if let Ok(relative) = path.strip_prefix(root) {
                    files.insert(relative.to_path_buf());
                }
            }
        }
    }

    let mut files = BTreeSet::new();
    if path.is_dir() {
        visit(path, path, &mut files);
    }
    files
}

fn matched_lens_frames(root: &Path) -> BTreeSet<PathBuf> {
    let lens0 = relative_files(&root.join("images/lens0"));
    let lens1 = relative_files(&root.join("images/lens1"));
    lens0.intersection(&lens1).cloned().collect()
}

fn masks_cover_images(root: &Path) -> bool {
    let images = relative_files(&root.join("images"));
    if images.is_empty() {
        return false;
    }
    images.into_iter().all(|relative| {
        // Regular masks are canonical PNGs keyed by the image stem. Project
        // recovery only needs this canonical mask for each source image.
        let regular = root.join("masks").join(relative.with_extension("png"));
        regular.is_file()
    })
}

fn valid_colmap_database(path: &Path) -> bool {
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

fn valid_align_checkpoint(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let Ok(checkpoint) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    checkpoint.get("schema_version").and_then(Value::as_u64) == Some(2)
        && checkpoint.get("completed").and_then(Value::as_bool) == Some(true)
        && checkpoint
            .get("fingerprint")
            .and_then(Value::as_str)
            .is_some_and(|fingerprint| !fingerprint.is_empty())
}

fn valid_sparse_model(root: &Path) -> bool {
    fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .any(|model| {
            ["rigs", "cameras", "frames", "images", "points3D"]
                .iter()
                .all(|name| {
                    ["bin", "txt"].iter().any(|extension| {
                        fs::metadata(model.join(format!("{name}.{extension}")))
                            .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
                    })
                })
        })
}

fn completed_checkpoint(message: &str, artifacts: Vec<String>) -> StageCheckpoint {
    StageCheckpoint {
        status: StageStatus::Completed,
        progress: 1.0,
        message: message.to_owned(),
        artifacts,
        warnings: Vec::new(),
        updated_at: now_timestamp(),
        started_at_ms: None,
        finished_at_ms: None,
        duration_ms: None,
    }
}

fn infer_stage_checkpoints(root: &Path) -> BTreeMap<String, StageCheckpoint> {
    let mut stages = default_stages();
    let images = root.join("images");
    if matched_lens_frames(root).len() >= 2 {
        stages.insert(
            "extract".to_owned(),
            completed_checkpoint(
                "已找到現有的雙魚眼影格",
                vec![images.to_string_lossy().into_owned()],
            ),
        );
    }
    let masks = root.join("masks");
    if masks_cover_images(root) {
        stages.insert(
            "mask".to_owned(),
            completed_checkpoint("已找到現有遮罩", vec![masks.to_string_lossy().into_owned()]),
        );
    }
    let sparse = root.join("sparse");
    if valid_align_checkpoint(&root.join("metadata/align.checkpoint.json"))
        && valid_colmap_database(&root.join("database.db"))
        && valid_sparse_model(&sparse)
    {
        stages.insert(
            "align".to_owned(),
            completed_checkpoint(
                "已找到現有的 COLMAP 重建結果",
                vec![
                    root.join("database.db").to_string_lossy().into_owned(),
                    sparse.to_string_lossy().into_owned(),
                ],
            ),
        );
    }
    stages
}

fn inspect_project(path: &Path) -> Option<ProjectInspection> {
    let root = if path.is_file() {
        path.parent().unwrap_or(path)
    } else {
        path
    };
    if let Some(manifest_path) = locate_manifest(path) {
        if let Ok(bytes) = fs::read(&manifest_path) {
            if let Ok(manifest) = serde_json::from_slice::<ProjectManifest>(&bytes) {
                return Some(ProjectInspection {
                    path: root.to_string_lossy().into_owned(),
                    manifest_path: Some(manifest_path.to_string_lossy().into_owned()),
                    status: "ready".to_owned(),
                    has_manifest: true,
                    stages: manifest.stages,
                });
            }
        }
    }
    infer_partial_project(root)
}

pub fn inspect(paths: Vec<String>) -> InspectPathsResponse {
    let mut source_paths = Vec::new();
    let mut project = None;
    let mut warnings = Vec::new();
    for raw in paths {
        let path = absolute_path(Path::new(&raw));
        if let Some(found) = inspect_project(&path) {
            project = Some(found);
        }
        collect_source_paths(&path, &mut source_paths);
        if !path.exists() {
            warnings.push(format!("Path does not exist: {}", path.display()));
        }
    }
    source_paths.sort();
    source_paths.dedup();
    let sources: Vec<_> = source_paths
        .iter()
        .map(|path| inspect_source(path))
        .collect();
    if let Some(current_project) = &project {
        if current_project.status == "partial" {
            warnings.push(
                "A partial project was found; completed checkpoints can be resumed".to_owned(),
            );
        }
    }
    let kind = match (sources.is_empty(), project.is_some()) {
        (false, true) => "mixed",
        (false, false) => "sources",
        (true, true) => "project",
        (true, false) => "sources",
    }
    .to_owned();
    let suggested_output_path = if let Some(current_project) = &project {
        Some(current_project.path.clone())
    } else {
        sources.first().map(|source| {
            let source_path = PathBuf::from(&source.path);
            let stem = source_path
                .file_stem()
                .and_then(|v| v.to_str())
                .filter(|v| !v.is_empty())
                .unwrap_or("capture");
            source_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!("colmap-{stem}"))
                .to_string_lossy()
                .into_owned()
        })
    };
    InspectPathsResponse {
        kind,
        sources,
        project,
        suggested_output_path,
        warnings,
    }
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "Manifest has no parent directory".to_owned())?;
    fs::create_dir_all(parent).map_err(|e| format!("create manifest directory: {e}"))?;
    let temp = path.with_extension("json.partial");
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| format!("serialize manifest: {e}"))?;
    fs::write(&temp, bytes).map_err(|e| format!("write manifest checkpoint: {e}"))?;
    #[cfg(not(windows))]
    {
        fs::rename(&temp, path).map_err(|e| format!("commit manifest checkpoint: {e}"))
    }
    #[cfg(windows)]
    {
        let backup = path.with_extension("json.backup");
        if path.exists() {
            let _ = fs::remove_file(&backup);
            fs::rename(path, &backup).map_err(|e| format!("prepare manifest checkpoint: {e}"))?;
        }
        match fs::rename(&temp, path) {
            Ok(()) => {
                let _ = fs::remove_file(backup);
                Ok(())
            }
            Err(error) => {
                if backup.exists() {
                    let _ = fs::rename(&backup, path);
                }
                Err(format!("commit manifest checkpoint: {error}"))
            }
        }
    }
}

pub fn save_manifest(manifest: &ProjectManifest) -> Result<(), String> {
    write_json_atomic(&manifest.manifest_path(), manifest)
}

pub fn load(path: impl AsRef<Path>) -> Result<ProjectManifest, String> {
    let path = absolute_path(path.as_ref());
    let manifest_path = match locate_manifest(&path) {
        Some(manifest_path) => manifest_path,
        None if infer_partial_project(&path).is_some() => {
            let root = if path.is_dir() {
                path.clone()
            } else {
                path.parent().unwrap_or(Path::new(".")).to_path_buf()
            };
            let name = root
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("GS360 project")
                .to_owned();
            let mut capabilities = BTreeMap::new();
            capabilities.insert("nativeFisheye".to_owned(), true);
            capabilities.insert("telemetry".to_owned(), false);
            capabilities.insert("imuApplied".to_owned(), false);
            let manifest = ProjectManifest {
                manifest_version: MANIFEST_VERSION,
                project_id: new_project_id(),
                name,
                root_path: root.to_string_lossy().into_owned(),
                input_paths: Vec::new(),
                output_path: root.to_string_lossy().into_owned(),
                settings: json!({}),
                stages: infer_stage_checkpoints(&root),
                capabilities,
                warnings: vec!["已依現有處理結果復原專案資訊".to_owned()],
                created_at: now_timestamp(),
                updated_at: now_timestamp(),
            };
            save_manifest(&manifest)?;
            manifest.manifest_path()
        }
        None => return Err(format!("No project manifest found at {}", path.display())),
    };
    let bytes = fs::read(&manifest_path).map_err(|e| format!("read project manifest: {e}"))?;
    let mut manifest: ProjectManifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse project manifest {}: {e}", manifest_path.display()))?;
    if manifest.root_path.is_empty() {
        manifest.root_path = manifest_path
            .parent()
            .unwrap_or(Path::new("."))
            .to_string_lossy()
            .into_owned();
    }
    if manifest.output_path.is_empty() {
        manifest.output_path = manifest.root_path.clone();
    }
    if manifest.project_id.is_empty() {
        manifest.project_id = new_project_id();
    }
    if manifest.stages.is_empty() {
        manifest.stages = default_stages();
    }
    let mut recovered_running_stage = false;
    let recovery_finished_at_ms = now_timestamp_ms();
    for checkpoint in manifest.stages.values_mut() {
        if matches!(checkpoint.status, StageStatus::Running) {
            checkpoint.status = StageStatus::Cancelled;
            checkpoint.message = "上次處理中斷，此階段可繼續執行".to_owned();
            checkpoint.updated_at = now_timestamp();
            checkpoint.finished_at_ms = Some(recovery_finished_at_ms);
            checkpoint.duration_ms = checkpoint
                .started_at_ms
                .and_then(|started| recovery_finished_at_ms.checked_sub(started));
            recovered_running_stage = true;
        }
    }
    if recovered_running_stage {
        manifest.updated_at = now_timestamp();
        save_manifest(&manifest)?;
    }
    Ok(manifest)
}

pub fn create(request: CreateProjectRequest) -> Result<ProjectManifest, String> {
    if request.input_paths.is_empty() {
        return Err("At least one input path is required".to_owned());
    }
    let input_paths: Vec<PathBuf> = request
        .input_paths
        .iter()
        .map(|path| absolute_path(Path::new(path)))
        .collect();
    let first = input_paths
        .first()
        .ok_or_else(|| "At least one input path is required".to_owned())?;
    let stem = first
        .file_stem()
        .or_else(|| first.file_name())
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("capture");
    let output_path = request
        .output_path
        .as_deref()
        .map(|path| absolute_path(Path::new(path)))
        .unwrap_or_else(|| {
            first
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!("colmap-{stem}"))
        });
    fs::create_dir_all(&output_path).map_err(|e| format!("create project directory: {e}"))?;

    let manifest_path = output_path.join(MANIFEST_FILE);
    if manifest_path.is_file() {
        return load(&output_path);
    }
    for relative in [
        "capture",
        "images/lens0",
        "images/lens1",
        "masks",
        "metadata",
        "sparse",
    ] {
        fs::create_dir_all(output_path.join(relative))
            .map_err(|e| format!("create project layout {relative}: {e}"))?;
    }

    let mut warnings = Vec::new();
    for input in &input_paths {
        if !input.exists() {
            warnings.push(format!(
                "Input path does not exist yet: {}",
                input.display()
            ));
        }
    }
    let name = request.name.unwrap_or_else(|| {
        output_path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or(stem)
            .to_owned()
    });
    let mut capabilities = BTreeMap::new();
    capabilities.insert("nativeFisheye".to_owned(), true);
    capabilities.insert("telemetry".to_owned(), false);
    capabilities.insert("imuApplied".to_owned(), false);
    let mut manifest = ProjectManifest {
        manifest_version: MANIFEST_VERSION,
        project_id: new_project_id(),
        name,
        root_path: output_path.to_string_lossy().into_owned(),
        input_paths: input_paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        output_path: output_path.to_string_lossy().into_owned(),
        settings: request.settings.unwrap_or_else(|| json!({})),
        stages: default_stages(),
        capabilities,
        warnings,
        created_at: now_timestamp(),
        updated_at: now_timestamp(),
    };
    save_manifest(&manifest)?;
    // Keep the in-memory value consistent with the persisted representation in
    // case a filesystem clock has coarse resolution.
    manifest.updated_at = now_timestamp();
    Ok(manifest)
}

pub fn update_queued(request: UpdateQueuedProjectRequest) -> Result<ProjectManifest, String> {
    let mut manifest = load(&request.project_path)?;
    if manifest
        .stages
        .values()
        .any(|stage| {
            !matches!(stage.status, StageStatus::Pending)
                || stage.progress > 0.0
                || stage.started_at_ms.is_some()
                || stage.finished_at_ms.is_some()
                || stage.duration_ms.is_some()
                || !stage.artifacts.is_empty()
        })
    {
        return Err("Only a project that has not started can be edited".to_owned());
    }
    if request.input_paths.is_empty() {
        return Err("At least one input path is required".to_owned());
    }
    let name = request.name.trim();
    if !name.is_empty() {
        manifest.name = name.to_owned();
    }
    manifest.input_paths = request
        .input_paths
        .iter()
        .map(|path| absolute_path(Path::new(path)).to_string_lossy().into_owned())
        .collect();
    manifest.settings = request.settings;
    manifest.updated_at = now_timestamp();
    save_manifest(&manifest)?;
    Ok(manifest)
}

#[allow(dead_code)]
pub fn update_stage(
    manifest: &mut ProjectManifest,
    stage: &StageName,
    status: StageStatus,
    progress: f32,
    message: impl Into<String>,
    artifacts: Vec<String>,
    warnings: Vec<String>,
) -> Result<(), String> {
    update_stage_timed(
        manifest, stage, status, progress, message, artifacts, warnings, None,
    )
}

/// Update a stage checkpoint and, when supplied, persist the monotonic timer
/// for the current invocation.  `Instant` is intentionally passed by the
/// caller: it cannot be serialized, but it is the only clock suitable for an
/// elapsed duration that is not affected by wall-clock adjustments.
pub fn update_stage_timed(
    manifest: &mut ProjectManifest,
    stage: &StageName,
    status: StageStatus,
    progress: f32,
    message: impl Into<String>,
    artifacts: Vec<String>,
    warnings: Vec<String>,
    started_at: Option<Instant>,
) -> Result<(), String> {
    let now_ms = now_timestamp_ms();
    let previous = manifest.stage(stage);
    let (started_at_ms, finished_at_ms, duration_ms) = match &status {
        StageStatus::Running => (Some(now_ms), None, None),
        StageStatus::Completed | StageStatus::Cancelled | StageStatus::Failed => {
            let started_at_ms = previous.started_at_ms;
            let duration_ms = started_at
                .map(|instant| instant.elapsed().as_millis().min(u128::from(u64::MAX)) as u64)
                .or_else(|| started_at_ms.and_then(|value| now_ms.checked_sub(value)));
            (started_at_ms, Some(now_ms), duration_ms)
        }
        StageStatus::Pending => (None, None, None),
    };
    manifest.set_stage(
        stage,
        StageCheckpoint {
            status,
            progress: progress.clamp(0.0, 1.0),
            message: message.into(),
            artifacts,
            warnings,
            updated_at: now_timestamp(),
            started_at_ms,
            finished_at_ms,
            duration_ms,
        },
    );
    save_manifest(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    static TEST_TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn queued_project_can_be_edited_but_started_project_cannot() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.osv");
        fs::write(&source, b"test").unwrap();
        let output = root.join("project");
        let manifest = create(CreateProjectRequest {
            input_paths: vec![source.to_string_lossy().into_owned()],
            output_path: Some(output.to_string_lossy().into_owned()),
            name: Some("before".to_owned()),
            settings: Some(json!({"extract": {"baseFps": 3}})),
        })
        .unwrap();

        let updated = update_queued(UpdateQueuedProjectRequest {
            project_path: manifest.root_path.clone(),
            name: "after".to_owned(),
            input_paths: manifest.input_paths.clone(),
            settings: json!({"extract": {"baseFps": 4}}),
        })
        .unwrap();
        assert_eq!(updated.name, "after");
        assert_eq!(updated.settings.pointer("/extract/baseFps"), Some(&json!(4)));

        let mut started = updated;
        update_stage(
            &mut started,
            &StageName::Extract,
            StageStatus::Completed,
            1.0,
            "done",
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert!(update_queued(UpdateQueuedProjectRequest {
            project_path: started.root_path.clone(),
            name: "too late".to_owned(),
            input_paths: started.input_paths.clone(),
            settings: json!({}),
        })
        .is_err());

        let _ = fs::remove_dir_all(root);
    }

    fn temp_dir() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let counter = TEST_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "gs360studio-project-test-{}-{id}-{counter}",
            std::process::id()
        ))
    }

    #[test]
    fn stage_names_use_wire_values() {
        assert_eq!(
            serde_json::to_string(&StageName::Extract).unwrap(),
            "\"extract\""
        );
        assert_eq!("align".parse::<StageName>().unwrap(), StageName::Align);
    }

    #[test]
    fn old_manifest_without_timing_fields_is_backward_compatible() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let manifest = json!({
            "manifestVersion": MANIFEST_VERSION,
            "projectId": "legacy",
            "name": "Legacy",
            "rootPath": root,
            "outputPath": root,
            "stages": {"extract": {"status": "pending"}}
        });
        fs::write(
            root.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let loaded = load(&root).unwrap();
        let extract = loaded.stage(&StageName::Extract);
        assert!(extract.started_at_ms.is_none());
        assert!(extract.finished_at_ms.is_none());
        assert!(extract.duration_ms.is_none());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn timed_stage_checkpoint_persists_elapsed_duration() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let mut manifest = ProjectManifest {
            manifest_version: MANIFEST_VERSION,
            project_id: "timed".to_owned(),
            name: "Timed".to_owned(),
            root_path: root.to_string_lossy().into_owned(),
            input_paths: Vec::new(),
            output_path: root.to_string_lossy().into_owned(),
            settings: json!({}),
            stages: default_stages(),
            capabilities: BTreeMap::new(),
            warnings: Vec::new(),
            created_at: now_timestamp(),
            updated_at: now_timestamp(),
        };
        save_manifest(&manifest).unwrap();

        let started_at = Instant::now();
        update_stage_timed(
            &mut manifest,
            &StageName::Mask,
            StageStatus::Running,
            0.0,
            "running",
            Vec::new(),
            Vec::new(),
            Some(started_at),
        )
        .unwrap();
        let running = manifest.stage(&StageName::Mask);
        assert!(running.started_at_ms.is_some());
        assert!(running.finished_at_ms.is_none());
        assert!(running.duration_ms.is_none());

        std::thread::sleep(Duration::from_millis(2));
        update_stage_timed(
            &mut manifest,
            &StageName::Mask,
            StageStatus::Completed,
            1.0,
            "completed",
            Vec::new(),
            Vec::new(),
            Some(started_at),
        )
        .unwrap();
        let completed = load(&root).unwrap().stage(&StageName::Mask);
        assert!(completed
            .finished_at_ms
            .zip(completed.started_at_ms)
            .is_some_and(|(finished, started)| finished >= started));
        assert!(completed.duration_ms.is_some());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn create_uses_first_input_sibling_as_default_output() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let input = root.join("capture.OSV");
        fs::write(&input, b"placeholder").unwrap();
        let manifest = create(CreateProjectRequest {
            input_paths: vec![input.to_string_lossy().into_owned()],
            output_path: None,
            name: None,
            settings: None,
        })
        .unwrap();
        assert_eq!(
            manifest.output_path,
            root.join("colmap-capture").to_string_lossy()
        );
        assert!(Path::new(&manifest.root_path).join(MANIFEST_FILE).is_file());
        assert!(!Path::new(&manifest.root_path).join("masks_colmap").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn inspect_supports_multiple_source_files_without_manifest() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.mp4"), b"not video").unwrap();
        fs::write(root.join("b.OSV"), b"not video").unwrap();
        let result = inspect(vec![root.to_string_lossy().into_owned()]);
        assert_eq!(result.kind, "sources");
        assert_eq!(result.sources.len(), 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_recovers_partial_project_and_infers_completed_stages() {
        let root = temp_dir();
        fs::create_dir_all(root.join("images/lens0")).unwrap();
        fs::create_dir_all(root.join("images/lens1")).unwrap();
        fs::create_dir_all(root.join("masks/lens0")).unwrap();
        fs::create_dir_all(root.join("masks/lens1")).unwrap();
        for lens in ["lens0", "lens1"] {
            for frame in ["frame1.jpg", "frame2.jpg"] {
                fs::write(root.join("images").join(lens).join(frame), b"frame").unwrap();
                fs::write(
                    root.join("masks")
                        .join(lens)
                        .join(Path::new(frame).with_extension("png")),
                    b"mask",
                )
                .unwrap();
            }
        }
        fs::create_dir_all(root.join("sparse/0")).unwrap();
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(
            root.join("metadata/align.checkpoint.json"),
            br#"{"schema_version":2,"fingerprint":"test","completed":true}"#,
        )
        .unwrap();
        let mut database = vec![0_u8; 512];
        database[..16].copy_from_slice(b"SQLite format 3\0");
        database[16..18].copy_from_slice(&512_u16.to_be_bytes());
        fs::write(root.join("database.db"), database).unwrap();
        for name in [
            "rigs.bin",
            "cameras.bin",
            "frames.bin",
            "images.bin",
            "points3D.bin",
        ] {
            fs::write(root.join("sparse/0").join(name), b"model").unwrap();
        }

        let inspection = inspect(vec![root.to_string_lossy().into_owned()]);
        let partial = inspection.project.expect("partial project");
        assert_eq!(partial.status, "partial");
        assert!(matches!(
            partial.stages["extract"].status,
            StageStatus::Completed
        ));
        assert!(matches!(
            partial.stages["mask"].status,
            StageStatus::Completed
        ));
        assert_eq!(
            partial.stages["mask"].artifacts,
            vec![root.join("masks").to_string_lossy().into_owned()]
        );
        assert!(matches!(
            partial.stages["align"].status,
            StageStatus::Completed
        ));

        let manifest = load(&root).expect("recover manifest");
        assert!(root.join(MANIFEST_FILE).is_file());
        assert!(manifest.input_paths.is_empty());
        assert!(manifest
            .warnings
            .iter()
            .any(|warning| warning.contains("復原專案資訊")));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn incomplete_masks_are_not_inferred_as_completed() {
        let root = temp_dir();
        for lens in ["lens0", "lens1"] {
            fs::create_dir_all(root.join("images").join(lens)).unwrap();
            fs::write(root.join("images").join(lens).join("frame.png"), b"frame").unwrap();
        }
        fs::create_dir_all(root.join("masks/lens0")).unwrap();
        fs::write(root.join("masks/lens0/frame.png"), b"mask").unwrap();

        let stages = infer_stage_checkpoints(&root);
        assert!(matches!(stages["mask"].status, StageStatus::Pending));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_marks_interrupted_running_stage_as_resumable() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let mut manifest = ProjectManifest {
            manifest_version: MANIFEST_VERSION,
            project_id: "interrupted".to_owned(),
            name: "Interrupted".to_owned(),
            root_path: root.to_string_lossy().into_owned(),
            input_paths: Vec::new(),
            output_path: root.to_string_lossy().into_owned(),
            settings: json!({}),
            stages: default_stages(),
            capabilities: BTreeMap::new(),
            warnings: Vec::new(),
            created_at: now_timestamp(),
            updated_at: now_timestamp(),
        };
        manifest.stages.get_mut("mask").unwrap().status = StageStatus::Running;
        save_manifest(&manifest).unwrap();

        let recovered = load(&root).unwrap();
        assert!(matches!(
            recovered.stages["mask"].status,
            StageStatus::Cancelled
        ));
        assert!(recovered.stages["mask"].message.contains("可繼續執行"));
        assert!(recovered.stages["mask"].finished_at_ms.is_some());
        let persisted = load(&root).unwrap();
        assert!(matches!(
            persisted.stages["mask"].status,
            StageStatus::Cancelled
        ));
        let _ = fs::remove_dir_all(root);
    }
}
