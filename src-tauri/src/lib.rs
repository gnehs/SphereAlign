mod camera_adapter;
mod cli;
mod colmap_feature_cache;
mod colmap_priors;
mod color;
mod doctor;
mod extraction;
mod fisheye;
mod gravity_alignment;
mod imu_calibration;
mod masking;
mod media_preview;
mod orientation_constraints;
mod pipeline;
mod process;
mod project;
mod reconstruction_benchmark;
mod telemetry;
mod visual_retrieval;

use pipeline::{JobManager, StartStageRequest, StartStageResponse};
use project::{
    CreateProjectRequest, InspectPathsResponse, ProjectManifest, UpdateQueuedProjectRequest,
};

#[tauri::command]
fn doctor(colmap_path: Option<String>) -> doctor::DoctorReport {
    doctor::report(colmap_path.as_deref())
}

#[tauri::command]
fn inspect_paths(paths: Vec<String>) -> InspectPathsResponse {
    project::inspect(paths)
}

#[tauri::command]
fn detect_color_profiles(paths: Vec<String>) -> Vec<color::ColorProfilePathInspection> {
    color::detect_paths(paths)
}

#[tauri::command]
async fn source_preview(path: String) -> Result<tauri::ipc::Response, String> {
    let bytes =
        tauri::async_runtime::spawn_blocking(move || media_preview::extract_first_frame(path))
            .await
            .map_err(|_| "預覽處理程序未完成".to_owned())??;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
fn create_project(request: CreateProjectRequest) -> Result<ProjectManifest, String> {
    project::create(request)
}

#[tauri::command]
fn update_queued_project(request: UpdateQueuedProjectRequest) -> Result<ProjectManifest, String> {
    project::update_queued(request)
}

#[tauri::command]
fn load_project(path: String) -> Result<ProjectManifest, String> {
    project::load(path)
}

#[tauri::command]
fn start_stage(
    app: tauri::AppHandle,
    jobs: tauri::State<'_, JobManager>,
    request: StartStageRequest,
) -> Result<StartStageResponse, String> {
    pipeline::start_stage(app, &jobs, request)
}

#[tauri::command]
fn cancel_job(jobs: tauri::State<'_, JobManager>, job_id: String) -> bool {
    jobs.cancel(&job_id)
}

#[tauri::command]
fn generate_benchmark_report(
    project_path: String,
    variant: reconstruction_benchmark::BenchmarkVariant,
) -> Result<reconstruction_benchmark::ReconstructionBenchmarkReport, String> {
    let root = std::path::PathBuf::from(project_path);
    let mut request = reconstruction_benchmark::BenchmarkRequest::new(variant);
    let text_model = root.join("metadata/final-model-text");
    if text_model.is_dir() {
        request.model_dir = Some(text_model);
    }
    let output = root
        .join("metadata")
        .join(format!("benchmark_{}.json", variant.as_str()));
    reconstruction_benchmark::write_benchmark_report(&root, &request, &output)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(JobManager::default())
        .invoke_handler(tauri::generate_handler![
            doctor,
            inspect_paths,
            detect_color_profiles,
            source_preview,
            create_project,
            update_queued_project,
            load_project,
            start_stage,
            cancel_job,
            generate_benchmark_report
        ])
        .run(app_context())
        .expect("error while running tauri application");
}

pub fn run_cli(args: Vec<String>) -> Result<(), String> {
    let mut context = app_context();
    context.config_mut().app.windows.clear();
    let app = tauri::Builder::default()
        .build(context)
        .map_err(|error| error.to_string())?;
    let handle = app.handle().clone();
    cli::run(&handle, args)
}

fn app_context() -> tauri::Context<tauri::Wry> {
    // Keep this macro expanded exactly once per crate. On macOS each expansion
    // embeds `_EMBED_INFO_PLIST`, so separate GUI/CLI expansions fail to link.
    tauri::generate_context!()
}
