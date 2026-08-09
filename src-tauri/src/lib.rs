mod doctor;
mod extraction;
mod masking;
mod pipeline;
mod project;
mod telemetry;

use pipeline::{JobManager, StartStageRequest, StartStageResponse};
use project::{CreateProjectRequest, InspectPathsResponse, ProjectManifest};

#[tauri::command]
fn doctor() -> doctor::DoctorReport {
    doctor::report()
}

#[tauri::command]
fn inspect_paths(paths: Vec<String>) -> InspectPathsResponse {
    project::inspect(paths)
}

#[tauri::command]
fn create_project(request: CreateProjectRequest) -> Result<ProjectManifest, String> {
    project::create(request)
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(JobManager::default())
        .invoke_handler(tauri::generate_handler![
            doctor,
            inspect_paths,
            create_project,
            load_project,
            start_stage,
            cancel_job
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
