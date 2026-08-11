use crate::pipeline::{self, JobManager, StartStageRequest};
use crate::project::{self, CreateProjectRequest, StageName, StageStatus};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Listener};

#[derive(Debug)]
struct AbcArgs {
    inputs: Vec<String>,
    output_root: PathBuf,
    colmap: String,
    gpu_index: String,
    variants: Vec<String>,
    profile_override: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VariantResult {
    variant: String,
    project_path: String,
    quality_profile: String,
    elapsed_ms: u64,
    benchmark_paths: Vec<String>,
    metrics: Option<BenchmarkMetrics>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkMetrics {
    selected_rig_frames: u64,
    any_registered_rig_frames: u64,
    complete_registered_rig_frames: u64,
    complete_registered_percent: f64,
    pair_count: u64,
    points_3d: u64,
    median_track_length: f64,
    median_reprojection_error_px: f64,
    connected_components: u64,
    largest_component_images: u64,
    extract_ms: u64,
    align_ms: u64,
    effective_mapper: String,
    gravity_prior_applied: bool,
    gravity_coverage_ratio: Option<f64>,
}

pub fn run(app: &AppHandle, args: Vec<String>) -> Result<(), String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(usage());
    };
    match command {
        "abc" => run_abc(app, parse_abc(&args[1..])?),
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown CLI command: {other}\n{}", usage())),
    }
}

fn usage() -> String {
    "Usage: gs360studio-cli abc --input <capture.osv> [--input <capture.osv> ...] --output-root <new-or-empty-directory> --colmap <colmap.exe> [--gpu-index 0] [--variants A,B,C] [--profile-override baseline|tuned]".to_owned()
}

fn parse_abc(args: &[String]) -> Result<AbcArgs, String> {
    let mut inputs = Vec::new();
    let mut output_root = None;
    let mut colmap = None;
    let mut gpu_index = "0".to_owned();
    let mut variants = vec!["A".to_owned(), "B".to_owned(), "C".to_owned()];
    let mut profile_override = None;
    let mut index = 0;
    while index < args.len() {
        let key = &args[index];
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--input" => inputs.push(value.clone()),
            "--output-root" => output_root = Some(PathBuf::from(value)),
            "--colmap" => colmap = Some(value.clone()),
            "--gpu-index" => gpu_index = value.clone(),
            "--variants" => {
                variants = value
                    .split(',')
                    .map(|value| value.trim().to_ascii_uppercase())
                    .filter(|value| !value.is_empty())
                    .collect();
                if variants.is_empty()
                    || variants
                        .iter()
                        .any(|value| !matches!(value.as_str(), "A" | "B" | "C"))
                {
                    return Err("--variants must contain A, B, and/or C".to_owned());
                }
            }
            "--profile-override" => {
                let profile = value.trim().to_ascii_lowercase();
                if !matches!(profile.as_str(), "baseline" | "tuned") {
                    return Err("--profile-override must be baseline or tuned".to_owned());
                }
                profile_override = Some(profile);
            }
            other => return Err(format!("unknown option: {other}\n{}", usage())),
        }
        index += 2;
    }
    if inputs.is_empty() {
        return Err("at least one --input is required".to_owned());
    }
    for input in &inputs {
        if !Path::new(input).is_file() {
            return Err(format!("input does not exist: {input}"));
        }
    }
    inputs = inputs
        .into_iter()
        .map(|input| {
            fs::canonicalize(&input)
                .map(|path| path.to_string_lossy().into_owned())
                .map_err(|error| format!("cannot canonicalize input {input}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_root = output_root.ok_or_else(|| "--output-root is required".to_owned())?;
    let colmap = colmap.ok_or_else(|| "--colmap is required".to_owned())?;
    if !Path::new(&colmap).is_file() {
        return Err(format!("COLMAP executable does not exist: {colmap}"));
    }
    let colmap = fs::canonicalize(&colmap)
        .map_err(|error| format!("cannot canonicalize COLMAP executable {colmap}: {error}"))?
        .to_string_lossy()
        .into_owned();
    Ok(AbcArgs {
        inputs,
        output_root,
        colmap,
        gpu_index,
        variants,
        profile_override,
    })
}

fn run_abc(app: &AppHandle, args: AbcArgs) -> Result<(), String> {
    ensure_empty_output_root(&args.output_root)?;
    fs::create_dir_all(&args.output_root).map_err(|error| error.to_string())?;
    let input_provenance = args
        .inputs
        .iter()
        .map(|input| input_provenance(Path::new(input)))
        .collect::<Result<Vec<_>, _>>()?;
    let provenance = serde_json::to_vec_pretty(&json!({
        "schemaVersion": 1,
        "inputs": &input_provenance,
        "colmap": &args.colmap,
        "gpuIndex": &args.gpu_index,
        "variants": &args.variants,
        "profileOverride": &args.profile_override,
    }))
    .map_err(|error| error.to_string())?;
    fs::write(args.output_root.join("run-provenance.json"), provenance)
        .map_err(|error| error.to_string())?;
    let event_file = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(args.output_root.join("cli-events.jsonl"))
            .map_err(|error| error.to_string())?,
    ));
    let progress_file = event_file.clone();
    let progress_listener = app.listen("pipeline-progress", move |event| {
        if let Ok(mut file) = progress_file.lock() {
            let _ = writeln!(
                file,
                "{{\"event\":\"progress\",\"payload\":{}}}",
                event.payload()
            );
            let _ = file.flush();
        }
    });
    let log_file = event_file.clone();
    let log_listener = app.listen("pipeline-log", move |event| {
        if let Ok(mut file) = log_file.lock() {
            let _ = writeln!(
                file,
                "{{\"event\":\"log\",\"payload\":{}}}",
                event.payload()
            );
            let _ = file.flush();
        }
    });
    let manager = JobManager::default();
    let variants = [
        ("A", "baseline", settings_a(&args.gpu_index)),
        ("B", "baseline", settings_b(&args.gpu_index)),
        ("C", "baseline", settings_c(&args.gpu_index)),
    ];
    let mut results = Vec::new();
    for (name, default_profile, mut settings) in variants {
        if !args.variants.iter().any(|variant| variant == name) {
            continue;
        }
        let profile = args
            .profile_override
            .as_deref()
            .unwrap_or(default_profile);
        merge(
            &mut settings,
            json!({ "align": { "colmapQualityProfile": profile } }),
        );
        let project_path = args.output_root.join(name);
        println!("=== Variant {name} ({profile}) ===");
        let manifest = project::create(CreateProjectRequest {
            input_paths: args.inputs.clone(),
            output_path: Some(project_path.to_string_lossy().into_owned()),
            name: Some(format!("ABC-{name}")),
            settings: Some(settings.clone()),
        })?;
        verify_project_inputs(&manifest, &args.inputs)?;
        let started = Instant::now();
        run_stage(
            app,
            &manager,
            &manifest.root_path,
            StageName::Extract,
            &settings,
            &args.colmap,
        )?;
        run_stage(
            app,
            &manager,
            &manifest.root_path,
            StageName::Align,
            &settings,
            &args.colmap,
        )?;
        let benchmark_paths = find_benchmarks(&project_path);
        let completed_manifest = project::load(&manifest.root_path)?;
        let metrics = benchmark_paths
            .first()
            .and_then(|path| load_benchmark_metrics(Path::new(path), &completed_manifest));
        results.push(VariantResult {
            variant: name.to_owned(),
            project_path: project_path.to_string_lossy().into_owned(),
            quality_profile: profile.to_owned(),
            elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            benchmark_paths,
            metrics,
        });
    }
    write_comparison_markdown(&args.output_root, &args.inputs, &results)?;
    let summary_path = args.output_root.join("abc-summary.json");
    let bytes = serde_json::to_vec_pretty(&json!({
        "schemaVersion": 2,
        "inputs": args.inputs,
        "inputProvenance": input_provenance,
        "colmap": args.colmap,
        "gpuIndex": args.gpu_index,
        "profileOverride": args.profile_override,
        "results": results,
    }))
    .map_err(|error| error.to_string())?;
    fs::write(&summary_path, bytes).map_err(|error| error.to_string())?;
    println!("ABC summary: {}", summary_path.display());
    app.unlisten(progress_listener);
    app.unlisten(log_listener);
    Ok(())
}

fn input_provenance(path: &Path) -> Result<Value, String> {
    let metadata = fs::metadata(path).map_err(|error| error.to_string())?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().to_string());
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(json!({
        "path": path.to_string_lossy(),
        "size": metadata.len(),
        "modifiedNanos": modified_nanos,
        "sha256": format!("{:x}", hasher.finalize()),
    }))
}

fn ensure_empty_output_root(output_root: &Path) -> Result<(), String> {
    if !output_root.exists() {
        return Ok(());
    }
    if !output_root.is_dir() {
        return Err(format!(
            "--output-root must be a new or empty directory: {}",
            output_root.display()
        ));
    }
    if fs::read_dir(output_root)
        .map_err(|error| error.to_string())?
        .next()
        .is_some()
    {
        return Err(format!(
            "--output-root is not empty; use a fresh directory so old projects and benchmarks cannot be attributed to new inputs: {}",
            output_root.display()
        ));
    }
    Ok(())
}

fn verify_project_inputs(
    manifest: &project::ProjectManifest,
    expected_inputs: &[String],
) -> Result<(), String> {
    let actual = manifest
        .input_paths
        .iter()
        .map(fs::canonicalize)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot verify project input provenance: {error}"))?;
    let expected = expected_inputs.iter().map(PathBuf::from).collect::<Vec<_>>();
    if actual != expected {
        return Err(format!(
            "project input provenance mismatch: expected {expected_inputs:?}, found {:?}",
            manifest.input_paths
        ));
    }
    Ok(())
}

fn run_stage(
    app: &AppHandle,
    manager: &JobManager,
    project_path: &str,
    stage: StageName,
    settings: &Value,
    colmap: &str,
) -> Result<(), String> {
    let stage_name = stage.as_str().to_owned();
    let response = pipeline::start_stage(
        app.clone(),
        manager,
        StartStageRequest {
            project_path: project_path.to_owned(),
            stage: stage.clone(),
            mode: Some("retry".to_owned()),
            settings: Some(settings.clone()),
            colmap_path: Some(colmap.to_owned()),
        },
    )?;
    println!("started {stage_name}: {}", response.job_id);
    while manager.is_running() {
        thread::sleep(Duration::from_millis(500));
    }
    let manifest = project::load(project_path)?;
    let checkpoint = manifest.stage(&stage);
    match checkpoint.status {
        StageStatus::Completed => {
            println!("completed {stage_name}: {}", checkpoint.message);
            Ok(())
        }
        StageStatus::Failed | StageStatus::Cancelled => Err(format!(
            "{stage_name} did not complete: {}",
            checkpoint.message
        )),
        StageStatus::Pending | StageStatus::Running => Err(format!(
            "{stage_name} stopped without a terminal manifest status"
        )),
    }
}

fn find_benchmarks(project: &Path) -> Vec<String> {
    let metadata = project.join("metadata");
    let Ok(entries) = fs::read_dir(metadata) else {
        return Vec::new();
    };
    let mut paths = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("benchmark_") && name.ends_with(".json"))
        })
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn load_benchmark_metrics(
    benchmark_path: &Path,
    manifest: &project::ProjectManifest,
) -> Option<BenchmarkMetrics> {
    let benchmark: Value = serde_json::from_slice(&fs::read(benchmark_path).ok()?).ok()?;
    let selected = benchmark.pointer("/capture/selectedRigFrameCount")?.as_u64()?;
    let any_registered = benchmark.pointer("/colmap/registeredRigFrameCount")?.as_u64()?;
    let complete_registered = benchmark
        .pointer("/colmap/completeRegisteredRigFrameCount")?
        .as_u64()?;
    let run_metadata = load_effective_run_metadata(benchmark_path.parent()?.parent()?, manifest);
    Some(BenchmarkMetrics {
        selected_rig_frames: selected,
        any_registered_rig_frames: any_registered,
        complete_registered_rig_frames: complete_registered,
        complete_registered_percent: if selected == 0 {
            0.0
        } else {
            100.0 * complete_registered as f64 / selected as f64
        },
        pair_count: benchmark.pointer("/capture/pairCount")?.as_u64()?,
        points_3d: benchmark.pointer("/colmap/points3dCount")?.as_u64()?,
        median_track_length: benchmark.pointer("/colmap/medianTrackLength")?.as_f64()?,
        median_reprojection_error_px: benchmark
            .pointer("/colmap/medianReprojectionErrorPx")?
            .as_f64()?,
        connected_components: benchmark
            .pointer("/colmap/connectedComponentCount")?
            .as_u64()?,
        largest_component_images: benchmark
            .pointer("/colmap/largestConnectedComponentImageCount")?
            .as_u64()?,
        extract_ms: manifest.stage(&StageName::Extract).duration_ms.unwrap_or(0),
        align_ms: manifest.stage(&StageName::Align).duration_ms.unwrap_or(0),
        effective_mapper: run_metadata.effective_mapper,
        gravity_prior_applied: run_metadata.gravity_prior_applied,
        gravity_coverage_ratio: run_metadata.gravity_coverage_ratio,
    })
}

struct EffectiveRunMetadata {
    effective_mapper: String,
    gravity_prior_applied: bool,
    gravity_coverage_ratio: Option<f64>,
}

fn load_effective_run_metadata(
    project_root: &Path,
    manifest: &project::ProjectManifest,
) -> EffectiveRunMetadata {
    let timing = fs::read(project_root.join("metadata/align_timings.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    let effective_mapper = timing
        .as_ref()
        .and_then(|value| value.pointer("/effectiveMapper"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let priors = fs::read(project_root.join("metadata/global_mapper_priors.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    let gravity_prior_valid = priors
        .as_ref()
        .and_then(|value| value.pointer("/gravityPriorValid"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let database_priors_injected = priors
        .as_ref()
        .and_then(|value| value.pointer("/databasePosePriorsInjected"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let gravity_requested = manifest
        .settings
        .pointer("/align/useGravityPrior")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    EffectiveRunMetadata {
        gravity_prior_applied: effective_mapper == "global_mapper"
            && gravity_requested
            && gravity_prior_valid
            && database_priors_injected,
        gravity_coverage_ratio: priors
            .as_ref()
            .and_then(|value| value.pointer("/gravityCoverageRatio"))
            .and_then(Value::as_f64),
        effective_mapper,
    }
}

fn write_comparison_markdown(
    output_root: &Path,
    inputs: &[String],
    results: &[VariantResult],
) -> Result<(), String> {
    let mut report = String::from(
        "# COLMAP A/B/C comparison\n\n\
         Registration coverage and track support take priority over a low reprojection error from a tiny surviving model.\n\n",
    );
    report.push_str("## Inputs\n\n");
    for input in inputs {
        report.push_str(&format!("- `{input}`\n"));
    }
    report.push_str(
        "\n## Metrics\n\n\
         | Variant | Profile | Complete rigs | Any-sensor rigs | Pairs | 3D points | Median track | Median reprojection | Components | Largest component | Extract | Align | Effective mapper | Gravity applied |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n",
    );
    for result in results {
        let Some(metrics) = &result.metrics else {
            continue;
        };
        report.push_str(&format!(
            "| {} | {} | {}/{} ({:.1}%) | {} | {} | {} | {:.1} | {:.3} px | {} | {} images | {:.1} s | {:.1} s | {} | {} |\n",
            result.variant,
            result.quality_profile,
            metrics.complete_registered_rig_frames,
            metrics.selected_rig_frames,
            metrics.complete_registered_percent,
            metrics.any_registered_rig_frames,
            metrics.pair_count,
            metrics.points_3d,
            metrics.median_track_length,
            metrics.median_reprojection_error_px,
            metrics.connected_components,
            metrics.largest_component_images,
            metrics.extract_ms as f64 / 1000.0,
            metrics.align_ms as f64 / 1000.0,
            metrics.effective_mapper,
            if metrics.gravity_prior_applied { "yes" } else { "no" },
        ));
    }
    report.push_str(
        "\nThe table is an automatic sparse-reconstruction comparison. Visual 3DGS quality still requires identical training and render settings.\n",
    );
    fs::write(output_root.join("abc-comparison.md"), report).map_err(|error| error.to_string())
}

fn common_settings(gpu_index: &str, profile: &str) -> Value {
    json!({
        "extract": {
            "baseFps": 3,
            "denseFps": 12,
            "skipBlurry": true
        },
        "mask": {
            "classes": [],
            "maskSky": false
        },
        "align": {
            "useGpu": true,
            "gpuIndex": gpu_index,
            "colmapQualityProfile": profile,
            "fixedRotationBa": false,
            "orientationPriorExecutable": ""
        }
    })
}

fn settings_a(gpu_index: &str) -> Value {
    let mut settings = common_settings(gpu_index, "baseline");
    merge(&mut settings, json!({
        "extract": { "keyframePruning": false },
        "align": {
            "mapperMode": "incremental",
            "useGravityPrior": false,
            "autoCalibrateTelemetry": false,
            "calibrateFocalPrior": false,
            "useVisualRetrieval": false,
            "useCalibratedFovPairs": false,
            "exportRollingShutterTrajectory": false
        }
    }));
    settings
}

fn settings_b(gpu_index: &str) -> Value {
    let mut settings = common_settings(gpu_index, "baseline");
    merge(&mut settings, json!({
        "extract": {
            "keyframePruning": true,
            "minRotationDeg": 5,
            "minGapMs": 200,
            "maxGapMs": 600,
            "minVisualNovelty": 0.08
        },
        "align": {
            "mapperMode": "incremental",
            "useGravityPrior": false,
            "autoCalibrateTelemetry": false,
            "calibrateFocalPrior": false,
            "useVisualRetrieval": true,
            "useCalibratedFovPairs": false,
            "exportRollingShutterTrajectory": false
        }
    }));
    settings
}

fn settings_c(gpu_index: &str) -> Value {
    let mut settings = settings_b(gpu_index);
    merge(&mut settings, json!({
        "align": {
            "mapperMode": "auto",
            "useGravityPrior": true,
            "autoCalibrateTelemetry": true,
            "calibrateFocalPrior": true,
            "useCalibratedFovPairs": true,
            "exportRollingShutterTrajectory": true
        }
    }));
    settings
}

fn merge(target: &mut Value, incoming: Value) {
    match (target, incoming) {
        (Value::Object(target), Value::Object(incoming)) => {
            for (key, value) in incoming {
                merge(target.entry(key).or_insert(Value::Null), value);
            }
        }
        (target, incoming) => *target = incoming,
    }
}

#[cfg(test)]
mod tests {
    use super::{ensure_empty_output_root, input_provenance, load_benchmark_metrics};
    use crate::project::{self, CreateProjectRequest};
    use serde_json::json;
    use std::fs;

    #[test]
    fn abc_output_root_must_be_new_or_empty() {
        let temp = tempfile::tempdir().unwrap();
        let empty = temp.path().join("empty");
        fs::create_dir(&empty).unwrap();
        assert!(ensure_empty_output_root(&empty).is_ok());
        fs::write(empty.join("stale.json"), b"{}").unwrap();
        let error = ensure_empty_output_root(&empty).unwrap_err();
        assert!(error.contains("not empty"));
        assert!(ensure_empty_output_root(&temp.path().join("new")).is_ok());
    }

    #[test]
    fn input_provenance_hashes_content_not_only_the_path() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("capture.osv");
        fs::write(&input, b"abc").unwrap();
        let provenance = input_provenance(&input).unwrap();
        assert_eq!(provenance["size"], 3);
        assert_eq!(
            provenance["sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn benchmark_metrics_use_complete_rigs_and_effective_prior_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("capture.osv");
        fs::write(&input, b"capture").unwrap();
        let root = temp.path().join("project");
        let manifest = project::create(CreateProjectRequest {
            input_paths: vec![input.to_string_lossy().into_owned()],
            output_path: Some(root.to_string_lossy().into_owned()),
            name: Some("metrics".to_owned()),
            settings: Some(json!({"align": {"useGravityPrior": true}})),
        })
        .unwrap();
        let metadata = root.join("metadata");
        let benchmark_path = metadata.join("benchmark.json");
        fs::write(
            &benchmark_path,
            serde_json::to_vec(&json!({
                "capture": {"selectedRigFrameCount": 10, "pairCount": 12},
                "colmap": {
                    "registeredRigFrameCount": 9,
                    "completeRegisteredRigFrameCount": 7,
                    "points3dCount": 100,
                    "medianTrackLength": 3.0,
                    "medianReprojectionErrorPx": 0.5,
                    "connectedComponentCount": 1,
                    "largestConnectedComponentImageCount": 14
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            metadata.join("align_timings.json"),
            br#"{"effectiveMapper":"global_mapper"}"#,
        )
        .unwrap();
        fs::write(
            metadata.join("global_mapper_priors.json"),
            br#"{"gravityPriorValid":true,"gravityCoverageRatio":0.9,"databasePosePriorsInjected":true}"#,
        )
        .unwrap();
        let metrics = load_benchmark_metrics(&benchmark_path, &manifest).unwrap();
        assert_eq!(metrics.any_registered_rig_frames, 9);
        assert_eq!(metrics.complete_registered_rig_frames, 7);
        assert_eq!(metrics.effective_mapper, "global_mapper");
        assert!(metrics.gravity_prior_applied);
        assert_eq!(metrics.gravity_coverage_ratio, Some(0.9));
    }
}
