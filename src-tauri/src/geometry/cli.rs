use super::{
    draft::{self, Settings},
    CancelToken,
};
pub fn run(args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let command=args.next().ok_or("Usage: spherealign-cli geometry <preflight|run|normals|inspect|cleanup> DATASET [--model ONNX] [--frames N; 0=all] [--keep-intermediates] [--run RUN_ID]")?;
    let root = args
        .next()
        .ok_or("Missing dataset root or --project PATH")?;
    let root = if root == "--project" {
        args.next().ok_or("Missing project path")?
    } else {
        root
    };
    let mut root = std::path::PathBuf::from(root);
    let mut settings = Settings::default();
    if root.join("project.json").exists() {
        let project = crate::project::snapshot(&root)?;
        if command == "normals" {
            settings = serde_json::from_value(
                project
                    .settings
                    .get("normals")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
            )
            .map_err(|e| e.to_string())?;
        }
        root = std::path::PathBuf::from(project.output_path);
    }
    let mut run_id = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => settings.model_path = args.next().ok_or("Missing model path")?,
            "--keep-intermediates" => settings.keep_intermediates = true,
            "--run" => run_id = Some(args.next().ok_or("Missing run ID")?),
            "--frames" => {
                settings.frame_limit = args
                    .next()
                    .ok_or("Missing frame count")?
                    .parse()
                    .map_err(|_| "Invalid frame count")?
            }
            _ => return Err(format!("Unknown option {flag}")),
        }
    }
    match command.as_str() {
        "preflight" => println!(
            "{}",
            serde_json::to_string_pretty(&draft::preflight(&root, &settings)?)
                .map_err(|e| e.to_string())?
        ),
        "inspect" => println!(
            "{}",
            serde_json::to_string_pretty(&draft::list(&root)?).map_err(|e| e.to_string())?
        ),
        "run" | "normals" => {
            let runner = if command == "normals" {
                draft::run_normals
            } else {
                draft::run
            };
            runner(
                &root,
                settings,
                &CancelToken::new(),
                None,
                |r: &draft::Report| {
                    println!(
                        "{}",
                        serde_json::json!({"id":r.id,"status":r.status,"completed":r.completed,"total":r.total,"message":r.message})
                    )
                },
            )?;
        }
        "cleanup" => {
            let run_id = run_id.ok_or("cleanup requires --run RUN_ID")?;
            draft::clean_completed(&root, &run_id, &CancelToken::new(), |r| {
                println!(
                    "{}",
                    serde_json::json!({"id":r.id,"message":r.message,
                    "removedIntermediateBytes":r.frames.iter().map(|f| f.removed_intermediate_bytes).sum::<u64>()})
                );
            })?;
        }
        _ => return Err("Expected preflight, run, normals, inspect or cleanup".into()),
    }
    Ok(())
}
