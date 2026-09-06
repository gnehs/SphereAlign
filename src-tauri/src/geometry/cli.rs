use super::{
    draft::{self, Settings},
    CancelToken,
};
pub fn run(args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let command=args.next().ok_or("Usage: spherealign-cli geometry <preflight|run|inspect> DATASET [--model ONNX] [--frames N; 0=all]")?;
    let root = args
        .next()
        .ok_or("Missing dataset root or --project PATH")?;
    let root = if root == "--project" {
        args.next().ok_or("Missing project path")?
    } else {
        root
    };
    let mut root = std::path::PathBuf::from(root);
    if root.join("project.json").exists() {
        root = std::path::PathBuf::from(crate::project::load(&root)?.output_path);
    }
    let mut settings = Settings::default();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => settings.model_path = args.next().ok_or("Missing model path")?,
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
        "run" => {
            draft::run(&root, settings, &CancelToken::new(), None, |r| {
                println!(
                    "{}",
                    serde_json::json!({"id":r.id,"status":r.status,"completed":r.completed,"total":r.total,"message":r.message})
                )
            })?;
        }
        _ => return Err("Expected preflight, run or inspect".into()),
    }
    Ok(())
}
