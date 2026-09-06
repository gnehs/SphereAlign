//! Developer-only native ONNX feasibility probe; does not open a Tauri window.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 3 && args[1] == "--prepare-static" {
        match spherealign_lib::geometry::specialize::prepare(
            std::path::Path::new(&args[0]),
            std::path::Path::new(&args[2]),
        ) {
            Ok(()) => {
                println!("{}", spherealign_lib::geometry::specialize::DERIVED_HASH);
                return;
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }
    if args.len() != 2
        && args.len() != 5
        && !((args.len() == 3 || args.len() == 4) && args[2] == "--static-experiment")
    {
        eprintln!("Usage: spherealign-geometry-probe <pinned-model.onnx> <DirectML|CoreML|CUDA> [width height tokens | --static-experiment [new-dump-directory]]");
        std::process::exit(2);
    }
    let shape = if args.len() == 5 {
        let parsed: Result<Vec<usize>, _> = args[2..].iter().map(|s| s.parse()).collect();
        match parsed {
            Ok(v) => Some([v[0], v[1], v[2]]),
            Err(e) => {
                eprintln!("invalid shape: {e}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    let report = if args.len() == 3 || args.len() == 4 {
        spherealign_lib::geometry::model::probe_static_experiment(
            std::path::Path::new(&args[0]),
            &args[1],
            args.get(3).map(std::path::Path::new),
        )
    } else {
        spherealign_lib::geometry::model::probe_shape(
            std::path::Path::new(&args[0]),
            &args[1],
            shape,
        )
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if report.status == "failed" {
        std::process::exit(1);
    }
}
