//! Native headless geometry runner for existing COLMAP datasets, no Python.
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    spherealign_lib::geometry::cli::run(std::env::args().skip(1).collect())
}
