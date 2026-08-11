fn main() {
    if let Err(error) = gs360studio_lib::run_cli(std::env::args().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
