use tracing_subscriber::EnvFilter;

fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();

    let args: Vec<String> = std::env::args().collect();
    if let Err(error) = tofy::run(&args) {
        if let Err(report_error) = tofy::tasks::pipeline::record_stage_failure_from_env(&error) {
            eprintln!("failed to record isolated-stage failure: {report_error:#}");
        }
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
