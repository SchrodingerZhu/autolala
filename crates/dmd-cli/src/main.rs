use clap::Parser;
use dmd_core::{AnalysisOptions, AnalysisReport, DmdError, analyze_source};
use std::fmt::Write;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "dmd-cli", about = "Symbolic data movement analysis for loop-tree DSL programs")]
struct Cli {
    #[arg(short, long)]
    input: Option<PathBuf>,

    #[arg(long, default_value_t = 1)]
    block_size: usize,

    #[arg(long, default_value_t = 1)]
    num_sets: usize,

    #[arg(long, default_value_t = 5_000_000)]
    max_operations: usize,

    #[arg(long)]
    json: bool,
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Analysis(#[from] DmdError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .without_time()
        .init();

    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    let cli = Cli::parse();
    let source = read_source(cli.input.as_deref())?;
    let report = analyze_source(
        &source,
        AnalysisOptions {
            block_size: cli.block_size,
            num_sets: cli.num_sets,
            max_operations: cli.max_operations,
        },
    )?;

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_report(&report));
    }

    Ok(())
}

fn read_source(path: Option<&Path>) -> Result<String, std::io::Error> {
    match path {
        Some(path) => std::fs::read_to_string(path),
        None => {
            let mut buffer = String::new();
            std::io::stdin().read_to_string(&mut buffer)?;
            Ok(buffer)
        }
    }
}

fn render_report(report: &AnalysisReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "DMD");
    let _ = writeln!(output, "  {}", report.dmd_formula_plain);
    let _ = writeln!(output);
    let _ = writeln!(output, "Access Counts");
    let _ = writeln!(output, "  total      = {}", report.total_accesses_plain);
    let _ = writeln!(output, "  warm       = {}", report.warm_accesses_plain);
    let _ = writeln!(output, "  compulsory = {}", report.compulsory_accesses_plain);
    let _ = writeln!(output);
    let _ = writeln!(output, "RI Distribution");
    render_distribution_section(&mut output, &report.ri_distribution);
    let _ = writeln!(output);
    let _ = writeln!(output, "RD Distribution");
    render_distribution_section(&mut output, &report.rd_distribution);
    let _ = writeln!(output);
    let _ = writeln!(output, "Notes");
    for note in &report.notes {
        let _ = writeln!(output, "  - {note}");
    }
    output
}

fn render_distribution_section(output: &mut String, entries: &[dmd_core::DistributionEntry]) {
    if entries.is_empty() {
        let _ = writeln!(output, "  <empty>");
        return;
    }

    for entry in entries {
        let _ = writeln!(output, "  value = {}", entry.value_plain);
        for region in &entry.regions {
            let _ = writeln!(
                output,
                "    [{}] count = {}",
                region.domain_plain, region.count_plain
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::render_report;
    use dmd_core::{AnalysisOptions, analyze_source};

    #[test]
    fn renders_text_report() {
        let report = analyze_source(
            r#"
params N;
array A[N];

for i in 0 .. N {
    read A[0];
}
"#,
            AnalysisOptions::default(),
        )
        .expect("analysis should succeed");

        let rendered = render_report(&report);
        assert!(rendered.contains("DMD"));
        assert!(rendered.contains("RI Distribution"));
        assert!(rendered.contains("RD Distribution"));
    }
}
