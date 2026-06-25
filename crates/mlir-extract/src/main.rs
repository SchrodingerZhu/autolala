//! Command-line frontend for `mlir-extract`.
//!
//! Reads an `.mlir` file, translates the `affine.for` loop tagged with a marker
//! attribute (default `slap.extract`) into AutoLALA DSL source, and prints it.
//! Unsupported constructs are reported as `ariadne` diagnostics pointing at the
//! offending operation.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use mlir_extract::{ExtractError, extract_dsl_from_source, report_error};

#[derive(Parser, Debug)]
#[command(
    name = "mlir-extract",
    about = "Translate an attribute-tagged MLIR affine loop into the AutoLALA DSL"
)]
struct Args {
    /// Input MLIR file.
    input: PathBuf,

    /// Attribute that marks the loop to extract.
    #[arg(short, long, default_value = "dmd.extract")]
    attr: String,

    /// Write the DSL to this file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Skip the round-trip validation of the generated DSL.
    #[arg(long)]
    no_validate: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let source = match std::fs::read_to_string(&args.input) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("error: cannot read {}: {error}", args.input.display());
            return ExitCode::FAILURE;
        }
    };

    let filename = args.input.display().to_string();

    let dsl = match extract_dsl_from_source(&source, &args.attr) {
        Ok(dsl) => dsl,
        Err(error) => {
            report_error(&filename, &source, &error);
            return ExitCode::FAILURE;
        }
    };

    if !args.no_validate
        && let Err(error) = validate(&dsl)
    {
        report_error(&filename, &source, &error);
        return ExitCode::FAILURE;
    }

    match &args.output {
        Some(path) => {
            if let Err(error) = std::fs::write(path, &dsl) {
                eprintln!("error: cannot write {}: {error}", path.display());
                return ExitCode::FAILURE;
            }
        }
        None => print!("{dsl}"),
    }

    ExitCode::SUCCESS
}

/// Self-check: the emitted DSL must parse and pass `dmd-core`'s semantic
/// validation. A failure here is an internal translator bug, not user error.
fn validate(dsl: &str) -> Result<(), ExtractError> {
    let program = dmd_core::parse_program(dsl).map_err(|error| {
        ExtractError::new(format!(
            "internal error: generated DSL failed to parse ({error})"
        ))
    })?;
    dmd_core::validate_program(program).map_err(|error| {
        ExtractError::new(format!(
            "internal error: generated DSL failed validation ({error})"
        ))
    })?;
    Ok(())
}
