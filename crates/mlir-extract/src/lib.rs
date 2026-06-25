//! `mlir-extract`: translate an attribute-tagged MLIR `affine.for` loop nest
//! into the AutoLALA DSL consumed by `dmd-core`.
//!
//! The entry point is [`extract_dsl_from_source`]; lower-level pieces
//! ([`extract_program`], [`emit_program`], [`report_error`]) are exposed for
//! callers that want the intermediate `dmd-core` AST or custom diagnostics.

pub mod context;
mod emit;
mod error;
mod ffi;
mod report;
mod translate;

use melior::ir::Module;

pub use context::new_context;
pub use emit::emit_program;
pub use error::{ExtractError, ExtractResult};
pub use ffi::SourceSpan;
pub use report::report_error;
pub use translate::extract_program;

/// Parse `source` as an MLIR module and translate the loop tagged with `attr`
/// into DSL source text.
///
/// A failure to parse surfaces as a span-less [`ExtractError`] (MLIR prints its
/// own parser diagnostics to stderr); translation failures carry the source
/// span of the offending operation so [`report_error`] can underline it.
pub fn extract_dsl_from_source(source: &str, attr: &str) -> ExtractResult<String> {
    let context = new_context();
    let module = Module::parse(&context, source).ok_or_else(|| {
        ExtractError::new("failed to parse the input as an MLIR module")
            .with_help("check the diagnostics printed above for the exact parse error")
    })?;
    let program = extract_program(&module, attr)?;
    Ok(emit_program(&program))
}
