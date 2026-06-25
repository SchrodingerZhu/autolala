//! `dmd-mcp`: an advisor-only Model Context Protocol server for affine-program
//! locality analysis.
//!
//! It speaks MCP over stdio via [`tower_mcp::StdioTransport`] and exposes a
//! read-only tool set built on `dmd-core` and `mlir-extract`: extract DSL from
//! tagged MLIR, validate DSL, run DMD analysis, and compare transformation
//! variants. No tool writes files or mutates state.
//!
//! Tool input schemas are derived from typed structs in [`tools`] via
//! `schemars`, so nothing about the wire format is hand-maintained here.

mod summary;
mod tools;

use tower_mcp::{BoxError, CallToolResult, McpRouter, StdioTransport, ToolBuilder};

use tools::{AnalyzeDslInput, AnalyzeMlirInput, CompareInput, ExtractInput, ValidateInput};

const SERVER_NAME: &str = "dmd-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

const INSTRUCTIONS: &str = "Advisor tools for affine-program locality analysis. Use analyze_mlir / analyze_dsl to get DMD formulas and reuse-distance distributions, and compare_variants to weigh loop transformations. All tools are read-only.";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Logs must stay off stdout so the JSON-RPC stream is never corrupted.
    eprintln!("{SERVER_NAME} {SERVER_VERSION} starting on stdio");

    let extract = ToolBuilder::new("extract_from_mlir")
        .description(
            "Translate the affine.for loop tagged with a marker attribute (default `dmd.extract`) in an MLIR module into AutoLALA DSL source. Returns the DSL, or a diagnostic pointing at the unsupported operation.",
        )
        .read_only()
        .idempotent()
        .handler(|input: ExtractInput| async move {
            blocking(move || tools::extract_from_mlir(input)).await
        })
        .build();

    let validate = ToolBuilder::new("validate_dsl")
        .description(
            "Parse and semantically validate AutoLALA DSL source without running analysis. Reports whether it is well-formed and, if not, why.",
        )
        .read_only()
        .idempotent()
        .handler(|input: ValidateInput| async move {
            blocking(move || tools::validate_dsl(input)).await
        })
        .build();

    let analyze_dsl = ToolBuilder::new("analyze_dsl")
        .description(
            "Run symbolic locality / data-movement (DMD) analysis on AutoLALA DSL source. Returns reuse-interval and reuse-distance distributions, the DMD formula, access counts, and a digest. Analysis is symbolic and may be slow for large nests.",
        )
        .read_only()
        .idempotent()
        .handler(|input: AnalyzeDslInput| async move {
            blocking(move || tools::analyze_dsl(input)).await
        })
        .build();

    let analyze_mlir = ToolBuilder::new("analyze_mlir")
        .description(
            "Extract the tagged affine loop from an MLIR module and run DMD analysis on it in one step. Returns the intermediate DSL plus the analysis report and digest.",
        )
        .read_only()
        .idempotent()
        .handler(|input: AnalyzeMlirInput| async move {
            blocking(move || tools::analyze_mlir(input)).await
        })
        .build();

    let compare = ToolBuilder::new("compare_variants")
        .description(
            "Analyze several DSL variants of a kernel and present their DMD formulas and access counts side by side, so an advisor can judge which transformation improves locality. Ranking is qualitative because the metrics are symbolic.",
        )
        .read_only()
        .idempotent()
        .handler(|input: CompareInput| async move {
            blocking(move || tools::compare_variants(input)).await
        })
        .build();

    let router = McpRouter::new()
        .server_info(SERVER_NAME, SERVER_VERSION)
        .instructions(INSTRUCTIONS)
        .tool(extract)
        .tool(validate)
        .tool(analyze_dsl)
        .tool(analyze_mlir)
        .tool(compare);

    StdioTransport::new(router).run().await?;
    Ok(())
}

/// Run a synchronous (and potentially slow or FFI-heavy) tool body off the async
/// reactor. `spawn_blocking` also contains panics: a panicking analysis surfaces
/// as a tool error instead of taking down the server.
async fn blocking<F>(f: F) -> tower_mcp::Result<CallToolResult>
where
    F: FnOnce() -> CallToolResult + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => Ok(result),
        Err(_) => Ok(CallToolResult::error(
            "internal error: the analysis panicked",
        )),
    }
}
