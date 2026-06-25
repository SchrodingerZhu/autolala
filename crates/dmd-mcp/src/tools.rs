//! The advisor tool set: extraction, validation, analysis, and comparison.
//!
//! Every tool is read-only — it consumes program text and returns analysis. No
//! tool writes files or mutates state, matching the advisor-only design.
//!
//! Tool input types derive [`schemars::JsonSchema`], so the MCP `inputSchema`
//! advertised to clients is generated from these structs rather than hand-written.

use dmd_core::{
    AnalysisOptions, ApproximationMethod, analyze_source, parse_program, validate_program,
};
use mlir_extract::{ExtractError, extract_dsl_from_source};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tower_mcp::CallToolResult;

use crate::summary::summarize;

const DEFAULT_ATTR: &str = "dmd.extract";

/// Cache / Barvinok knobs shared by every analysis tool. Flattened into each
/// tool's input so callers pass them as top-level fields.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct AnalysisOpts {
    /// Cache block size (elements per line).
    pub block_size: Option<usize>,
    /// Number of cache sets.
    pub num_sets: Option<usize>,
    /// Barvinok operation budget.
    pub max_operations: Option<usize>,
}

impl AnalysisOpts {
    fn to_options(&self) -> AnalysisOptions {
        let mut options = AnalysisOptions::default();
        if let Some(block_size) = self.block_size {
            options.block_size = block_size;
        }
        if let Some(num_sets) = self.num_sets {
            options.num_sets = num_sets;
        }
        if let Some(max_operations) = self.max_operations {
            options.max_operations = max_operations;
        }
        // `scale` is currently the only supported method.
        options.approximation_method = ApproximationMethod::Scale;
        options
    }
}

/// Input for `extract_from_mlir`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractInput {
    /// MLIR module text containing a tagged `affine.for` loop.
    pub mlir: String,
    /// Marker attribute on the loop to extract (default `dmd.extract`).
    #[serde(default)]
    pub attr: Option<String>,
}

/// Input for `validate_dsl`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ValidateInput {
    /// AutoLALA DSL program text.
    pub dsl: String,
}

/// Input for `analyze_dsl`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AnalyzeDslInput {
    /// AutoLALA DSL program text.
    pub dsl: String,
    #[serde(flatten)]
    pub options: AnalysisOpts,
}

/// Input for `analyze_mlir`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AnalyzeMlirInput {
    /// MLIR module text containing a tagged `affine.for` loop.
    pub mlir: String,
    /// Marker attribute on the loop to extract (default `dmd.extract`).
    #[serde(default)]
    pub attr: Option<String>,
    #[serde(flatten)]
    pub options: AnalysisOpts,
}

/// A single labelled DSL variant for `compare_variants`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Variant {
    /// Human-readable name for the variant.
    pub label: String,
    /// AutoLALA DSL program text.
    pub dsl: String,
}

/// Input for `compare_variants`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompareInput {
    /// The DSL variants to compare.
    pub variants: Vec<Variant>,
    #[serde(flatten)]
    pub options: AnalysisOpts,
}

/// Translate the tagged `affine.for` loop in an MLIR module into AutoLALA DSL.
pub fn extract_from_mlir(input: ExtractInput) -> CallToolResult {
    let attr = input.attr.as_deref().unwrap_or(DEFAULT_ATTR);
    match extract_dsl_from_source(&input.mlir, attr) {
        Ok(dsl) => json_result(json!({ "dsl": dsl })),
        Err(error) => CallToolResult::error(format_extract_error(&error)),
    }
}

/// Parse and semantically validate DSL without running analysis.
pub fn validate_dsl(input: ValidateInput) -> CallToolResult {
    match parse_program(&input.dsl).and_then(validate_program) {
        Ok(model) => json_result(json!({
            "valid": true,
            "params": model.params,
            "arrays": model.arrays.iter().map(|a| json!({ "name": a.name, "rank": a.rank })).collect::<Vec<_>>(),
            "parallel": model.parallel_loop.as_ref().map(|p| json!({ "var": p.var, "threads": p.threads })),
        })),
        Err(error) => json_result(json!({ "valid": false, "error": error.to_string() })),
    }
}

/// Run DMD analysis on DSL source.
pub fn analyze_dsl(input: AnalyzeDslInput) -> CallToolResult {
    run_analysis(&input.dsl, input.options.to_options(), None)
}

/// Extract the tagged loop from MLIR, then run DMD analysis on it.
pub fn analyze_mlir(input: AnalyzeMlirInput) -> CallToolResult {
    let attr = input.attr.as_deref().unwrap_or(DEFAULT_ATTR);
    let dsl = match extract_dsl_from_source(&input.mlir, attr) {
        Ok(dsl) => dsl,
        Err(error) => return CallToolResult::error(format_extract_error(&error)),
    };
    run_analysis(&dsl, input.options.to_options(), Some(dsl.clone()))
}

/// Analyze several DSL variants and present their metrics side by side.
pub fn compare_variants(input: CompareInput) -> CallToolResult {
    let options = input.options.to_options();

    let rows = input
        .variants
        .iter()
        .map(
            |variant| match analyze_source(&variant.dsl, options.clone()) {
                Ok(report) => json!({
                    "label": variant.label,
                    "ok": true,
                    "dmd_formula": report.dmd_formula_plain,
                    "total_accesses": report.total_accesses_plain,
                    "warm_accesses": report.warm_accesses_plain,
                    "compulsory_accesses": report.compulsory_accesses_plain,
                    "parallel": report.parallel.is_some(),
                }),
                Err(error) => {
                    json!({ "label": variant.label, "ok": false, "error": error.to_string() })
                }
            },
        )
        .collect::<Vec<_>>();

    json_result(json!({
        "variants": rows,
        "note": "Metrics are symbolic; compare the DMD formulas and access counts qualitatively. Lower data movement and more warm (reused) accesses indicate better locality.",
    }))
}

/// Shared analysis path for `analyze_dsl` / `analyze_mlir`.
fn run_analysis(dsl: &str, options: AnalysisOptions, dsl_echo: Option<String>) -> CallToolResult {
    match analyze_source(dsl, options) {
        Ok(report) => {
            let summary = summarize(&report);
            let mut value = json!({ "report": report, "summary": summary });
            if let Some(dsl) = dsl_echo
                && let Some(object) = value.as_object_mut()
            {
                object.insert("dsl".to_string(), Value::String(dsl));
            }
            json_result(value)
        }
        Err(error) => CallToolResult::error(format!("analysis failed: {error}")),
    }
}

fn format_extract_error(error: &ExtractError) -> String {
    let mut text = format!("extraction failed: {}", error.message);
    if let Some(span) = error.span {
        text.push_str(&format!(
            "\n  at line {}, column {}",
            span.start_line, span.start_col
        ));
    }
    if let Some(help) = &error.help {
        text.push_str(&format!("\n  help: {help}"));
    }
    text
}

/// Render a JSON value as pretty text content for a successful tool result.
fn json_result(value: Value) -> CallToolResult {
    CallToolResult::text(pretty(&value))
}

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first text content block of a result, as the client would see it.
    fn text_of(result: &CallToolResult) -> String {
        serde_json::to_value(result)
            .ok()
            .and_then(|v| {
                v.pointer("/content/0/text")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_default()
    }

    /// Whether the result is flagged as an error to the client.
    fn is_error(result: &CallToolResult) -> bool {
        serde_json::to_value(result)
            .ok()
            .and_then(|v| v.get("isError").and_then(Value::as_bool))
            .unwrap_or(false)
    }

    /// The first text block parsed as JSON (for successful, JSON-bearing results).
    fn json_of(result: &CallToolResult) -> Value {
        serde_json::from_str::<Value>(&text_of(result)).unwrap_or(Value::Null)
    }

    const DSL: &str = "params N; array A[N];\nfor i in 0 .. N { read A[i]; }";

    #[test]
    fn analyze_dsl_returns_report_and_summary() {
        let result = analyze_dsl(AnalyzeDslInput {
            dsl: DSL.to_string(),
            options: AnalysisOpts::default(),
        });
        assert!(!is_error(&result));
        let value = json_of(&result);
        assert!(value.get("report").is_some());
        assert!(
            value["summary"]
                .as_str()
                .unwrap_or_default()
                .contains("DMD")
        );
    }

    #[test]
    fn validate_dsl_reports_semantic_errors() {
        let result = validate_dsl(ValidateInput {
            dsl: "params N; array A[N]; for i in 0..N { read A[i*i]; }".to_string(),
        });
        assert!(
            !is_error(&result),
            "validation result is data, not a tool error"
        );
        assert_eq!(json_of(&result)["valid"], Value::Bool(false));
    }

    #[test]
    fn extract_from_mlir_rejects_unsupported_op_as_tool_error() {
        let mlir = r#"
module {
  func.func @bad(%A: memref<10xf32>, %N: index) {
    affine.for %i = 0 to %N {
      %v = memref.load %A[%i] : memref<10xf32>
      affine.store %v, %A[%i] : memref<10xf32>
    } { dmd.extract }
    return
  }
}
"#;
        let result = extract_from_mlir(ExtractInput {
            mlir: mlir.to_string(),
            attr: None,
        });
        assert!(is_error(&result));
        let text = text_of(&result);
        assert!(text.contains("memref.load"));
        assert!(text.contains("line"));
    }

    #[test]
    fn compare_variants_lists_each_label() {
        let result = compare_variants(CompareInput {
            variants: vec![
                Variant {
                    label: "v1".to_string(),
                    dsl: DSL.to_string(),
                },
                Variant {
                    label: "v2".to_string(),
                    dsl: DSL.to_string(),
                },
            ],
            options: AnalysisOpts::default(),
        });
        assert!(!is_error(&result));
        let value = json_of(&result);
        let variants = value["variants"].as_array().expect("array");
        assert_eq!(variants.len(), 2);
        assert!(variants.iter().all(|v| v["ok"] == Value::Bool(true)));
    }
}
