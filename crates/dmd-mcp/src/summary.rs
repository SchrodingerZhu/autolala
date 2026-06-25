//! Compact, agent-friendly digests of an [`AnalysisReport`].
//!
//! The full report is also returned to the tool caller; the summary exists so an
//! agent can reason about a kernel without parsing every distribution entry.

use std::fmt::Write;

use dmd_core::AnalysisReport;

/// A short multi-line digest highlighting the headline locality metrics.
pub fn summarize(report: &AnalysisReport) -> String {
    let mut out = String::new();

    if let Some(parallel) = &report.parallel {
        let _ = writeln!(
            out,
            "Parallel CRI model ({} threads, schedule = {}).",
            parallel.thread_count, parallel.schedule
        );
        let _ = writeln!(
            out,
            "Reuse-interval regions feeding the CRI laws: {}.",
            parallel.cri_entries.len()
        );
        for entry in &parallel.cri_entries {
            let _ = writeln!(
                out,
                "  - {:?} law over [{}] from source RI {}",
                entry.model_kind, entry.domain_plain, entry.source_ri_plain
            );
        }
    } else {
        let _ = writeln!(out, "DMD (data-movement) formula:");
        let _ = writeln!(out, "  {}", report.dmd_formula_plain);
        let _ = writeln!(
            out,
            "Accesses: total = {}, warm (reused) = {}, compulsory (cold) = {}.",
            report.total_accesses_plain,
            report.warm_accesses_plain,
            report.compulsory_accesses_plain
        );
        let _ = writeln!(
            out,
            "Reuse-distance distribution has {} distinct value region(s); reuse-interval distribution has {}.",
            report.rd_distribution.len(),
            report.ri_distribution.len()
        );
        for entry in report.rd_distribution.iter().take(5) {
            let domains = entry
                .regions
                .iter()
                .map(|region| format!("{} @ [{}]", region.count_plain, region.domain_plain))
                .collect::<Vec<_>>()
                .join("; ");
            let _ = writeln!(out, "  - RD = {}: {}", entry.value_plain, domains);
        }
        if report.rd_distribution.len() > 5 {
            let _ = writeln!(
                out,
                "  - ... {} more RD region(s)",
                report.rd_distribution.len() - 5
            );
        }
    }

    if !report.notes.is_empty() {
        let _ = writeln!(out, "Notes:");
        for note in &report.notes {
            let _ = writeln!(out, "  - {note}");
        }
    }

    out
}
