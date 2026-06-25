//! Render an [`ExtractError`] as an `ariadne` diagnostic against the original
//! MLIR source, translating MLIR's 1-based line/column spans into byte offsets.

use ariadne::{Color, Label, Report, ReportKind, Source};

use crate::error::ExtractError;
use crate::ffi::SourceSpan;

/// Print `error` to stderr, underlining the offending operation in `source`
/// when a span is available. `filename` is used only as the report's source id.
pub fn report_error(filename: &str, source: &str, error: &ExtractError) {
    let id = filename.to_string();

    let Some(span) = error.span.and_then(|span| byte_range(source, span)) else {
        // No usable location: fall back to a header-only diagnostic.
        let mut builder =
            Report::build(ReportKind::Error, (id.clone(), 0..0)).with_message(&error.message);
        if let Some(help) = &error.help {
            builder = builder.with_help(help);
        }
        let _ = builder.finish().eprint((id, Source::from(source)));
        return;
    };

    let mut builder = Report::build(ReportKind::Error, (id.clone(), span.clone()))
        .with_message(&error.message)
        .with_label(
            Label::new((id.clone(), span))
                .with_message(&error.message)
                .with_color(Color::Red),
        );
    if let Some(help) = &error.help {
        builder = builder.with_help(help);
    }
    let _ = builder.finish().eprint((id, Source::from(source)));
}

/// Convert a 1-based file/line/column span into a byte range in `source`.
fn byte_range(source: &str, span: SourceSpan) -> Option<std::ops::Range<usize>> {
    let starts = line_starts(source);
    let start = offset_of(&starts, source.len(), span.start_line, span.start_col)?;
    let mut end = offset_of(&starts, source.len(), span.end_line, span.end_col)?;

    if end <= start {
        // Point location: widen to cover the operation token for a visible
        // underline (scan to the next whitespace, staying on the line).
        end = source[start..]
            .find(|c: char| c.is_whitespace())
            .map(|delta| start + delta)
            .unwrap_or(source.len())
            .max(start + 1)
            .min(source.len());
    }

    Some(start..end)
}

fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (index, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(index + 1);
        }
    }
    starts
}

fn offset_of(starts: &[usize], len: usize, line: u32, column: u32) -> Option<usize> {
    if line == 0 {
        return None;
    }
    let line_start = *starts.get((line - 1) as usize)?;
    let column = column.max(1) as usize - 1;
    Some((line_start + column).min(len))
}
