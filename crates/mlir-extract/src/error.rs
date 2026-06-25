//! Error type for extraction, carrying an optional MLIR source span so the CLI
//! can render an `ariadne` diagnostic that points at the offending operation.

use crate::ffi::SourceSpan;

/// A translation failure: an unsupported or malformed construct in the input.
#[derive(Debug, Clone)]
pub struct ExtractError {
    /// Primary, human-readable description of the problem.
    pub message: String,
    /// The location of the offending operation, when MLIR recorded one.
    pub span: Option<SourceSpan>,
    /// Optional actionable hint shown as a `help:` note.
    pub help: Option<String>,
}

impl ExtractError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            help: None,
        }
    }

    pub fn at(message: impl Into<String>, span: Option<SourceSpan>) -> Self {
        Self {
            message: message.into(),
            span,
            help: None,
        }
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ExtractError {}

pub type ExtractResult<T> = Result<T, ExtractError>;
