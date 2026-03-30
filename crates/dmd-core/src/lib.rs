pub mod ast;
mod error;
mod formula;
mod lexer;
mod parser;
mod polyhedral;
mod semantics;

pub use error::{DmdError, DmdResult};
pub use parser::parse_program;
pub use polyhedral::{
    AnalysisOptions, AnalysisReport, ApproximationMethod, DistributionEntry, DistributionRegion,
    DmdTerm, analyze_program, analyze_source,
};
pub use semantics::{ArrayInfo, SemanticProgram, validate_program};
