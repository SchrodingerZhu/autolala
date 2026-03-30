use crate::ast::Program;
use crate::error::{DmdError, DmdResult};
use crate::lexer::lex;

mod grammar {
    #![allow(clippy::all)]
    #![allow(clippy::unwrap_used)]
    include!(concat!(env!("OUT_DIR"), "/grammar.rs"));
}

pub fn parse_program(source: &str) -> DmdResult<Program> {
    let tokens = lex(source)?;
    grammar::ProgramParser::new()
        .parse(tokens)
        .map_err(DmdError::from_parse_error)
}

#[cfg(test)]
mod tests {
    use super::parse_program;
    use crate::ast::{AccessKind, Expr, Stmt};

    #[test]
    fn parses_simple_loop_tree() {
        let source = r#"
params N, M;
array A[N, M];

for i in 0 .. N {
    for j in 0 .. M {
        read A[i, j];
    }
}
"#;
        let program = parse_program(source).expect("parser should accept affine program");
        assert_eq!(program.params, vec!["N", "M"]);
        assert_eq!(program.arrays.len(), 1);
        assert_eq!(program.body.statements.len(), 1);
        let Stmt::For(outer) = &program.body.statements[0] else {
            panic!("expected an outer loop");
        };
        assert_eq!(outer.var, "i");
        let Stmt::For(inner) = &outer.body.statements[0] else {
            panic!("expected an inner loop");
        };
        let Stmt::Access(access) = &inner.body.statements[0] else {
            panic!("expected an access");
        };
        assert!(matches!(access.kind, AccessKind::Read));
        assert!(matches!(access.indices[0], Expr::Var(ref name) if name == "i"));
    }
}
