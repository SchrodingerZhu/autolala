//! Render a `dmd-core` AST [`Program`] back into AutoLALA DSL source text.
//!
//! The output is round-trip safe: feeding it back through
//! `dmd_core::parse_program` reproduces an equivalent program (the CLI asserts
//! this as a self-check).

use std::fmt::Write;

use dmd_core::ast::{
    Access, AccessKind, Block, Comparison, ComparisonOp, Expr, ForLoop, IfStmt, Program, Stmt,
};

const INDENT: &str = "    ";

/// Precedence levels used to parenthesize expressions minimally.
mod precedence {
    pub const ADD: u8 = 1;
    pub const MUL: u8 = 2;
    pub const UNARY: u8 = 3;
    pub const ATOM: u8 = 4;
}

/// Emit a full program: `params`, `array` declarations, then the body.
pub fn emit_program(program: &Program) -> String {
    let mut out = String::new();

    if !program.params.is_empty() {
        let _ = writeln!(out, "params {};", program.params.join(", "));
    }

    for array in &program.arrays {
        let extents = array
            .extents
            .iter()
            .map(|extent| emit_expr(extent, 0))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "array {}[{}];", array.name, extents);
    }

    if !program.params.is_empty() || !program.arrays.is_empty() {
        out.push('\n');
    }

    emit_block(&program.body, 0, &mut out);
    out
}

fn emit_block(block: &Block, indent: usize, out: &mut String) {
    for statement in &block.statements {
        emit_stmt(statement, indent, out);
    }
}

fn emit_stmt(statement: &Stmt, indent: usize, out: &mut String) {
    match statement {
        Stmt::For(for_loop) => emit_for(for_loop, indent, out),
        Stmt::If(if_stmt) => emit_if(if_stmt, indent, out),
        Stmt::Access(access) => emit_access(access, indent, out),
    }
}

fn emit_for(for_loop: &ForLoop, indent: usize, out: &mut String) {
    let pad = INDENT.repeat(indent);
    let _ = write!(
        out,
        "{pad}for {} in {} .. {}",
        for_loop.var,
        emit_expr(&for_loop.lower, 0),
        emit_expr(&for_loop.upper, 0),
    );
    if for_loop.step != 1 {
        let _ = write!(out, " step {}", for_loop.step);
    }
    out.push_str(" {\n");
    emit_block(&for_loop.body, indent + 1, out);
    let _ = writeln!(out, "{pad}}}");
}

fn emit_if(if_stmt: &IfStmt, indent: usize, out: &mut String) {
    let pad = INDENT.repeat(indent);
    let conditions = if_stmt
        .conditions
        .iter()
        .map(emit_comparison)
        .collect::<Vec<_>>()
        .join(" && ");
    let _ = writeln!(out, "{pad}if {conditions} {{");
    emit_block(&if_stmt.then_branch, indent + 1, out);
    if let Some(else_branch) = &if_stmt.else_branch {
        let _ = writeln!(out, "{pad}}} else {{");
        emit_block(else_branch, indent + 1, out);
    }
    let _ = writeln!(out, "{pad}}}");
}

fn emit_access(access: &Access, indent: usize, out: &mut String) {
    let pad = INDENT.repeat(indent);
    let keyword = match access.kind {
        AccessKind::Read => "read",
        AccessKind::Write => "write",
        AccessKind::Update => "update",
    };
    let indices = access
        .indices
        .iter()
        .map(|index| emit_expr(index, 0))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "{pad}{keyword} {}[{indices}];", access.array);
}

fn emit_comparison(comparison: &Comparison) -> String {
    let op = match comparison.op {
        ComparisonOp::Lt => "<",
        ComparisonOp::Le => "<=",
        ComparisonOp::Eq => "==",
        ComparisonOp::Ge => ">=",
        ComparisonOp::Gt => ">",
    };
    format!(
        "{} {op} {}",
        emit_expr(&comparison.lhs, 0),
        emit_expr(&comparison.rhs, 0)
    )
}

/// If `expr` is syntactically negative (`-k`, `k` negative, or `-1 * y`),
/// return the corresponding positive expression so the caller can emit a `-`.
fn negated(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Int(value) if *value < 0 => Some(Expr::Int(-value)),
        Expr::Neg(inner) => Some((**inner).clone()),
        Expr::Mul(lhs, rhs) => match (lhs.as_const_i64(), rhs.as_const_i64()) {
            (Some(-1), _) => Some((**rhs).clone()),
            (_, Some(-1)) => Some((**lhs).clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Render an expression, wrapping in parentheses only when the node's
/// precedence is below what the surrounding context (`parent`) requires.
fn emit_expr(expr: &Expr, parent: u8) -> String {
    use precedence::*;
    let (text, prec) = match expr {
        Expr::Int(value) => (value.to_string(), ATOM),
        Expr::Var(name) => (name.clone(), ATOM),
        Expr::Neg(inner) => (format!("-{}", emit_expr(inner, UNARY)), UNARY),
        Expr::Mul(lhs, rhs) => (
            format!("{} * {}", emit_expr(lhs, MUL), emit_expr(rhs, MUL)),
            MUL,
        ),
        Expr::FloorDiv(lhs, rhs) => (
            format!("{} / {}", emit_expr(lhs, MUL), emit_expr(rhs, UNARY)),
            MUL,
        ),
        // Render `x + (-k)` and `x + (-1 * y)` as subtractions for readability;
        // MLIR encodes subtraction this way since affine exprs have no `-`.
        Expr::Add(lhs, rhs) => {
            let text = match negated(rhs) {
                Some(positive) => {
                    format!("{} - {}", emit_expr(lhs, ADD), emit_expr(&positive, MUL))
                }
                None => format!("{} + {}", emit_expr(lhs, ADD), emit_expr(rhs, ADD)),
            };
            (text, ADD)
        }
        Expr::Sub(lhs, rhs) => (
            format!("{} - {}", emit_expr(lhs, ADD), emit_expr(rhs, MUL)),
            ADD,
        ),
    };
    if prec < parent {
        format!("({text})")
    } else {
        text
    }
}
