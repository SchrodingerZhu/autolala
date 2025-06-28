use std::io::Write;

use anyhow::{Result, anyhow};
use palc::Parser;
use raffine::Context;
use raffine::affine::{AffineExpr, AffineMap};
use raffine::tree::Tree;

struct CProgramEmitter<W> {
    writer: W,
    depth: usize,
    array_sizes: Box<[Box<[usize]>]>,
}

impl<W: Write> CProgramEmitter<W> {
    fn new(writer: W, iter: impl IntoIterator<Item = Box<[usize]>>) -> Self {
        CProgramEmitter {
            writer,
            depth: 0,
            array_sizes: iter.into_iter().collect(),
        }
    }

    fn emit_access(&mut self, array: usize, operands: &[usize], map: AffineMap) -> Result<()> {
        write!(self.writer, "\t{{ auto val_ = ARRAY_{array}[")?;
        for i in 0..map.num_results() {
            let expr = map
                .get_result_expr(i as isize)
                .ok_or_else(|| anyhow!("invalid affine map: result {i} does not exist in map"))?;
            if i > 0 {
                write!(self.writer, ",")?;
            }
            self.emit_affine_expr(expr.clone(), operands)?;
        }
        write!(
            self.writer,
            r#"]; __asm__ __volatile__ ("" : "+r,m"(val_) :: "memory"); }}\n"#
        )?;
        Ok(())
    }

    fn emit_affine_expr(&mut self, expr: AffineExpr, operands: &[usize]) -> Result<()> {
        match expr.get_kind() {
            raffine::affine::AffineExprKind::Add
            | raffine::affine::AffineExprKind::FloorDiv
            | raffine::affine::AffineExprKind::Mul
            | raffine::affine::AffineExprKind::Mod => {
                let operator = match expr.get_kind() {
                    raffine::affine::AffineExprKind::Add => "+",
                    raffine::affine::AffineExprKind::FloorDiv => "/",
                    raffine::affine::AffineExprKind::Mul => "*",
                    raffine::affine::AffineExprKind::Mod => "%",
                    _ => unreachable!(),
                };
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow!("addition should have lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow!("addition should have rhs"))?;
                write!(self.writer, "(")?;
                self.emit_affine_expr(lhs, operands)?;
                write!(self.writer, " {operator}")?;
                self.emit_affine_expr(rhs, operands)?;
                write!(self.writer, ")")?;
            }

            raffine::affine::AffineExprKind::Dim | raffine::affine::AffineExprKind::Symbol => {
                let operand = expr
                    .get_position()
                    .ok_or_else(|| anyhow!("dimension expression should have position"))?
                    as usize;
                let target = *operands
                    .get(operand)
                    .ok_or_else(|| anyhow!("invalid operand index"))?;
                let prefix = if matches!(expr.get_kind(), raffine::affine::AffineExprKind::Symbol) {
                    "SYM"
                } else {
                    "ivar"
                };
                write!(self.writer, "{prefix}_{target}",)?;
            }
            raffine::affine::AffineExprKind::CeilDiv => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow!("ceil division should have lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow!("ceil division should have rhs"))?;
                let decomposed = (lhs + lhs % rhs) / rhs;
                self.emit_affine_expr(decomposed, operands)?;
            }
            raffine::affine::AffineExprKind::Constant => {
                let value = expr
                    .get_value()
                    .ok_or_else(|| anyhow!("constant expression should have value"))?;
                write!(self.writer, "{value}")?;
            }
        }
        Ok(())
    }
}

fn main() {
    println!("Hello, world!");
}
