use std::collections::HashMap;

use anyhow::Result;
use barvinok::space::Space;
use raffine::affine::AffineExpr;
use raffine::tree::Tree;
use raffine::tree::ValID;
use symbolica::atom::Symbol;
use symbolica::domains::Ring;
use symbolica::domains::integer::IntegerRing;
use symbolica::domains::rational_polynomial::FromNumeratorAndDenominator;
use symbolica::domains::rational_polynomial::RationalPolynomial;
use symbolica::domains::rational_polynomial::RationalPolynomialField;
use symbolica::symbol;

pub type Poly = RationalPolynomial<IntegerRing, u32>;

use crate::AnalysisContext;

fn get_max_param_ivar<'a>(tree: &Tree<'a>) -> Result<(usize, usize)> {
    let mut max_param = 0;
    let mut max_ivar = 0;
    match tree {
        Tree::For {
            lower_bound_operands,
            upper_bound_operands,
            ivar,
            body,
            ..
        } => {
            let ValID::IVar(n) = ivar else {
                return Err(anyhow::anyhow!("invalid tree"));
            };
            max_ivar = max_ivar.max(*n);
            for id in lower_bound_operands
                .iter()
                .chain(upper_bound_operands.iter())
            {
                match id {
                    ValID::Symbol(n) => {
                        max_param = max_param.max(*n);
                    }
                    ValID::IVar(n) => {
                        max_ivar = max_ivar.max(*n);
                    }
                    _ => {}
                }
            }
            let (param, ivar) = get_max_param_ivar(body)?;
            max_param = max_param.max(param);
            max_ivar = max_ivar.max(ivar);
        }
        Tree::Block(blk) => {
            for subtree in blk.iter() {
                let (param, ivar) = get_max_param_ivar(subtree)?;
                max_param = max_param.max(param);
                max_ivar = max_ivar.max(ivar);
            }
        }
        Tree::Access { operands, .. } => {
            for id in operands.iter() {
                match id {
                    ValID::Symbol(n) => {
                        max_param = max_param.max(*n);
                    }
                    ValID::IVar(n) => {
                        max_ivar = max_ivar.max(*n);
                    }
                    _ => {}
                }
            }
        }
        Tree::If { .. } => return Err(anyhow::anyhow!("not implemented for conditional branch")),
    }
    Ok((max_param, max_ivar))
}

pub fn get_space<'a, 'b: 'a>(context: &AnalysisContext<'b>, tree: &Tree<'a>) -> Result<Space<'b>> {
    let (max_param, max_ivar) = get_max_param_ivar(tree)?;
    let space = Space::set(context.bcontext(), max_param as u32, max_ivar as u32)?;
    Ok(space)
}

/// Return levels of nesting if the loop is perfectly nested.
pub fn get_nesting_level(tree: &Tree) -> Option<usize> {
    match tree {
        Tree::For { body, .. } => get_nesting_level(body).map(|level| level + 1),
        Tree::Block(trees) => {
            if trees.iter().all(|t| matches!(t, Tree::Access { .. })) {
                Some(0)
            } else if trees.len() != 1 {
                None
            } else {
                let Some(t) = trees.first() else {
                    return None;
                };
                get_nesting_level(t)
            }
        }
        Tree::Access { .. } => Some(0),
        Tree::If { .. } => None,
    }
}

pub fn create_symbol_mapping<'a>(space: &Space<'a>) -> Result<HashMap<ValID, Symbol>> {
    let mut sym_map = HashMap::new();
    for i in 0..space.get_dim(barvinok::DimType::Param)? {
        let symbol = symbol!(format!("s{i}"));
        sym_map.insert(ValID::Symbol(i as usize), symbol);
    }
    for i in 0..space.get_dim(barvinok::DimType::Out)? {
        let symbol = symbol!(format!("i{i}"));
        sym_map.insert(ValID::IVar(i as usize), symbol);
    }
    Ok(sym_map)
}

pub struct ExprConverter<'b> {
    operands: &'b [ValID],
    sym_map: HashMap<ValID, Symbol>,
    integer_ring: IntegerRing,
    poly_field: RationalPolynomialField<IntegerRing, u32>,
}

impl<'b> ExprConverter<'b> {
    pub fn new(operands: &'b [ValID], sym_map: HashMap<ValID, Symbol>) -> Self {
        let integer_ring = IntegerRing::new();
        let poly_field = RationalPolynomialField::new(integer_ring);
        Self {
            operands,
            sym_map,
            integer_ring,
            poly_field,
        }
    }

    pub fn convert_polynomial<'a>(&self, affine_expr: AffineExpr<'a>) -> Result<Poly> {
        match affine_expr.get_kind() {
            raffine::affine::AffineExprKind::Add => {
                let lhs = affine_expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let rhs = affine_expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let lhs = self.convert_polynomial(lhs)?;
                let rhs = self.convert_polynomial(rhs)?;
                Ok(self.poly_field.add(&lhs, &rhs))
            }
            raffine::affine::AffineExprKind::Dim => {
                let id = affine_expr.get_position().ok_or_else(|| {
                    anyhow::anyhow!("invalid affine expression: missing position")
                })?;
                let val_id = self.operands.get(id as usize).ok_or_else(|| {
                    anyhow::anyhow!("invalid affine expression: invalid position")
                })?;
                let symbol = self
                    .sym_map
                    .get(val_id)
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid symbol"))?;
                todo!()
            }
            raffine::affine::AffineExprKind::Mod => todo!(),
            raffine::affine::AffineExprKind::Mul => todo!(),
            raffine::affine::AffineExprKind::Symbol => todo!(),
            raffine::affine::AffineExprKind::CeilDiv => todo!(),
            raffine::affine::AffineExprKind::Constant => todo!(),
            raffine::affine::AffineExprKind::FloorDiv => todo!(),
        }
    }
}
