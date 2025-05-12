use anyhow::Result;
use barvinok::space::Space;
use raffine::affine::AffineExpr;
use raffine::affine::AffineMap;
use raffine::tree::Tree;
use raffine::tree::ValID;
use symbolica::atom::Atom;
use symbolica::atom::AtomCore;
use symbolica::domains::Field;
use symbolica::domains::Ring;
use symbolica::domains::integer::IntegerRing;
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
                let t = trees.first()?;
                get_nesting_level(t)
            }
        }
        Tree::Access { .. } => Some(0),
        Tree::If { .. } => None,
    }
}

struct ExprConverter<'b> {
    operands: &'b [ValID],
    integer_ring: IntegerRing,
    poly_field: RationalPolynomialField<IntegerRing, u32>,
}

impl<'b> ExprConverter<'b> {
    pub fn new(operands: &'b [ValID]) -> Self {
        let integer_ring = IntegerRing::new();
        let poly_field = RationalPolynomialField::new(integer_ring);
        Self {
            operands,
            integer_ring,
            poly_field,
        }
    }

    pub fn convert_polynomial<'a, 'm: 'a>(
        &self,
        map: AffineMap<'m>,
        affine_expr: AffineExpr<'a>,
    ) -> Result<Poly> {
        let kind = affine_expr.get_kind();
        match kind {
            raffine::affine::AffineExprKind::Add => {
                let lhs = affine_expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let rhs = affine_expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let lhs = self.convert_polynomial(map, lhs)?;
                let rhs = self.convert_polynomial(map, rhs)?;
                Ok(self.poly_field.add(&lhs, &rhs))
            }
            raffine::affine::AffineExprKind::Dim | raffine::affine::AffineExprKind::Symbol => {
                let mut id = affine_expr
                    .get_position()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: missing position"))?
                    as usize;
                if matches!(kind, raffine::affine::AffineExprKind::Symbol) {
                    id += map.num_dims();
                }
                let val_id = self.operands.get(id).ok_or_else(|| {
                    anyhow::anyhow!("invalid affine expression: invalid position")
                })?;
                let symbol = match val_id {
                    ValID::Symbol(n) => symbol!(format!("s{n}")),
                    ValID::IVar(n) => symbol!(format!("i{n}")),
                    _ => return Err(anyhow::anyhow!("invalid affine expression")),
                };
                let atom = Atom::new_var(symbol);
                let poly =
                    atom.to_rational_polynomial(&self.integer_ring, &self.integer_ring, None);
                Ok(poly)
            }
            raffine::affine::AffineExprKind::Mod => Err(anyhow::anyhow!("mod not supported")),
            raffine::affine::AffineExprKind::Mul => {
                let lhs = affine_expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let rhs = affine_expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let lhs = self.convert_polynomial(map, lhs)?;
                let rhs = self.convert_polynomial(map, rhs)?;
                Ok(self.poly_field.mul(&lhs, &rhs))
            }
            raffine::affine::AffineExprKind::CeilDiv
            | raffine::affine::AffineExprKind::FloorDiv => {
                let lhs = affine_expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let rhs = affine_expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression"))?;
                let lhs = self.convert_polynomial(map, lhs)?;
                let rhs = self.convert_polynomial(map, rhs)?;
                Ok(self.poly_field.div(&lhs, &rhs))
            }
            raffine::affine::AffineExprKind::Constant => {
                let val = affine_expr
                    .get_value()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: missing value"))?;
                let atom = Atom::new_num(val);
                let poly =
                    atom.to_rational_polynomial(&self.integer_ring, &self.integer_ring, None);
                Ok(poly)
            }
        }
    }
}

pub fn convert_affine_map<'a>(map: AffineMap<'a>, operands: &'a [ValID]) -> Result<Box<[Poly]>> {
    let converter = ExprConverter::new(operands);
    let mut result = Vec::with_capacity(map.num_results());
    for i in 0..map.num_results() {
        let affine_expr = map.get_result_expr(i as isize);
        let poly = converter.convert_polynomial(map, affine_expr)?;
        result.push(poly);
    }
    Ok(result.into_boxed_slice())
}

pub fn walk_tree_print_converted_affine_map<'a>(tree: &'a Tree<'a>, indent: usize) -> Result<()> {
    fn print_sequence(context: &str, poly_vec: &[Poly], indent: usize) {
        let indent_str = "  ".repeat(indent);
        println!("{indent_str}{}: ", context);
        for poly in poly_vec.iter() {
            println!("\t- {}", poly);
        }
    }
    match tree {
        Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands,
            upper_bound_operands,
            body,
            ..
        } => {
            let lower_bound_coverted = convert_affine_map(*lower_bound, lower_bound_operands)?;
            let upper_bound_converted = convert_affine_map(*upper_bound, upper_bound_operands)?;
            print_sequence("Lower bound", &lower_bound_coverted, indent);
            print_sequence("Upper bound", &upper_bound_converted, indent);
            walk_tree_print_converted_affine_map(body, indent + 1)?;
        }
        Tree::Block(trees) => {
            for subtree in trees.iter() {
                walk_tree_print_converted_affine_map(subtree, indent + 1)?;
            }
        }
        Tree::Access { map, operands, .. } => {
            let converted_map = convert_affine_map(*map, operands)?;
            print_sequence("Access", &converted_map, indent);
        }
        Tree::If { .. } => return Err(anyhow::anyhow!("not implemented for conditional branch")),
    }
    Ok(())
}
