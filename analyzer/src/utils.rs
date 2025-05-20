use anyhow::Result;
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

pub(crate) fn get_max_param_ivar<'a>(tree: &Tree<'a>) -> Result<(isize, isize)> {
    let mut max_param = -1;
    let mut max_ivar = -1;
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
            max_ivar = max_ivar.max(*n as isize);
            for id in lower_bound_operands
                .iter()
                .chain(upper_bound_operands.iter())
            {
                match id {
                    ValID::Symbol(n) => {
                        max_param = max_param.max(*n as isize);
                    }
                    ValID::IVar(n) => {
                        max_ivar = max_ivar.max(*n as isize);
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
                        max_param = max_param.max(*n as isize);
                    }
                    ValID::IVar(n) => {
                        max_ivar = max_ivar.max(*n as isize);
                    }
                    _ => {}
                }
            }
        }
        Tree::If { .. } => return Err(anyhow::anyhow!("not implemented for conditional branch")),
    }
    Ok((max_param, max_ivar))
}

pub(crate) fn get_max_array_dim<'a>(tree: &Tree<'a>) -> Result<usize> {
    match tree {
        Tree::For { body, .. } => get_max_array_dim(body),
        Tree::Block(blk) => blk.iter().try_fold(0, |acc, subtree| {
            let dim = get_max_array_dim(subtree)?;
            Ok(dim.max(acc))
        }),
        Tree::Access { map, .. } => Ok(map.num_results()),
        Tree::If { then, r#else, .. } => {
            let a = get_max_array_dim(then)?;
            let b = r#else.map(|r| get_max_array_dim(r)).unwrap_or(Ok(0))?;
            Ok(a.max(b))
        }
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
        let affine_expr = map
            .get_result_expr(i as isize)
            .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid result index"))?;
        let poly = converter.convert_polynomial(map, affine_expr)?;
        result.push(poly);
    }
    Ok(result.into_boxed_slice())
}

// pub fn walk_tree_print_converted_affine_map<'a>(tree: &'a Tree<'a>, indent: usize) -> Result<()> {
//     fn print_sequence(context: &str, poly_vec: &[Poly], indent: usize) {
//         let indent_str = "  ".repeat(indent);
//         println!("{indent_str}{}: ", context);
//         for poly in poly_vec.iter() {
//             println!("\t- {}", poly);
//         }
//     }
//     match tree {
//         Tree::For {
//             lower_bound,
//             upper_bound,
//             lower_bound_operands,
//             upper_bound_operands,
//             body,
//             ..
//         } => {
//             let lower_bound_coverted = convert_affine_map(*lower_bound, lower_bound_operands)?;
//             let upper_bound_converted = convert_affine_map(*upper_bound, upper_bound_operands)?;
//             print_sequence("Lower bound", &lower_bound_coverted, indent);
//             print_sequence("Upper bound", &upper_bound_converted, indent);
//             walk_tree_print_converted_affine_map(body, indent + 1)?;
//         }
//         Tree::Block(trees) => {
//             for subtree in trees.iter() {
//                 walk_tree_print_converted_affine_map(subtree, indent + 1)?;
//             }
//         }
//         Tree::Access { map, operands, .. } => {
//             let converted_map = convert_affine_map(*map, operands)?;
//             print_sequence("Access", &converted_map, indent);
//         }
//         Tree::If { .. } => return Err(anyhow::anyhow!("not implemented for conditional branch")),
//     }
//     Ok(())
// }

pub fn create_table(dist: &[(Poly, Poly)]) -> comfy_table::Table {
    use comfy_table::ContentArrangement;
    use comfy_table::modifiers::UTF8_ROUND_CORNERS;
    use comfy_table::presets::UTF8_FULL;
    let mut table = comfy_table::Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["RI Value", "Portion"]);
    for item in dist.iter() {
        let value = format!("{}", item.0);
        let portion = format!("{}", item.1);
        table.add_row([value, portion]);
    }
    table
}
