use anyhow::Result;
use barvinok::{
    DimType, aff::Affine, constraint::Constraint, local_space::LocalSpace, set::Set, space::Space,
    value::Value,
};
use raffine::{
    affine::{AffineExpr, AffineExprKind, AffineMap},
    tree::{Tree, ValID},
};

use crate::{AnalysisContext, utils::get_max_param_ivar};

pub fn get_space<'a, 'b: 'a>(context: &AnalysisContext<'b>, tree: &Tree<'a>) -> Result<Space<'b>> {
    let (max_param, max_ivar) = get_max_param_ivar(tree)?;
    let space = Space::set(
        context.bcontext(),
        max_param as u32 + 1,
        max_ivar as u32 + 1,
    )?;
    Ok(space)
}

pub fn get_timestamp_space<'a, 'b: 'a>(
    num_params: usize,
    depth: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
) -> Result<Set<'b>> {
    match tree {
        Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands,
            upper_bound_operands,
            step,
            body,
            ..
        } => {
            let set = get_timestamp_space(num_params, depth + 1, context, body)?;
            let space = set.get_space()?;
            let lower_converter =
                ExprConverter::new(space.clone(), *lower_bound, lower_bound_operands)?;
            let lower_bound =
                lower_converter.convert_polynomial(lower_bound.get_result_expr(0).ok_or_else(
                    || anyhow::anyhow!("invalid affine expression: at least one result expression"),
                )?)?;
            let upper_converter =
                ExprConverter::new(space.clone(), *upper_bound, upper_bound_operands)?;
            let upper_bound =
                upper_converter.convert_polynomial(upper_bound.get_result_expr(0).ok_or_else(
                    || anyhow::anyhow!("invalid affine expression: at least one result expression"),
                )?)?;
            let local_space = LocalSpace::try_from(space.clone())?;
            let step = Value::new_si(context.bcontext(), *step as i64);
            let step = Affine::val_on_domain(local_space.clone(), step)?;
            let trip_size = upper_bound
                .checked_sub(lower_bound)?
                .checked_div(step)?
                .checked_ceil()?;
            let ge_0 = Constraint::new_inequality(local_space.clone()).set_coefficient_si(
                DimType::Out,
                depth as u32,
                1,
            )?;
            let affine_minus_ivar = trip_size
                .checked_sub(Affine::var_on_domain(
                    local_space.clone(),
                    DimType::Out,
                    depth as u32,
                )?)?
                .checked_sub(Affine::val_on_domain(
                    local_space.clone(),
                    Value::new_si(context.bcontext(), 1),
                )?)?;
            let affine_minus_ivar_gt_0 = Constraint::new_inequality_from_affine(affine_minus_ivar);
            Ok(set
                .add_constraint(ge_0)?
                .add_constraint(affine_minus_ivar_gt_0)?)
        }
        Tree::Block(stmts) => {
            let mut sub_sets = stmts
                .iter()
                .map(|stmt| get_timestamp_space(num_params, depth + 1, context, stmt))
                .collect::<Result<Vec<_>>>()?;
            let longest = sub_sets
                .iter()
                .max_by_key(|set| set.num_dims().unwrap_or_default())
                .ok_or_else(|| anyhow::anyhow!("no sets found"))?
                .clone();
            let space = longest.get_space()?;
            let local_space = LocalSpace::try_from(space.clone())?;
            for (idx, i) in sub_sets.iter_mut().enumerate() {
                let length = i.num_dims()?;
                let mut s = i.clone().reset_space(space.clone())?;
                for j in length..longest.num_dims()? {
                    // add constraint eq 0
                    let constraint = Constraint::new_equality(local_space.clone())
                        .set_coefficient_si(DimType::Out, j as u32, 1)?;
                    s = s.add_constraint(constraint)?;
                }
                let current_dim_eq_i = Constraint::new_equality(local_space.clone())
                    .set_coefficient_si(DimType::Out, depth as u32, 1)?
                    .set_constant_si(-(idx as i32))?;
                *i = s.add_constraint(current_dim_eq_i)?;
            }
            let total_set = sub_sets
                .into_iter()
                .try_fold(Set::empty(space.clone())?, |acc, set| acc.union(set))?;
            Ok(total_set)
        }
        Tree::Access { .. } => {
            let space = Space::set(context.bcontext(), num_params as u32, depth as u32)?;
            Ok(Set::universe(space)?)
        }
        Tree::If { .. } => Err(anyhow::anyhow!("not implemented for conditional branch")),
    }
}

struct ExprConverter<'isl, 'mlir> {
    local_space: LocalSpace<'isl>,
    map: AffineMap<'mlir>,
    operands: &'mlir [ValID],
}

impl<'isl, 'mlir> ExprConverter<'isl, 'mlir> {
    pub fn new(
        space: Space<'isl>,
        map: AffineMap<'mlir>,
        operands: &'mlir [ValID],
    ) -> Result<Self> {
        let local_space = LocalSpace::try_from(space)?;
        Ok(Self {
            local_space,
            map,
            operands,
        })
    }

    pub fn convert_polynomial<'a>(&self, expr: AffineExpr<'a>) -> Result<Affine<'isl>> {
        let kind = expr.get_kind();
        match kind {
            AffineExprKind::Add => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid rhs"))?;
                let lhs = self.convert_polynomial(lhs)?;
                let rhs = self.convert_polynomial(rhs)?;
                Ok(lhs.checked_add(rhs)?)
            }
            AffineExprKind::Mod => Err(anyhow::anyhow!(
                "invalid affine expression: mod is not supported"
            )),
            AffineExprKind::Mul => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid rhs"))?;
                let lhs = self.convert_polynomial(lhs)?;
                let rhs = self.convert_polynomial(rhs)?;
                Ok(lhs.checked_mul(rhs)?)
            }
            AffineExprKind::Symbol | AffineExprKind::Dim => {
                let position = expr.get_position().ok_or_else(|| {
                    anyhow::anyhow!("invalid affine expression: invalid position")
                })?;
                self.position_to_var(position as usize, kind)
            }
            AffineExprKind::CeilDiv => todo!(),
            AffineExprKind::Constant => {
                let constant = expr.get_value().ok_or_else(|| {
                    anyhow::anyhow!("invalid affine expression: invalid constant")
                })?;
                let value = Value::new_si(self.local_space.context_ref(), constant);
                Ok(Affine::val_on_domain(self.local_space.clone(), value)?)
            }
            AffineExprKind::FloorDiv => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid rhs"))?;
                let lhs = self.convert_polynomial(lhs)?;
                let rhs = self.convert_polynomial(rhs)?;
                Ok(lhs.checked_div(rhs)?)
            }
        }
    }

    pub fn position_to_var(
        &self,
        mut position: usize,
        kind: AffineExprKind,
    ) -> Result<Affine<'isl>> {
        let dim_type = if matches!(kind, raffine::affine::AffineExprKind::Symbol) {
            position += self.map.num_dims();
            DimType::Param
        } else {
            DimType::Out
        };

        let val_id = *self
            .operands
            .get(position)
            .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid position"))?;

        match val_id {
            ValID::Symbol(n) | ValID::IVar(n) => Ok(Affine::var_on_domain(
                self.local_space.clone(),
                dim_type,
                n as u32,
            )?),
            _ => Err(anyhow::anyhow!("invalid affine expression")),
        }
    }
}

pub fn convert_affine_map<'isl, 'mlir>(
    map: AffineMap<'mlir>,
    operands: &'mlir [ValID],
    space: Space<'isl>,
) -> Result<Box<[Affine<'isl>]>> {
    let converter = ExprConverter::new(space, map, operands)?;
    let mut result = Vec::with_capacity(map.num_results());
    for i in 0..map.num_results() {
        let expr = map.get_result_expr(i as isize).ok_or_else(|| {
            anyhow::anyhow!("invalid affine expression: invalid result expression")
        })?;
        let poly = converter.convert_polynomial(expr)?;
        result.push(poly);
    }
    Ok(result.into_boxed_slice())
}

pub fn walk_tree_print_converted_affine_map<'a, 'b>(
    tree: &'a Tree<'a>,
    indent: usize,
    space: &Space<'b>,
) -> Result<()> {
    fn print_sequence<'b>(context: &str, poly_vec: &[Affine<'b>], indent: usize) {
        let indent_str = "  ".repeat(indent);
        println!("{indent_str}{}: ", context);
        for poly in poly_vec.iter() {
            println!("\t- {:?}", poly);
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
            let lower_bound_coverted =
                convert_affine_map(*lower_bound, lower_bound_operands, space.clone())?;
            let upper_bound_converted =
                convert_affine_map(*upper_bound, upper_bound_operands, space.clone())?;
            print_sequence("Lower bound", &lower_bound_coverted, indent);
            print_sequence("Upper bound", &upper_bound_converted, indent);
            walk_tree_print_converted_affine_map(body, indent + 1, space)?;
        }
        Tree::Block(trees) => {
            for subtree in trees.iter() {
                walk_tree_print_converted_affine_map(subtree, indent + 1, space)?;
            }
        }
        Tree::Access { map, operands, .. } => {
            let converted_map = convert_affine_map(*map, operands, space.clone())?;
            print_sequence("Access", &converted_map, indent);
        }
        Tree::If { .. } => return Err(anyhow::anyhow!("not implemented for conditional branch")),
    }
    Ok(())
}
