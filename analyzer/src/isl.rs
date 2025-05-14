use anyhow::Result;
use barvinok::{
    DimType,
    aff::Affine,
    constraint::Constraint,
    ident::Ident,
    list::List,
    local_space::LocalSpace,
    map::{BasicMap, Map},
    set::Set,
    space::Space,
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
    ivar_map: &mut Vec<usize>,
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
            ivar_map.push(depth);
            let set = get_timestamp_space(num_params, depth + 1, context, body, ivar_map)?;
            let space = set.get_space()?;
            let lower_converter =
                ExprConverter::new(space.clone(), *lower_bound, lower_bound_operands, ivar_map)?;
            let lower_bound =
                lower_converter.convert_polynomial(lower_bound.get_result_expr(0).ok_or_else(
                    || anyhow::anyhow!("invalid affine expression: at least one result expression"),
                )?)?;
            let upper_converter =
                ExprConverter::new(space.clone(), *upper_bound, upper_bound_operands, ivar_map)?;
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
            ivar_map.pop();
            Ok(set
                .add_constraint(ge_0)?
                .add_constraint(affine_minus_ivar_gt_0)?)
        }
        Tree::Block(stmts) => {
            let mut sub_sets = stmts
                .iter()
                .map(|stmt| get_timestamp_space(num_params, depth + 1, context, stmt, ivar_map))
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
                        .set_coefficient_si(DimType::Out, j, 1)?;
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

pub fn get_access_map<'a, 'b: 'a>(
    num_params: usize,
    depth: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
    ivar_map: &mut Vec<usize>,
) -> Result<Map<'b>> {
    match tree {
        Tree::For { body, .. } => {
            ivar_map.push(depth);
            let res = get_access_map(num_params, depth + 1, context, body, ivar_map);
            ivar_map.pop();
            res
        }
        Tree::Block(stmts) => {
            let mut sub_maps = stmts
                .iter()
                .map(|stmt| get_access_map(num_params, depth + 1, context, stmt, ivar_map))
                .collect::<Result<Vec<_>>>()?;
            let longest = sub_maps
                .iter()
                .max_by_key(|map| {
                    map.get_space()
                        .and_then(|s| s.get_dim(DimType::In))
                        .unwrap_or_default()
                })
                .ok_or_else(|| anyhow::anyhow!("no maps found"))?
                .clone();
            let space = longest.get_space()?;
            let local_space = LocalSpace::try_from(space.clone())?;
            let longest_length = longest.get_space()?.get_dim(DimType::In)?;
            for (idx, i) in sub_maps.iter_mut().enumerate() {
                let length = i.get_space()?.get_dim(DimType::In)?;
                let mut s = i.clone().add_dims(DimType::In, longest_length - length)?;
                for j in length..longest_length {
                    // add constraint eq 0
                    let constraint = Constraint::new_equality(local_space.clone())
                        .set_coefficient_si(DimType::In, j, 1)?;
                    s = s.add_constraint(constraint)?;
                }
                let current_dim_eq_i = Constraint::new_equality(local_space.clone())
                    .set_coefficient_si(DimType::In, depth as u32, 1)?
                    .set_constant_si(-(idx as i32))?;
                *i = s.add_constraint(current_dim_eq_i)?;
            }
            let total_map = sub_maps
                .into_iter()
                .try_fold(Map::empty(space.clone())?, |acc, set| acc.union(set))?;
            Ok(total_map)
        }
        Tree::Access {
            map,
            operands,
            memref,
            ..
        } => {
            let domain_space = Space::set(context.bcontext(), num_params as u32, depth as u32)?;
            let converter = ExprConverter::new(domain_space.clone(), *map, operands, ivar_map)?;
            let mut aff_list = List::new(domain_space.context_ref(), map.num_results());
            let ValID::Memref(memref) = *memref else {
                return Err(anyhow::anyhow!("invalid access map: invalid memref"));
            };
            let val = Value::new_ui(domain_space.context_ref(), memref as u64);
            let aff = Affine::val_on_domain_space(domain_space.clone(), val)?;
            aff_list.push(aff);
            for i in 0..map.num_results() {
                let expr = map
                    .get_result_expr(i as isize)
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid result"))?;
                tracing::debug!("expr: {}", expr);
                let aff = converter.convert_polynomial(expr)?;
                aff_list.push(aff);
            }
            let basic_map = BasicMap::from_affine_list(domain_space, aff_list)?;
            Ok(basic_map.try_into()?)
        }
        Tree::If { .. } => Err(anyhow::anyhow!("not implemented for conditional branch")),
    }
}

struct ExprConverter<'isl, 'mlir, 'map> {
    local_space: LocalSpace<'isl>,
    ivar_map: &'map Vec<usize>,
    map: AffineMap<'mlir>,
    operands: &'mlir [ValID],
}

impl<'isl, 'mlir, 'map> ExprConverter<'isl, 'mlir, 'map> {
    pub fn new(
        space: Space<'isl>,
        map: AffineMap<'mlir>,
        operands: &'mlir [ValID],
        ivar_map: &'map Vec<usize>,
    ) -> Result<Self> {
        let local_space = LocalSpace::try_from(space)?;
        Ok(Self {
            local_space,
            map,
            operands,
            ivar_map,
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
            ValID::Symbol(n) => Ok(Affine::var_on_domain(
                self.local_space.clone(),
                dim_type,
                n as u32,
            )?),
            ValID::IVar(n) => Ok(Affine::var_on_domain(
                self.local_space.clone(),
                dim_type,
                self.ivar_map[n] as u32,
            )?),
            _ => Err(anyhow::anyhow!("invalid affine expression")),
        }
    }
}
