use anyhow::Result;
use barvinok::{
    DimType,
    aff::Affine,
    constraint::Constraint,
    list::List,
    local_space::LocalSpace,
    map::{BasicMap, Map},
    polynomial::{PiecewiseQuasiPolynomial, QuasiPolynomial},
    set::Set,
    space::Space,
    value::Value,
};
use comfy_table::Table;
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

#[derive(Clone, Debug)]
pub struct RIProcessor<'a> {
    pw_qpoly: PiecewiseQuasiPolynomial<'a>,
}

#[derive(Clone, Debug)]
pub struct Piece<'a> {
    domain: Set<'a>,
    qpoly: QuasiPolynomial<'a>,
}

#[derive(Clone, Debug)]
pub struct DistItem<'a> {
    qpoly: QuasiPolynomial<'a>,
    cardinality: PiecewiseQuasiPolynomial<'a>,
}

impl<'a> RIProcessor<'a> {
    pub fn new(pw_qpoly: PiecewiseQuasiPolynomial<'a>) -> Self {
        RIProcessor { pw_qpoly }
    }
    pub fn get_all_pieces(&self) -> Result<Box<[Piece<'a>]>, barvinok::Error> {
        let mut pieces: Vec<Piece<'_>> = Vec::new();
        self.pw_qpoly.foreach_piece(|qpoly, domain| {
            let mut to_merge = None;
            for existing in pieces.iter().enumerate() {
                // TODO: this can be sped up. currently O(n^2)
                if qpoly.plain_is_equal(&existing.1.qpoly)? {
                    to_merge = Some(existing.0);
                    break;
                }
            }
            if let Some(index) = to_merge {
                pieces[index].domain = pieces[index].domain.clone().union(domain)?;
            } else {
                pieces.push(Piece { domain, qpoly });
            }
            Ok(())
        })?;
        Ok(pieces.into_boxed_slice())
    }
    fn get_processed_pieces(&self) -> Result<Box<[Piece<'a>]>, barvinok::Error> {
        let mut pieces = self.get_all_pieces()?;
        for piece in pieces.iter_mut() {
            let involved_dims = piece.involved_input_dims()?;
            // move involved_dims into params space (currently for domain only)
            let mut domain = piece.domain.clone();
            for (shift, dim) in involved_dims.iter().enumerate() {
                // TODO: this should not unwrap
                let num_params = domain.get_dims(DimType::Param).unwrap();
                domain = domain.move_dims(
                    DimType::Param,
                    num_params,
                    DimType::Out,
                    *dim - shift as u32,
                    1,
                )?;
            }
            piece.domain = domain;
        }
        Ok(pieces)
    }
    pub fn get_distribution(&self) -> Result<Box<[DistItem<'a>]>, barvinok::Error> {
        let mut pieces = self.get_processed_pieces()?;
        let mut dist_items: Vec<DistItem<'_>> = Vec::new();
        for piece in pieces.iter_mut() {
            let cardinality = piece.cardinality()?;
            dist_items.push(DistItem {
                qpoly: piece.qpoly.clone(),
                cardinality,
            });
        }
        Ok(dist_items.into_boxed_slice())
    }
}

impl<'a> Piece<'a> {
    pub fn domain(&self) -> Set<'a> {
        self.domain.clone()
    }
    pub fn cardinality(&self) -> Result<PiecewiseQuasiPolynomial<'a>, barvinok::Error> {
        self.domain().cardinality()
    }
    fn involved_input_dims(&self) -> Result<Box<[u32]>, barvinok::Error> {
        let dims = self.qpoly.get_dim(DimType::In)?;
        let mut res = Vec::with_capacity(dims as usize);
        for i in 0..dims {
            if self.qpoly.involves_dims(DimType::In, i, 1)? {
                res.push(i);
            }
        }
        Ok(res.into_boxed_slice())
    }
}

pub fn create_table(dist: &[DistItem]) -> Option<Table> {
    let mut table = Table::new();
    table.set_header(vec!["RI Value", "Count", "Symbol Range"]);
    for item in dist.iter() {
        let poly_str = format!("{:?}", item.qpoly);
        // extract after second -> and before }
        let value = poly_str
            .split("->")
            .nth(2)
            .unwrap_or("")
            .split("}")
            .next()
            .unwrap_or("")
            .trim();
        let card_str = format!("{:?}", item.cardinality);
        // extract after { before :
        let counts = card_str
            .split("{")
            .nth(1)
            .unwrap_or("")
            .split("}")
            .next()
            .unwrap_or("")
            .trim();
        for count_range in counts.split(";") {
            let mut split = count_range.split(":");
            let count = split.next().unwrap_or("").trim();
            let range = split.next().unwrap_or("").trim();
            table.add_row([value, count, range]);
        }
    }
    Some(table)
}
