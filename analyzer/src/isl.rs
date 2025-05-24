use std::collections::hash_map::Entry;

use crate::utils::{Poly, get_max_array_dim};
use ahash::AHashMap;
use anyhow::Result;
use barvinok::{
    DimType,
    aff::Affine,
    constraint::Constraint,
    list::List,
    local_space::LocalSpace,
    map::{BasicMap, Map},
    point::Point,
    polynomial::{PiecewiseQuasiPolynomial, QuasiPolynomial, Term},
    set::Set,
    space::Space,
    value::Value,
};
use comfy_table::Table;
use raffine::{
    affine::{AffineExpr, AffineExprKind, AffineMap},
    tree::{Tree, ValID},
};

use serde::Serialize;
use symbolica::{atom::Atom, domains::Field, domains::integer::IntegerRing};
use symbolica::{atom::AtomCore, symbol};
use symbolica::{
    domains::{Ring, rational_polynomial::RationalPolynomialField},
    printer::PrintOptions,
};

use crate::AnalysisContext;

struct ConvertedIVar<'a> {
    lower_bound: AffineMap<'a>,
    step_size: i64,
    index: usize,
    operands: &'a [ValID],
}

type IVarMap<'a> = Vec<ConvertedIVar<'a>>;

pub fn get_timestamp_space<'a, 'b: 'a>(
    num_params: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
) -> Result<Set<'b>> {
    let mut ivar_map = Vec::new();
    get_timestamp_space_impl(num_params, 0, context, tree, &mut ivar_map)
}

fn get_timestamp_space_impl<'a, 'b: 'a>(
    num_params: usize,
    depth: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
    ivar_map: &mut IVarMap<'a>,
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
            {
                let step_size = *step as i64;
                let index = depth;
                let ivar = ConvertedIVar {
                    lower_bound: *lower_bound,
                    step_size,
                    index,
                    operands: lower_bound_operands,
                };
                ivar_map.push(ivar);
            }
            let set = get_timestamp_space_impl(num_params, depth + 1, context, body, ivar_map)?;
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
                .add_constraint(affine_minus_ivar_gt_0)?
                .set_dim_name(
                    DimType::Out,
                    depth as u32,
                    &format!("i{}", ivar_map.len() + 1),
                )?)
        }
        Tree::Block(stmts) => {
            let mut sub_sets = stmts
                .iter()
                .map(|stmt| {
                    get_timestamp_space_impl(num_params, depth + 1, context, stmt, ivar_map)
                })
                .collect::<Result<Vec<_>>>()?;
            let longest = sub_sets
                .iter()
                .max_by_key(|set| set.num_dims().unwrap_or_default())
                .ok_or_else(|| anyhow::anyhow!("no sets found"))?
                .clone();
            let space = longest.get_space()?;
            let longest_dim = longest.num_dims()?;
            let value_one = Value::new_si(context.bcontext(), 1);
            let local_space = LocalSpace::try_from(space.clone())?;
            for (idx, i) in sub_sets.iter_mut().enumerate() {
                let length = i.num_dims()?;
                let mut s = i
                    .clone()
                    .insert_dims(DimType::Out, length, longest_dim - length)?;
                for j in length..longest.num_dims()? {
                    // add constraint eq 0
                    let constraint = Constraint::new_equality(local_space.clone())
                        .set_coefficient_val(DimType::Out, j, value_one.clone())?;
                    s = s.add_constraint(constraint)?;
                }
                let current_dim_eq_i = Constraint::new_equality(local_space.clone())
                    .set_coefficient_val(DimType::Out, depth as u32, value_one.clone())?
                    .set_constant_si(-(idx as i32))?;
                *i = s.add_constraint(current_dim_eq_i)?;
            }
            let total_set = sub_sets
                .into_iter()
                .try_fold(Set::empty(space.clone())?, |acc, set| acc.union(set))?
                .set_dim_name(
                    DimType::Out,
                    depth as u32,
                    &format!("t{}", depth - ivar_map.len()),
                )?;
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
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
    block_size: usize,
) -> Result<Map<'b>> {
    let mut ivar_map = Vec::new();
    let max_array_dim = get_max_array_dim(tree)?;
    get_access_map_impl(
        num_params,
        0,
        context,
        tree,
        &mut ivar_map,
        block_size,
        max_array_dim,
    )
}

fn get_access_map_impl<'a, 'b: 'a>(
    num_params: usize,
    depth: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
    ivar_map: &mut IVarMap<'a>,
    block_size: usize,
    max_array_dim: usize,
) -> Result<Map<'b>> {
    match tree {
        Tree::For {
            body,
            lower_bound,
            lower_bound_operands,
            step,
            ..
        } => {
            {
                let step_size = *step as i64;
                let index = depth;
                let ivar = ConvertedIVar {
                    lower_bound: *lower_bound,
                    step_size,
                    index,
                    operands: lower_bound_operands,
                };
                ivar_map.push(ivar);
            }
            let res = get_access_map_impl(
                num_params,
                depth + 1,
                context,
                body,
                ivar_map,
                block_size,
                max_array_dim,
            )?;
            ivar_map.pop();
            Ok(res.set_dim_name(
                DimType::In,
                depth as u32,
                &format!("i{}", ivar_map.len() + 1),
            )?)
        }
        Tree::Block(stmts) => {
            let mut sub_maps = stmts
                .iter()
                .map(|stmt| {
                    get_access_map_impl(
                        num_params,
                        depth + 1,
                        context,
                        stmt,
                        ivar_map,
                        block_size,
                        max_array_dim,
                    )
                })
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
                .try_fold(Map::empty(space.clone())?, |acc, set| acc.union(set))?
                .set_dim_name(
                    DimType::In,
                    depth as u32,
                    &format!("t{}", depth - ivar_map.len()),
                )?;
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
            // align array dimension
            for _ in 0..max_array_dim - map.num_results() {
                let val = Value::new_ui(domain_space.context_ref(), 0);
                let aff = Affine::val_on_domain_space(domain_space.clone(), val)?;
                aff_list.push(aff);
            }
            for i in 0..map.num_results() {
                let expr = map
                    .get_result_expr(i as isize)
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid result"))?;
                tracing::debug!("expr: {}", expr);
                let mut aff = converter.convert_polynomial(expr)?;
                if block_size > 1 && i == map.num_results() - 1 {
                    let block_size = Value::new_ui(domain_space.context_ref(), block_size as u64);
                    let block_size = Affine::val_on_domain_space(domain_space.clone(), block_size)?;
                    aff = aff.checked_div(block_size)?;
                    aff = aff.floor()?;
                }
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
    ivar_map: &'map IVarMap<'mlir>,
    map: AffineMap<'mlir>,
    operands: &'mlir [ValID],
}

impl<'isl, 'mlir, 'map> ExprConverter<'isl, 'mlir, 'map> {
    pub fn new(
        space: Space<'isl>,
        map: AffineMap<'mlir>,
        operands: &'mlir [ValID],
        ivar_map: &'map IVarMap<'mlir>,
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
                Ok(lhs.checked_div(rhs)?.floor()?)
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
            ValID::IVar(n) => {
                let ivar = Affine::var_on_domain(
                    self.local_space.clone(),
                    dim_type,
                    self.ivar_map[n].index as u32,
                )?;
                let step_size =
                    Value::new_si(self.local_space.context_ref(), self.ivar_map[n].step_size);
                let step_size = Affine::val_on_domain(self.local_space.clone(), step_size)?;
                let converter = Self {
                    local_space: self.local_space.clone(),
                    ivar_map: self.ivar_map,
                    map: self.ivar_map[n].lower_bound,
                    operands: self.ivar_map[n].operands,
                };
                let lower_bound = converter.convert_polynomial(
                    self.ivar_map[n]
                        .lower_bound
                        .get_result_expr(0)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "invalid affine expression: at least one result expression"
                            )
                        })?,
                )?;
                Ok(ivar.checked_mul(step_size)?.checked_add(lower_bound)?)
            }
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

pub fn create_table(
    dist: &[DistItem],
    total: PiecewiseQuasiPolynomial<'_>,
    infinite_repeat: bool,
) -> Result<Table> {
    use comfy_table::ContentArrangement;
    use comfy_table::modifiers::UTF8_ROUND_CORNERS;
    use comfy_table::presets::UTF8_FULL;
    let mut total_count = None;
    total.foreach_piece(|qpoly, _| {
        total_count.replace(qpoly);
        Ok(())
    })?;
    let total_count = total_count.ok_or_else(|| anyhow::anyhow!("no total count found"))?;
    let total_count_poly = convert_quasi_poly(total_count)?;
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["RI Value", "Count", "Symbol Range", "Portion"]);
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    for item in dist.iter() {
        let value = convert_quasi_poly(item.qpoly.clone())?;
        let value_str = format!("{value}");
        item.cardinality.foreach_piece(|qpoly, domain| {
            let poly = convert_quasi_poly(qpoly.clone())?;
            let count = format!("{poly}");
            let range = format!("{domain:?}");
            let range = range
                .split("{  : ")
                .nth(1)
                .unwrap_or_default()
                .split(" }")
                .next()
                .unwrap_or_default();
            // L'Hôpital's rule
            let portion = if infinite_repeat {
                tracing::debug!("applying L'Hôpital's rule for {poly}/{total_count_poly}");
                let poly_var = poly.get_variables().iter().position(|x| {
                    x.to_id()
                        .map(|x| x.get_stripped_name() == "R")
                        .unwrap_or_default()
                });
                if poly.is_zero() || poly_var.is_none() {
                    table.add_row([&value_str, &count, range, "0"]);
                    return Ok(());
                }
                let total_var = total_count_poly
                    .get_variables()
                    .iter()
                    .position(|x| {
                        x.to_id()
                            .map(|x| x.get_stripped_name() == "R")
                            .unwrap_or_default()
                    })
                    .unwrap();
                let poly = poly.derivative(poly_var.unwrap());
                let total = total_count_poly.derivative(total_var);
                if total.is_zero() {
                    table.add_row([&value_str, &count, range, "∞"]);
                    return Ok(());
                }
                field.div(&poly, &total)
            } else {
                field.div(&poly, &total_count_poly)
            };
            let portion_str = format!("{portion}");
            table.add_row([&value_str, &count, range, &portion_str]);
            Ok(())
        })?;
    }
    Ok(table)
}

struct QpolyConverter<'a> {
    ring: IntegerRing,
    field: RationalPolynomialField<IntegerRing, u32>,
    space: Space<'a>,
}

fn convert_quasi_poly<'a>(
    qpoly: QuasiPolynomial<'a>,
) -> std::result::Result<Poly, barvinok::Error> {
    let space = qpoly.get_space()?;
    let converter = QpolyConverter::new(space)?;
    converter.quasi_poly_to_rational_poly(qpoly)
}

impl<'a> QpolyConverter<'a> {
    pub fn new(space: Space<'a>) -> std::result::Result<Self, barvinok::Error> {
        let ring = IntegerRing::new();
        let field = RationalPolynomialField::new(ring);
        Ok(QpolyConverter { ring, field, space })
    }
    fn value_to_rational_poly(&self, value: Value<'a>) -> Poly {
        let num = value.numerator();
        let denom = value.denominator();
        let num: Atom = Atom::new_num(num);
        let denom: Atom = Atom::new_num(denom);
        let num = num.to_rational_polynomial(&self.ring, &self.ring, None);
        let denom = denom.to_rational_polynomial(&self.ring, &self.ring, None);
        self.field.div(&num, &denom)
    }

    fn term_to_rational_poly(&self, term: Term<'a>) -> std::result::Result<Poly, barvinok::Error> {
        let mut poly = self.value_to_rational_poly(term.coefficient()?);
        let params = self.space.get_dim(DimType::Param)?;
        let in_dims = self.space.get_dim(DimType::In)?;
        let dims = std::iter::repeat_n(DimType::Param, params as usize)
            .enumerate()
            .chain(std::iter::repeat_n(DimType::Out, in_dims as usize).enumerate());
        for (i, ty) in dims {
            let exp = term.exponent(ty, i as u32)?;
            if exp > 0 {
                let ty = if matches!(ty, DimType::Param) {
                    DimType::Param
                } else {
                    DimType::In
                };
                let name = self.space.get_dim_name(ty, i as u32)?;
                let symbol = symbol!(name);
                let exp = Atom::new_num(exp as i64);
                let atom = Atom::new_var(symbol).pow(exp);
                let atom = atom.to_rational_polynomial(&self.ring, &self.ring, None);
                poly = self.field.mul(&poly, &atom);
            }
        }
        Ok(poly)
    }

    fn quasi_poly_to_rational_poly(
        &self,
        qpoly: QuasiPolynomial<'a>,
    ) -> std::result::Result<Poly, barvinok::Error> {
        let mut poly = Atom::new_num(0).to_rational_polynomial(&self.ring, &self.ring, None);
        qpoly.foreach_term(|term| {
            let term_poly = self.term_to_rational_poly(term)?;
            poly = self.field.add(&poly, &term_poly);
            Ok(())
        })?;
        Ok(poly)
    }
}

pub(crate) fn ensure_set_name<'a>(mut set: Set<'a>) -> Result<Set<'a>> {
    let params = set.num_params()?;
    let dims = set.num_dims()?;
    for i in 0..params {
        if !set.has_dim_name(DimType::Param, i)? {
            set = set.set_dim_name(DimType::Param, i, &format!("p{i}"))?;
        }
    }
    for i in 0..dims {
        if !set.has_dim_name(DimType::Out, i)? {
            set = set.set_dim_name(DimType::Out, i, &format!("i{i}"))?;
        }
    }
    Ok(set)
}

pub(crate) fn ensure_map_domain_name<'a>(mut map: Map<'a>) -> Result<Map<'a>> {
    let params = map.dim(DimType::Param)?;
    let in_dims = map.dim(DimType::In)?;
    for i in 0..params {
        if !map.has_dim_name(DimType::Param, i)? {
            map = map.set_dim_name(DimType::Param, i, &format!("p{i}"))?;
        }
    }
    for i in 0..in_dims {
        if !map.has_dim_name(DimType::In, i)? {
            map = map.set_dim_name(DimType::In, i, &format!("i{i}"))?;
        }
    }
    Ok(map)
}

#[derive(Clone, Debug)]
struct ConvertedDistItem<'a> {
    value: Poly,
    portion: Poly,
    domain: Set<'a>,
}

impl<'a> ConvertedDistItem<'a> {
    fn evaluate(&self, point: Point<'a>) -> Result<(isize, f64)> {
        let value = evaluate_poly(&self.value, &point)? as isize;
        let portion = evaluate_poly(&self.portion, &point)?;
        Ok((value, portion))
    }
    fn add_to_dist(&self, dist: &mut AHashMap<isize, f64>) -> Result<()> {
        let mut points = Vec::new();
        self.domain.foreach_point(|point| {
            points.push(point);
            Ok(())
        })?;
        points.into_iter().try_for_each(|point| {
            let (value, portion) = self.evaluate(point)?;
            match dist.entry(value) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += portion;
                }
                Entry::Vacant(entry) => {
                    entry.insert(portion);
                }
            }
            Ok(())
        })
    }
}

fn evaluate_poly<'a>(poly: &Poly, point: &Point<'a>) -> Result<f64> {
    let variables = poly.get_variables();
    let name_map = variables
        .iter()
        .filter_map(|x| {
            x.to_id()
                .map(|x| (x.get_stripped_name().to_string(), Atom::new_var(x)))
        })
        .collect::<AHashMap<String, Atom>>();
    let space = point.get_space()?;
    let dims = space.get_dim(DimType::Out)?;
    let mut const_map = AHashMap::new();
    for i in 0..dims {
        let name = space.get_dim_name(DimType::Out, i)?;
        let Some(atom) = name_map.get(name).cloned() else {
            continue;
        };
        let val = point.get_coordinate_val(DimType::Out, i)?.to_f64();
        const_map.insert(atom, val);
    }
    let expr = poly.to_expression();
    expr.evaluate(|x| x.to_f64(), &const_map, &AHashMap::new())
        .map_err(|msg| anyhow::anyhow!("failed to evaluate polynomial: {msg}"))
}

fn convert_bounded_set<'a>(mut set: Set<'a>) -> Result<Set<'a>> {
    let mut num_params = set.num_params()?;
    let mut infinite_repeat_dim = None;
    for i in 0..num_params {
        let name = set.get_dim_name(DimType::Param, i)?;
        if name.starts_with("p") {
            return Err(anyhow::anyhow!(
                "all parameters must be instantiated for numerical analysis"
            ));
        }
        if name == "R" {
            infinite_repeat_dim = Some(i);
        }
    }
    if let Some(dim) = infinite_repeat_dim {
        set = set.remove_dims(DimType::Param, dim, 1)?;
        num_params -= 1;
    }
    // move all parameters to set space
    set = set.move_dims(DimType::Out, 0, DimType::Param, 0, num_params)?;
    if !set.is_bounded()? {
        return Err(anyhow::anyhow!("set {set:?} is not bounded"));
    }
    Ok(set)
}

pub fn get_distro<'a>(
    dist: &[DistItem<'a>],
    total: PiecewiseQuasiPolynomial<'a>,
    infinite_repeat: bool,
) -> Result<Box<[(isize, f64)]>> {
    let dist = convert_dist(dist, total, infinite_repeat)?;
    let mut result = AHashMap::new();
    for item in dist.iter() {
        item.add_to_dist(&mut result)?;
    }
    let mut vector = vec![(0, 0.0)];
    vector.extend(result.iter().map(|(k, v)| (*k, *v)));
    vector.sort_unstable_by_key(|a| a.0);
    Ok(vector.into_boxed_slice())
}

fn convert_dist<'a>(
    dist: &[DistItem<'a>],
    total: PiecewiseQuasiPolynomial<'a>,
    infinite_repeat: bool,
) -> Result<Box<[ConvertedDistItem<'a>]>> {
    let mut total_count = None;
    total.foreach_piece(|qpoly, _| {
        total_count.replace(qpoly);
        Ok(())
    })?;
    let total_count = total_count.ok_or_else(|| anyhow::anyhow!("no total count found"))?;
    let total_count_poly = convert_quasi_poly(total_count)?;
    let mut output = Vec::new();
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    for item in dist.iter() {
        let value = convert_quasi_poly(item.qpoly.clone())?;
        let mut res = Ok(());
        item.cardinality.foreach_piece(|qpoly, domain| {
            if res.is_err() {
                return Ok(());
            }
            let poly = convert_quasi_poly(qpoly.clone())?;
            // L'Hôpital's rule
            let portion = if infinite_repeat {
                tracing::debug!("applying L'Hôpital's rule for {poly}/{total_count_poly}");
                let poly_var = poly.get_variables().iter().position(|x| {
                    x.to_id()
                        .map(|x| x.get_stripped_name() == "R")
                        .unwrap_or_default()
                });
                if poly.is_zero() || poly_var.is_none() {
                    return Ok(());
                }
                let Some(total_var) = total_count_poly.get_variables().iter().position(|x| {
                    x.to_id()
                        .map(|x| x.get_stripped_name() == "R")
                        .unwrap_or_default()
                }) else {
                    res = Err(anyhow::anyhow!("no total var found"));
                    return Ok(());
                };
                let poly = poly.derivative(poly_var.unwrap());
                let total = total_count_poly.derivative(total_var);
                if total.is_zero() {
                    return Ok(());
                }
                field.div(&poly, &total)
            } else {
                field.div(&poly, &total_count_poly)
            };
            match convert_bounded_set(domain) {
                Ok(domain) => {
                    let item = ConvertedDistItem {
                        value: value.clone(),
                        portion,
                        domain,
                    };
                    output.push(item);
                }
                Err(e) => res = Err(e),
            }
            Ok(())
        })?;
        res?;
    }
    Ok(output.into_boxed_slice())
}

#[derive(Serialize)]
struct BarvinokResult {
    ri_values: Box<[String]>,
    symbol_ranges: Box<[String]>,
    counts: Box<[String]>,
    portions: Box<[String]>,
    total_count: String,
    distribution: Box<[(isize, f64)]>,
}

pub fn create_json_output<'a>(
    dist: &[DistItem<'a>],
    total: PiecewiseQuasiPolynomial<'a>,
    infinite_repeat: bool,
) -> Result<String> {
    let distribution = get_distro(dist, total.clone(), infinite_repeat).unwrap_or_default();
    let mut total_count = None;
    total.foreach_piece(|qpoly, _| {
        total_count.replace(qpoly);
        Ok(())
    })?;
    let total_count = total_count.ok_or_else(|| anyhow::anyhow!("no total count found"))?;
    let total_count_poly = convert_quasi_poly(total_count)?;
    let mut ri_values = Vec::new();
    let mut symbol_ranges = Vec::new();
    let mut counts = Vec::new();
    let mut portions = Vec::new();
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    for item in dist.iter() {
        let value = convert_quasi_poly(item.qpoly.clone())?;
        let value_str = format!("{}", value.to_expression().printer(PrintOptions::latex()));
        item.cardinality.foreach_piece(|qpoly, domain| {
            let poly = convert_quasi_poly(qpoly.clone())?;
            let count = format!("{}", poly.to_expression().printer(PrintOptions::latex()));
            let range = format!("{domain:?}");
            let range = range
                .split("{  : ")
                .nth(1)
                .unwrap_or_default()
                .split(" }")
                .next()
                .unwrap_or_default();
            // L'Hôpital's rule
            let portion = if infinite_repeat {
                tracing::debug!("applying L'Hôpital's rule for {poly}/{total_count_poly}");
                let poly_var = poly.get_variables().iter().position(|x| {
                    x.to_id()
                        .map(|x| x.get_stripped_name() == "R")
                        .unwrap_or_default()
                });
                if poly.is_zero() || poly_var.is_none() {
                    ri_values.push(value_str.clone());
                    symbol_ranges.push(range.to_string());
                    counts.push(count);
                    portions.push("0".to_string());
                    return Ok(());
                }
                let total_var = total_count_poly
                    .get_variables()
                    .iter()
                    .position(|x| {
                        x.to_id()
                            .map(|x| x.get_stripped_name() == "R")
                            .unwrap_or_default()
                    })
                    .unwrap();
                let poly = poly.derivative(poly_var.unwrap());
                let total = total_count_poly.derivative(total_var);
                if total.is_zero() {
                    ri_values.push(value_str.clone());
                    symbol_ranges.push(range.to_string());
                    counts.push(count);
                    portions.push("∞".to_string());
                    return Ok(());
                }
                field.div(&poly, &total)
            } else {
                field.div(&poly, &total_count_poly)
            };
            let portion_str = format!("{}", portion.to_expression().printer(PrintOptions::latex()));
            ri_values.push(value_str.clone());
            symbol_ranges.push(range.to_string());
            counts.push(count);
            portions.push(portion_str);
            Ok(())
        })?;
    }
    let ri_values = ri_values.into_boxed_slice();
    let symbol_ranges = symbol_ranges.into_boxed_slice();
    let counts = counts.into_boxed_slice();
    let portions = portions.into_boxed_slice();
    let total_count = format!(
        "{}",
        total_count_poly
            .to_expression()
            .printer(PrintOptions::latex())
    );
    let result = BarvinokResult {
        ri_values,
        symbol_ranges,
        counts,
        portions,
        distribution,
        total_count,
    };
    let json = serde_json::to_string(&result)
        .map_err(|e| anyhow::anyhow!("failed to serialize to JSON: {e}"))?;
    Ok(json)
}
