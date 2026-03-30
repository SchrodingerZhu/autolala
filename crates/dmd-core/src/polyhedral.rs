use crate::ast::{Block, Comparison, ComparisonOp, Expr, Program, Stmt};
use crate::error::{DmdError, DmdResult};
use crate::formula::{FormulaFormatter, format_domain};
use crate::parse_program;
use crate::semantics::{ArrayInfo, SemanticProgram, validate_program};
use barvinok::{
    Context, ContextRef, DimType,
    aff::Affine,
    constraint::Constraint,
    list::AffineList,
    local_space::LocalSpace,
    map::{BasicMap, Map},
    polynomial::{PiecewiseQuasiPolynomial, QuasiPolynomial},
    set::Set,
    space::Space,
    value::Value,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOptions {
    pub block_size: usize,
    pub num_sets: usize,
    pub max_operations: usize,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            block_size: 1,
            num_sets: 1,
            max_operations: 5_000_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionRegion {
    pub domain_plain: String,
    pub count_plain: String,
    pub count_latex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionEntry {
    pub value_plain: String,
    pub value_latex: String,
    pub regions: Vec<DistributionRegion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DmdTerm {
    pub domain_plain: String,
    pub multiplicity_plain: String,
    pub multiplicity_latex: String,
    pub reuse_distance_plain: String,
    pub reuse_distance_latex: String,
    pub term_plain: String,
    pub term_latex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub options: AnalysisOptions,
    pub total_accesses_plain: String,
    pub total_accesses_latex: String,
    pub warm_accesses_plain: String,
    pub warm_accesses_latex: String,
    pub compulsory_accesses_plain: String,
    pub compulsory_accesses_latex: String,
    pub timestamp_space: String,
    pub access_map: String,
    pub ri_distribution: Vec<DistributionEntry>,
    pub rd_distribution: Vec<DistributionEntry>,
    pub dmd_terms: Vec<DmdTerm>,
    pub dmd_formula_plain: String,
    pub dmd_formula_latex: String,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
struct DistributionItem<'a> {
    value: QuasiPolynomial<'a>,
    cardinality: PiecewiseQuasiPolynomial<'a>,
}

#[derive(Clone, Debug)]
struct DistributionPiece<'a> {
    domain: Set<'a>,
    value: QuasiPolynomial<'a>,
}

#[derive(Clone, Debug)]
struct LoopBinding {
    name: String,
    lower: Expr,
    step: i64,
    index: usize,
    env_prefix_len: usize,
}

#[derive(Clone)]
struct Lowering<'model> {
    model: &'model SemanticProgram,
    array_map: HashMap<&'model str, &'model ArrayInfo>,
}

impl<'model> Lowering<'model> {
    fn new(model: &'model SemanticProgram) -> Self {
        let array_map = model
            .arrays
            .iter()
            .map(|array| (array.name.as_str(), array))
            .collect::<HashMap<_, _>>();
        Self { model, array_map }
    }

    fn parameter_count(&self) -> usize {
        self.model.params.len()
    }

    fn get_timestamp_space<'ctx>(&self, ctx: ContextRef<'ctx>) -> DmdResult<Set<'ctx>> {
        let mut loop_env = Vec::new();
        let set = self.timestamp_block(ctx, &self.model.body, 0, &mut loop_env)?;
        self.ensure_set_names(set)
    }

    fn timestamp_block<'ctx>(
        &self,
        ctx: ContextRef<'ctx>,
        block: &Block,
        depth: usize,
        loop_env: &mut Vec<LoopBinding>,
    ) -> DmdResult<Set<'ctx>> {
        if block.statements.is_empty() {
            let space = Space::set(ctx, self.parameter_count() as u32, depth as u32)?;
            return Ok(Set::empty(space)?);
        }

        let mut sub_sets = block
            .statements
            .iter()
            .map(|stmt| self.timestamp_stmt(ctx, stmt, depth + 1, loop_env))
            .collect::<DmdResult<Vec<_>>>()?;
        let longest = sub_sets
            .iter()
            .max_by_key(|set| set.n_dim().unwrap_or_default())
            .cloned()
            .ok_or_else(|| DmdError::analysis("failed to build timestamp subspaces"))?;
        let space = longest.get_space()?;
        align_sets(longest, depth, sub_sets.iter_mut(), true)?;
        Ok(sub_sets
            .into_iter()
            .try_fold(Set::empty(space)?, |acc, set| -> DmdResult<_> { Ok(acc.union(set)?) })?
            .set_dim_name(
                DimType::Out,
                depth as u32,
                &format!("t{}", depth.saturating_sub(loop_env.len())),
            )?)
    }

    fn timestamp_stmt<'ctx>(
        &self,
        ctx: ContextRef<'ctx>,
        stmt: &Stmt,
        depth: usize,
        loop_env: &mut Vec<LoopBinding>,
    ) -> DmdResult<Set<'ctx>> {
        match stmt {
            Stmt::For(for_loop) => {
                let env_prefix_len = loop_env.len();
                loop_env.push(LoopBinding {
                    name: for_loop.var.clone(),
                    lower: for_loop.lower.clone(),
                    step: for_loop.step,
                    index: depth,
                    env_prefix_len,
                });
                let set = self.timestamp_block(ctx, &for_loop.body, depth + 1, loop_env)?;
                loop_env.pop();

                let space = set.get_space()?;
                let local_space = LocalSpace::try_from(space.clone())?;
                let lower = self.convert_expr(local_space.clone(), &for_loop.lower, loop_env, env_prefix_len)?;
                let upper = self.convert_expr(local_space.clone(), &for_loop.upper, loop_env, env_prefix_len)?;
                let step = Affine::val_on_domain(
                    local_space.clone(),
                    Value::int_from_si(ctx, for_loop.step)?,
                )?;
                let trip = upper.checked_sub(lower)?.checked_div(step)?.ceil()?;
                let ge_zero = Constraint::new_inequality(local_space.clone())?
                    .set_coefficient_si(DimType::Out, depth as i32, 1)?;
                let ub = trip
                    .checked_sub(Affine::var_on_domain(
                        local_space.clone(),
                        DimType::Out,
                        depth as u32,
                    )?)?
                    .checked_sub(Affine::val_on_domain(
                        local_space.clone(),
                        Value::int_from_si(ctx, 1)?,
                    )?)?;
                Ok(set
                    .add_constraint(ge_zero)?
                    .add_constraint(Constraint::new_inequality_from_affine(ub))?
                    .set_dim_name(DimType::Out, depth as u32, &for_loop.var)?)
            }
            Stmt::If(if_stmt) => {
                let then_set = self.timestamp_block(ctx, &if_stmt.then_branch, depth, loop_env)?;
                let else_set = if let Some(else_branch) = &if_stmt.else_branch {
                    self.timestamp_block(ctx, else_branch, depth, loop_env)?
                } else {
                    Set::empty(then_set.get_space()?)?
                };

                let longest = if then_set.n_dim()? >= else_set.n_dim()? {
                    then_set.clone()
                } else {
                    else_set.clone()
                };
                let mut subsets = [then_set, else_set];
                let dom_space = longest.get_space()?;
                align_sets(longest, depth, subsets.iter_mut(), false)?;
                let cond = self.condition_set(dom_space.clone(), &if_stmt.conditions, loop_env, loop_env.len())?;
                let complement = cond.clone().complement()?;
                let [then_set, else_set] = subsets;
                Ok(then_set
                    .intersect(cond)?
                    .union(else_set.intersect(complement)?)?)
            }
            Stmt::Access(_) => {
                let space = Space::set(ctx, self.parameter_count() as u32, depth as u32)?;
                Ok(Set::universe(space)?)
            }
        }
    }

    fn get_access_map<'ctx>(&self, ctx: ContextRef<'ctx>, options: &AnalysisOptions) -> DmdResult<Map<'ctx>> {
        let mut loop_env = Vec::new();
        let max_rank = self.model.max_access_rank();
        let map = self.access_block(ctx, &self.model.body, 0, &mut loop_env, max_rank, options)?;
        self.ensure_map_names(map)
    }

    fn access_block<'ctx>(
        &self,
        ctx: ContextRef<'ctx>,
        block: &Block,
        depth: usize,
        loop_env: &mut Vec<LoopBinding>,
        max_rank: usize,
        options: &AnalysisOptions,
    ) -> DmdResult<Map<'ctx>> {
        if block.statements.is_empty() {
            let range_dims = (max_rank + 1 + usize::from(options.num_sets > 1)) as u32;
            let space = Space::new(ctx, self.parameter_count() as u32, depth as u32, range_dims)?;
            return Ok(Map::empty(space)?);
        }

        let mut sub_maps = block
            .statements
            .iter()
            .map(|stmt| self.access_stmt(ctx, stmt, depth + 1, loop_env, max_rank, options))
            .collect::<DmdResult<Vec<_>>>()?;
        let longest = sub_maps
            .iter()
            .max_by_key(|map| map.dim(DimType::In).unwrap_or_default())
            .cloned()
            .ok_or_else(|| DmdError::analysis("failed to build access submaps"))?;
        let space = longest.get_space()?;
        align_maps(longest, depth, sub_maps.iter_mut(), true)?;
        Ok(sub_maps
            .into_iter()
            .try_fold(Map::empty(space)?, |acc, map| -> DmdResult<_> { Ok(acc.union(map)?) })?
            .set_dim_name(
                DimType::In,
                depth as u32,
                &format!("t{}", depth.saturating_sub(loop_env.len())),
            )?)
    }

    fn access_stmt<'ctx>(
        &self,
        ctx: ContextRef<'ctx>,
        stmt: &Stmt,
        depth: usize,
        loop_env: &mut Vec<LoopBinding>,
        max_rank: usize,
        options: &AnalysisOptions,
    ) -> DmdResult<Map<'ctx>> {
        match stmt {
            Stmt::For(for_loop) => {
                let env_prefix_len = loop_env.len();
                loop_env.push(LoopBinding {
                    name: for_loop.var.clone(),
                    lower: for_loop.lower.clone(),
                    step: for_loop.step,
                    index: depth,
                    env_prefix_len,
                });
                let map = self.access_block(ctx, &for_loop.body, depth + 1, loop_env, max_rank, options)?;
                loop_env.pop();
                Ok(map.set_dim_name(DimType::In, depth as u32, &for_loop.var)?)
            }
            Stmt::If(if_stmt) => {
                let then_map = self.access_block(ctx, &if_stmt.then_branch, depth, loop_env, max_rank, options)?;
                let else_map = if let Some(else_branch) = &if_stmt.else_branch {
                    self.access_block(ctx, else_branch, depth, loop_env, max_rank, options)?
                } else {
                    Map::empty(then_map.get_space()?)?
                };

                let longest = if then_map.dim(DimType::In)? >= else_map.dim(DimType::In)? {
                    then_map.clone()
                } else {
                    else_map.clone()
                };
                let mut submaps = [then_map, else_map];
                let dom_space = longest.clone().domain()?.get_space()?;
                align_maps(longest, depth, submaps.iter_mut(), false)?;
                let cond = self.condition_set(dom_space, &if_stmt.conditions, loop_env, loop_env.len())?;
                let complement = cond.clone().complement()?;
                let [then_map, else_map] = submaps;
                Ok(then_map
                    .intersect_domain(cond)?
                    .union(else_map.intersect_domain(complement)?)?)
            }
            Stmt::Access(access) => {
                let domain_space = Space::set(ctx, self.parameter_count() as u32, depth as u32)?;
                let local_space = LocalSpace::try_from(domain_space.clone())?;
                let mut affines = AffineList::new(ctx, max_rank + 1 + usize::from(options.num_sets > 1));
                let array = self
                    .array_map
                    .get(access.array.as_str())
                    .ok_or_else(|| DmdError::analysis(format!("missing array info for `{}`", access.array)))?;
                affines.push(Affine::val_on_domain_space(
                    domain_space.clone(),
                    Value::int_from_si(ctx, array.id as i64)?,
                )?);
                for _ in 0..max_rank.saturating_sub(access.indices.len()) {
                    affines.push(Affine::val_on_domain_space(
                        domain_space.clone(),
                        Value::int_from_si(ctx, 0)?,
                    )?);
                }

                for (index, expr) in access.indices.iter().enumerate() {
                    let mut affine = self.convert_expr(local_space.clone(), expr, loop_env, loop_env.len())?;
                    if options.block_size > 1 && index + 1 == access.indices.len() {
                        affine = affine
                            .checked_div(Affine::val_on_domain(
                                local_space.clone(),
                                Value::int_from_si(ctx, options.block_size as i64)?,
                            )?)?
                            .floor()?;
                    }
                    if options.num_sets > 1 && index + 1 == access.indices.len() {
                        let set_tag = affine
                            .clone()
                            .modulo(Value::int_from_si(ctx, options.num_sets as i64)?)?;
                        affines.push(set_tag);
                    }
                    affines.push(affine);
                }

                Ok(BasicMap::from_affine_list(domain_space, affines)?.try_into()?)
            }
        }
    }

    fn condition_set<'ctx>(
        &self,
        space: Space<'ctx>,
        conditions: &[Comparison],
        loop_env: &[LoopBinding],
        env_limit: usize,
    ) -> DmdResult<Set<'ctx>> {
        let local_space = LocalSpace::try_from(space.clone())?;
        let mut set = Set::universe(space)?;
        for comparison in conditions {
            let lhs = self.convert_expr(local_space.clone(), &comparison.lhs, loop_env, env_limit)?;
            let rhs = self.convert_expr(local_space.clone(), &comparison.rhs, loop_env, env_limit)?;
            let one = Affine::val_on_domain(
                local_space.clone(),
                Value::int_from_si(local_space.context_ref(), 1)?,
            )?;
            let constraint = match comparison.op {
                ComparisonOp::Lt => Constraint::new_inequality_from_affine(rhs.checked_sub(lhs)?.checked_sub(one)?),
                ComparisonOp::Le => Constraint::new_inequality_from_affine(rhs.checked_sub(lhs)?),
                ComparisonOp::Eq => Constraint::new_equality_from_affine(lhs.checked_sub(rhs)?),
                ComparisonOp::Ge => Constraint::new_inequality_from_affine(lhs.checked_sub(rhs)?),
                ComparisonOp::Gt => Constraint::new_inequality_from_affine(lhs.checked_sub(rhs)?.checked_sub(one)?),
            };
            set = set.add_constraint(constraint)?;
        }
        Ok(set)
    }

    fn convert_expr<'ctx>(
        &self,
        local_space: LocalSpace<'ctx>,
        expr: &Expr,
        loop_env: &[LoopBinding],
        env_limit: usize,
    ) -> DmdResult<Affine<'ctx>> {
        match expr {
            Expr::Int(value) => Ok(Affine::val_on_domain(
                local_space.clone(),
                Value::int_from_si(local_space.context_ref(), *value)?,
            )?),
            Expr::Var(name) => {
                if let Some(index) = self.model.params.iter().position(|param| param == name) {
                    return Ok(Affine::var_on_domain(local_space, DimType::Param, index as u32)?);
                }

                let (_, binding) = loop_env
                    .iter()
                    .take(env_limit)
                    .enumerate()
                    .find(|(_, binding)| binding.name == *name)
                    .ok_or_else(|| DmdError::analysis(format!("unknown variable `{name}` during lowering")))?;
                let ordinal = Affine::var_on_domain(local_space.clone(), DimType::Out, binding.index as u32)?;
                let step = Affine::val_on_domain(
                    local_space.clone(),
                    Value::int_from_si(local_space.context_ref(), binding.step)?,
                )?;
                let lower =
                    self.convert_expr(local_space.clone(), &binding.lower, loop_env, binding.env_prefix_len)?;
                Ok(ordinal.checked_mul(step)?.checked_add(lower)?)
            }
            Expr::Add(lhs, rhs) => Ok(self
                .convert_expr(local_space.clone(), lhs, loop_env, env_limit)?
                .checked_add(self.convert_expr(local_space, rhs, loop_env, env_limit)?)?),
            Expr::Sub(lhs, rhs) => Ok(self
                .convert_expr(local_space.clone(), lhs, loop_env, env_limit)?
                .checked_sub(self.convert_expr(local_space, rhs, loop_env, env_limit)?)?),
            Expr::Mul(lhs, rhs) => Ok(self
                .convert_expr(local_space.clone(), lhs, loop_env, env_limit)?
                .checked_mul(self.convert_expr(local_space, rhs, loop_env, env_limit)?)?),
            Expr::FloorDiv(lhs, rhs) => Ok(self
                .convert_expr(local_space.clone(), lhs, loop_env, env_limit)?
                .checked_div(self.convert_expr(local_space, rhs, loop_env, env_limit)?)?
                .floor()?),
            Expr::Neg(inner) => Ok(Affine::val_on_domain(
                local_space.clone(),
                Value::int_from_si(local_space.context_ref(), -1)?,
            )?
            .checked_mul(self.convert_expr(local_space, inner, loop_env, env_limit)?)?),
        }
    }

    fn ensure_set_names<'ctx>(&self, mut set: Set<'ctx>) -> DmdResult<Set<'ctx>> {
        for (index, param) in self.model.params.iter().enumerate() {
            set = set.set_dim_name(DimType::Param, index as u32, param)?;
        }
        let dims = set.n_dim()?;
        for index in 0..dims {
            if !set.has_dim_name(DimType::Out, index)? {
                set = set.set_dim_name(DimType::Out, index, &format!("d{index}"))?;
            }
        }
        Ok(set)
    }

    fn ensure_map_names<'ctx>(&self, mut map: Map<'ctx>) -> DmdResult<Map<'ctx>> {
        for (index, param) in self.model.params.iter().enumerate() {
            map = map.set_dim_name(DimType::Param, index as u32, param)?;
        }
        let dims = map.dim(DimType::In)?;
        for index in 0..dims {
            if !map.has_dim_name(DimType::In, index)? {
                map = map.set_dim_name(DimType::In, index, &format!("d{index}"))?;
            }
        }
        Ok(map)
    }
}

pub fn analyze_source(source: &str, options: AnalysisOptions) -> DmdResult<AnalysisReport> {
    let program = parse_program(source)?;
    analyze_program(&program, options)
}

pub fn analyze_program(program: &Program, options: AnalysisOptions) -> DmdResult<AnalysisReport> {
    if options.block_size == 0 {
        return Err(DmdError::semantic("block size must be at least one"));
    }
    if options.num_sets == 0 {
        return Err(DmdError::semantic("num_sets must be at least one"));
    }

    let model = validate_program(program.clone())?;
    let lowering = Lowering::new(&model);
    let context = Context::new();
    context.scope(|ctx| analyze_with_context(ctx, &lowering, options))
}

fn analyze_with_context<'ctx>(
    ctx: ContextRef<'ctx>,
    lowering: &Lowering<'_>,
    options: AnalysisOptions,
) -> DmdResult<AnalysisReport> {
    ctx.set_max_operations(options.max_operations);
    let timestamp_space = lowering.get_timestamp_space(ctx)?;
    let access_map = lowering.get_access_map(ctx, &options)?.intersect_domain(timestamp_space.clone())?;

    let space = timestamp_space.get_space()?;
    let lt = Map::lex_lt(space.clone())?
        .intersect_domain(timestamp_space.clone())?
        .intersect_range(timestamp_space.clone())?;
    let le = Map::lex_le(space.clone())?
        .intersect_domain(timestamp_space.clone())?
        .intersect_range(timestamp_space.clone())?;
    let access_rev = access_map.clone().reverse()?;
    let same_element = access_map.clone().apply_range(access_rev)?;
    let immediate_next = same_element.intersect(lt.clone())?.lexmin()?;
    let immediate_prev = immediate_next.reverse()?;
    let interval = immediate_prev.apply_range(lt)?;
    let ri_relation = interval.intersect(le.reverse()?)?;
    let ri_values = ri_relation.clone().cardinality()?;
    let rd_relation = ri_relation.apply_range(access_map.clone())?;
    let rd_values = rd_relation.cardinality()?;
    let total_accesses = timestamp_space.clone().cardinality()?;

    let ri_distribution = DistributionProcessor::new(ri_values).collect()?;
    let rd_distribution = DistributionProcessor::new(rd_values).collect()?;
    let total_rendered = render_piecewise(total_accesses.clone())?;
    let ri_rendered = render_distribution(&ri_distribution)?;
    let rd_rendered = render_distribution(&rd_distribution)?;
    let warm_plain = join_add(rd_rendered.count_plain_terms.iter().cloned());
    let warm_latex = join_add(rd_rendered.count_latex_terms.iter().cloned());
    let compulsory_plain = format!("({}) - ({})", total_rendered.0, warm_plain);
    let compulsory_latex = format!("({}) - ({})", total_rendered.1, warm_latex);
    let mut dmd_terms = Vec::new();
    let mut dmd_plain_terms = vec![compulsory_plain.clone()];
    let mut dmd_latex_terms = vec![compulsory_latex.clone()];

    for entry in &rd_rendered.entries {
        for region in &entry.regions {
            let term_plain = format!("({}) * sqrt({})", region.count_plain, entry.value_plain);
            let term_latex = format!("{} \\cdot \\sqrt{{{}}}", region.count_latex, entry.value_latex);
            dmd_plain_terms.push(term_plain.clone());
            dmd_latex_terms.push(term_latex.clone());
            dmd_terms.push(DmdTerm {
                domain_plain: region.domain_plain.clone(),
                multiplicity_plain: region.count_plain.clone(),
                multiplicity_latex: region.count_latex.clone(),
                reuse_distance_plain: entry.value_plain.clone(),
                reuse_distance_latex: entry.value_latex.clone(),
                term_plain,
                term_latex,
            });
        }
    }

    Ok(AnalysisReport {
        options,
        total_accesses_plain: total_rendered.0,
        total_accesses_latex: total_rendered.1,
        warm_accesses_plain: warm_plain,
        warm_accesses_latex: warm_latex,
        compulsory_accesses_plain: compulsory_plain,
        compulsory_accesses_latex: compulsory_latex,
        timestamp_space: format!("{timestamp_space:?}"),
        access_map: format!("{access_map:?}"),
        ri_distribution: ri_rendered.entries,
        rd_distribution: rd_rendered.entries,
        dmd_terms,
        dmd_formula_plain: join_add(dmd_plain_terms),
        dmd_formula_latex: join_add(dmd_latex_terms),
        notes: vec![
            "Reuse intervals are computed from the symbolic access relation, following the autolala-style Barvinok construction.".to_string(),
            "Reuse distance is derived from the cardinality of the access-image inside each reuse interval window, without Denning recursion.".to_string(),
            "The compulsory-access term is modeled separately as total accesses minus warm reuses.".to_string(),
        ],
    })
}

struct RenderedDistribution {
    entries: Vec<DistributionEntry>,
    count_plain_terms: Vec<String>,
    count_latex_terms: Vec<String>,
}

fn render_distribution(items: &[DistributionItem<'_>]) -> DmdResult<RenderedDistribution> {
    let mut entries = Vec::new();
    let mut count_plain_terms = Vec::new();
    let mut count_latex_terms = Vec::new();

    for item in items {
        let value_formatter = FormulaFormatter::new(item.value.get_space()?);
        let value_expr = value_formatter.quasi_polynomial(item.value.clone())?;
        let mut regions = Vec::new();
        item.cardinality.foreach_piece(|qpoly, domain| {
            let formatter = FormulaFormatter::new(qpoly.get_space()?);
            let count_expr = formatter.quasi_polynomial(qpoly)?;
            count_plain_terms.push(count_expr.to_plain());
            count_latex_terms.push(count_expr.to_latex());
            regions.push(DistributionRegion {
                domain_plain: format_domain(&domain),
                count_plain: count_expr.to_plain(),
                count_latex: count_expr.to_latex(),
            });
            Ok(())
        })?;
        entries.push(DistributionEntry {
            value_plain: value_expr.to_plain(),
            value_latex: value_expr.to_latex(),
            regions,
        });
    }

    Ok(RenderedDistribution {
        entries,
        count_plain_terms,
        count_latex_terms,
    })
}

fn render_piecewise(pw: PiecewiseQuasiPolynomial<'_>) -> DmdResult<(String, String)> {
    let mut plain = Vec::new();
    let mut latex = Vec::new();
    pw.foreach_piece(|qpoly, domain| {
        let formatter = FormulaFormatter::new(qpoly.get_space()?);
        let expr = formatter.quasi_polynomial(qpoly)?;
        if plain.is_empty() && latex.is_empty() {
            plain.push(expr.to_plain());
            latex.push(expr.to_latex());
        } else {
            let region = format_domain(&domain);
            plain.push(format!("[{region}] => {}", expr.to_plain()));
            latex.push(format!("\\left[{}\\right] \\Rightarrow {}", region, expr.to_latex()));
        }
        Ok(())
    })?;
    Ok((join_add(plain), join_add(latex)))
}

struct DistributionProcessor<'a> {
    qpoly: PiecewiseQuasiPolynomial<'a>,
}

impl<'a> DistributionProcessor<'a> {
    fn new(qpoly: PiecewiseQuasiPolynomial<'a>) -> Self {
        Self { qpoly }
    }

    fn collect(&self) -> DmdResult<Vec<DistributionItem<'a>>> {
        let mut pieces = self.qpoly_pieces()?;
        for piece in &mut pieces {
            let involved = involved_input_dims(&piece.value)?;
            let mut domain = piece.domain.clone();
            for (shift, dim) in involved.iter().enumerate() {
                let num_params = domain.dim(DimType::Param)?;
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

        pieces
            .into_iter()
            .map(|piece| {
                Ok(DistributionItem {
                    value: piece.value,
                    cardinality: piece.domain.cardinality()?,
                })
            })
            .collect()
    }

    fn qpoly_pieces(&self) -> DmdResult<Vec<DistributionPiece<'a>>> {
        let mut pieces = Vec::new();
        self.qpoly.foreach_piece(|qpoly, domain| {
            if let Some(existing) = pieces
                .iter_mut()
                .find(|piece: &&mut DistributionPiece<'a>| qpoly.plain_is_equal(&piece.value).unwrap_or(false))
            {
                existing.domain = existing.domain.clone().union(domain)?;
            } else {
                pieces.push(DistributionPiece { domain, value: qpoly });
            }
            Ok(())
        })?;
        Ok(pieces)
    }
}

fn involved_input_dims(qpoly: &QuasiPolynomial<'_>) -> Result<Vec<u32>, barvinok::Error> {
    let dims = qpoly.get_dim(DimType::In)?;
    let mut involved = Vec::new();
    for index in 0..dims {
        if qpoly.involves_dims(DimType::In, index, 1)? {
            involved.push(index);
        }
    }
    Ok(involved)
}

fn align_sets<'a, 'b: 'a>(
    longest: Set<'b>,
    depth: usize,
    iter: impl Iterator<Item = &'a mut Set<'b>>,
    add_dim_constraint: bool,
) -> DmdResult<()> {
    let space = longest.get_space()?;
    let longest_dims = longest.n_dim()?;
    let local_space = LocalSpace::try_from(space.clone())?;
    for (index, set) in iter.enumerate() {
        let dims = set.n_dim()?;
        let mut aligned = set.clone().insert_dims(DimType::Out, dims, longest_dims - dims)?;
        for dim in dims..longest_dims {
            aligned = aligned.add_constraint(
                Constraint::new_equality(local_space.clone())?
                    .set_coefficient_si(DimType::Out, dim as i32, 1)?,
            )?;
        }
        if add_dim_constraint {
            aligned = aligned.add_constraint(
                Constraint::new_equality(local_space.clone())?
                    .set_coefficient_si(DimType::Out, depth as i32, 1)?
                    .set_constant_si(-(index as i32))?,
            )?;
        }
        *set = aligned;
    }
    Ok(())
}

fn align_maps<'a, 'b: 'a>(
    longest: Map<'b>,
    depth: usize,
    iter: impl Iterator<Item = &'a mut Map<'b>>,
    add_dim_constraint: bool,
) -> DmdResult<()> {
    let space = longest.get_space()?;
    let local_space = LocalSpace::try_from(space.clone())?;
    let longest_dims = longest.dim(DimType::In)?;
    for (index, map) in iter.enumerate() {
        let dims = map.dim(DimType::In)?;
        let mut aligned = map.clone().add_dims(DimType::In, longest_dims - dims)?;
        for dim in dims..longest_dims {
            aligned = aligned.add_constraint(
                Constraint::new_equality(local_space.clone())?
                    .set_coefficient_si(DimType::In, dim as i32, 1)?,
            )?;
        }
        if add_dim_constraint {
            aligned = aligned.add_constraint(
                Constraint::new_equality(local_space.clone())?
                    .set_coefficient_si(DimType::In, depth as i32, 1)?
                    .set_constant_si(-(index as i32))?,
            )?;
        }
        *map = aligned;
    }
    Ok(())
}

fn join_add(items: impl IntoIterator<Item = String>) -> String {
    let values = items
        .into_iter()
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        "0".to_string()
    } else {
        values.join(" + ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = r#"
params N;
array A[N];

for i in 0 .. N {
    read A[0];
}
"#;

    #[test]
    fn repeated_single_access_has_unit_rd() {
        let report = analyze_source(SIMPLE, AnalysisOptions::default()).expect("analysis should succeed");
        assert!(!report.rd_distribution.is_empty());
        assert!(report
            .rd_distribution
            .iter()
            .any(|entry| entry.value_plain == "1"));
        assert!(report.warm_accesses_plain.contains("N"));
    }
}
