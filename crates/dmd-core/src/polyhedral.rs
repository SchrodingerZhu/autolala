use crate::ast::{Block, Comparison, ComparisonOp, Expr, Program, Stmt};
use crate::error::{DmdError, DmdResult};
use crate::formula::{FormulaExpr, FormulaFormatter, format_domain};
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
    set::{BasicSet, Set},
    space::Space,
    value::Value,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{LazyLock, Mutex},
};

static BARVINOK_ANALYSIS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOptions {
    pub block_size: usize,
    pub num_sets: usize,
    pub max_operations: usize,
    pub approximation_method: ApproximationMethod,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApproximationMethod {
    Scale,
}

impl ApproximationMethod {
    fn as_barvinok_arg(self) -> &'static str {
        match self {
            Self::Scale => "--approximation-method=scale",
        }
    }
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            block_size: 1,
            num_sets: 1,
            max_operations: 5_000_000,
            approximation_method: ApproximationMethod::Scale,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParallelModelKind {
    NegativeBinomial,
    Racetrack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelCriLaw {
    pub range_plain: String,
    pub range_latex: String,
    pub cri_plain: String,
    pub cri_latex: String,
    pub probability_plain: String,
    pub probability_latex: String,
    pub count_plain: String,
    pub count_latex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelCriEntry {
    pub domain_plain: String,
    pub multiplicity_plain: String,
    pub multiplicity_latex: String,
    pub source_ri_plain: String,
    pub source_ri_latex: String,
    pub model_kind: ParallelModelKind,
    pub law: ParallelCriLaw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelAnalysis {
    pub thread_count: i64,
    pub schedule: String,
    pub cri_entries: Vec<ParallelCriEntry>,
    pub notes: Vec<String>,
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
    pub parallel: Option<ParallelAnalysis>,
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
            .try_fold(Set::empty(space)?, |acc, set| -> DmdResult<_> {
                Ok(acc.union(set)?)
            })?
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
                let lower = self.convert_expr(
                    local_space.clone(),
                    &for_loop.lower,
                    loop_env,
                    env_prefix_len,
                )?;
                let upper = self.convert_expr(
                    local_space.clone(),
                    &for_loop.upper,
                    loop_env,
                    env_prefix_len,
                )?;
                let step = Affine::val_on_domain(
                    local_space.clone(),
                    Value::int_from_si(ctx, for_loop.step)?,
                )?;
                let trip = upper.checked_sub(lower)?.checked_div(step)?.ceil()?;
                let ge_zero = Constraint::new_inequality(local_space.clone())?.set_coefficient_si(
                    DimType::Out,
                    depth as i32,
                    1,
                )?;
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
                let cond = self.condition_set(
                    dom_space.clone(),
                    &if_stmt.conditions,
                    loop_env,
                    loop_env.len(),
                )?;
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

    fn get_access_map<'ctx>(
        &self,
        ctx: ContextRef<'ctx>,
        options: &AnalysisOptions,
    ) -> DmdResult<Map<'ctx>> {
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
            .try_fold(Map::empty(space)?, |acc, map| -> DmdResult<_> {
                Ok(acc.union(map)?)
            })?
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
                let map =
                    self.access_block(ctx, &for_loop.body, depth + 1, loop_env, max_rank, options)?;
                loop_env.pop();
                Ok(map.set_dim_name(DimType::In, depth as u32, &for_loop.var)?)
            }
            Stmt::If(if_stmt) => {
                let then_map = self.access_block(
                    ctx,
                    &if_stmt.then_branch,
                    depth,
                    loop_env,
                    max_rank,
                    options,
                )?;
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
                let cond =
                    self.condition_set(dom_space, &if_stmt.conditions, loop_env, loop_env.len())?;
                let complement = cond.clone().complement()?;
                let [then_map, else_map] = submaps;
                Ok(then_map
                    .intersect_domain(cond)?
                    .union(else_map.intersect_domain(complement)?)?)
            }
            Stmt::Access(access) => {
                let domain_space = Space::set(ctx, self.parameter_count() as u32, depth as u32)?;
                let local_space = LocalSpace::try_from(domain_space.clone())?;
                let mut affines =
                    AffineList::new(ctx, max_rank + 1 + usize::from(options.num_sets > 1));
                let array = self.array_map.get(access.array.as_str()).ok_or_else(|| {
                    DmdError::analysis(format!("missing array info for `{}`", access.array))
                })?;
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
                    let mut affine =
                        self.convert_expr(local_space.clone(), expr, loop_env, loop_env.len())?;
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
            let lhs =
                self.convert_expr(local_space.clone(), &comparison.lhs, loop_env, env_limit)?;
            let rhs =
                self.convert_expr(local_space.clone(), &comparison.rhs, loop_env, env_limit)?;
            let one = Affine::val_on_domain(
                local_space.clone(),
                Value::int_from_si(local_space.context_ref(), 1)?,
            )?;
            let constraint = match comparison.op {
                ComparisonOp::Lt => {
                    Constraint::new_inequality_from_affine(rhs.checked_sub(lhs)?.checked_sub(one)?)
                }
                ComparisonOp::Le => Constraint::new_inequality_from_affine(rhs.checked_sub(lhs)?),
                ComparisonOp::Eq => Constraint::new_equality_from_affine(lhs.checked_sub(rhs)?),
                ComparisonOp::Ge => Constraint::new_inequality_from_affine(lhs.checked_sub(rhs)?),
                ComparisonOp::Gt => {
                    Constraint::new_inequality_from_affine(lhs.checked_sub(rhs)?.checked_sub(one)?)
                }
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
                    return Ok(Affine::var_on_domain(
                        local_space,
                        DimType::Param,
                        index as u32,
                    )?);
                }

                let (_, binding) = loop_env
                    .iter()
                    .take(env_limit)
                    .enumerate()
                    .find(|(_, binding)| binding.name == *name)
                    .ok_or_else(|| {
                        DmdError::analysis(format!("unknown variable `{name}` during lowering"))
                    })?;
                let ordinal =
                    Affine::var_on_domain(local_space.clone(), DimType::Out, binding.index as u32)?;
                let step = Affine::val_on_domain(
                    local_space.clone(),
                    Value::int_from_si(local_space.context_ref(), binding.step)?,
                )?;
                let lower = self.convert_expr(
                    local_space.clone(),
                    &binding.lower,
                    loop_env,
                    binding.env_prefix_len,
                )?;
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
            .checked_mul(self.convert_expr(
                local_space,
                inner,
                loop_env,
                env_limit,
            )?)?),
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

    // `Context::from_args` is not stable under parallel in-process initialization,
    // so keep Barvinok/ISL execution serialized within this process.
    let _analysis_guard = BARVINOK_ANALYSIS_LOCK
        .lock()
        .map_err(|_| DmdError::analysis("barvinok analysis lock poisoned"))?;

    let model = validate_program(program.clone())?;
    let lowering = Lowering::new(&model);
    let context = unsafe {
        Context::from_args([options.approximation_method.as_barvinok_arg()].into_iter())
    }?;
    context.scope(|ctx| analyze_with_context(ctx, &lowering, options))
}

fn analyze_with_context<'ctx>(
    ctx: ContextRef<'ctx>,
    lowering: &Lowering<'_>,
    options: AnalysisOptions,
) -> DmdResult<AnalysisReport> {
    ctx.set_max_operations(options.max_operations);
    if lowering.model.parallel_loop.is_some() {
        return analyze_parallel_with_context(ctx, lowering.model, options);
    }

    let timestamp_space = lowering.get_timestamp_space(ctx)?;
    let access_map = lowering
        .get_access_map(ctx, &options)?
        .intersect_domain(timestamp_space.clone())?;

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
    let ri_distribution = DistributionProcessor::new(ri_values).collect()?;
    let ri_rendered = render_distribution(&ri_distribution)?;
    let (ri_entries, filtered_ri_region_count) = filter_rendered_entries(ri_rendered.entries)?;
    if let Some(parallel) = build_parallel_analysis(lowering.model, &ri_entries)? {
        let approximation_arg = options.approximation_method.as_barvinok_arg().to_string();
        let mut notes = vec![
            format!(
                "Barvinok runs inside a context initialized with `{}`.",
                approximation_arg
            ),
            "Reuse intervals are computed from the symbolic access relation, following the autolala-style Barvinok construction.".to_string(),
            "Parallel loop mode stops after RI collection and CRI table generation; sequential RD/DMD statistics are skipped to keep the analysis fast.".to_string(),
        ];
        if filtered_ri_region_count > 0 {
            notes.push(format!(
                "Filtered {filtered_ri_region_count} RI region(s) whose domains add equality or upper-bound constraints on named scaling dimensions."
            ));
        }
        return Ok(AnalysisReport {
            options,
            total_accesses_plain: String::new(),
            total_accesses_latex: String::new(),
            warm_accesses_plain: String::new(),
            warm_accesses_latex: String::new(),
            compulsory_accesses_plain: String::new(),
            compulsory_accesses_latex: String::new(),
            timestamp_space: String::new(),
            access_map: String::new(),
            ri_distribution: ri_entries.into_iter().map(|entry| entry.entry).collect(),
            rd_distribution: Vec::new(),
            dmd_terms: Vec::new(),
            dmd_formula_plain: String::new(),
            dmd_formula_latex: String::new(),
            parallel: Some(parallel),
            notes,
        });
    }

    let rd_relation = ri_relation.apply_range(access_map.clone())?;
    let rd_values = rd_relation.cardinality()?;
    let total_accesses = timestamp_space.clone().cardinality()?;
    let rd_distribution = DistributionProcessor::new(rd_values).collect()?;
    let total_expr = render_piecewise(total_accesses.clone())?;
    let rd_rendered = render_distribution(&rd_distribution)?;
    let warm_expr = FormulaExpr::add(rd_rendered.count_exprs.iter().cloned());
    let compulsory_expr = FormulaExpr::sub(total_expr.clone(), warm_expr.clone());
    let mut dmd_terms = Vec::new();
    let mut dmd_formula_terms = vec![compulsory_expr.clone()];
    let mut filtered_region_count = 0usize;

    for entry in &rd_rendered.entries {
        for region in &entry.regions {
            if !region_scales(&region.domain)? {
                filtered_region_count += 1;
                continue;
            }
            let term_expr = FormulaExpr::mul([
                region.count_expr.clone(),
                FormulaExpr::sqrt(entry.value_expr.clone()),
            ]);
            dmd_formula_terms.push(term_expr.clone());
            dmd_terms.push(DmdTerm {
                domain_plain: region.region.domain_plain.clone(),
                multiplicity_plain: region.region.count_plain.clone(),
                multiplicity_latex: region.region.count_latex.clone(),
                reuse_distance_plain: entry.entry.value_plain.clone(),
                reuse_distance_latex: entry.entry.value_latex.clone(),
                term_plain: term_expr.to_plain(),
                term_latex: term_expr.to_latex(),
            });
        }
    }

    let ri_distribution = ri_entries
        .into_iter()
        .map(|entry| entry.entry)
        .collect::<Vec<_>>();
    let rd_distribution = rd_rendered
        .entries
        .into_iter()
        .map(|entry| entry.entry)
        .collect::<Vec<_>>();
    let dmd_formula = FormulaExpr::add(dmd_formula_terms);
    let approximation_arg = options.approximation_method.as_barvinok_arg().to_string();
    let mut notes = vec![
        format!(
            "Barvinok runs inside a context initialized with `{}`.",
            approximation_arg
        ),
        "Reuse intervals are computed from the symbolic access relation, following the autolala-style Barvinok construction.".to_string(),
        "Reuse distance is derived from the cardinality of the access-image inside each reuse interval window, without Denning recursion.".to_string(),
        "The compulsory-access term is modeled separately as total accesses minus warm reuses.".to_string(),
    ];
    if filtered_region_count > 0 {
        notes.push(format!(
            "Filtered {filtered_region_count} DMD region(s) whose domains add equality or upper-bound constraints on named scaling dimensions."
        ));
    }
    if filtered_ri_region_count > 0 {
        notes.push(format!(
            "Filtered {filtered_ri_region_count} RI region(s) whose domains add equality or upper-bound constraints on named scaling dimensions."
        ));
    }

    Ok(AnalysisReport {
        options,
        total_accesses_plain: total_expr.to_plain(),
        total_accesses_latex: total_expr.to_latex(),
        warm_accesses_plain: warm_expr.to_plain(),
        warm_accesses_latex: warm_expr.to_latex(),
        compulsory_accesses_plain: compulsory_expr.to_plain(),
        compulsory_accesses_latex: compulsory_expr.to_latex(),
        timestamp_space: format!("{timestamp_space:?}"),
        access_map: format!("{access_map:?}"),
        ri_distribution,
        rd_distribution,
        dmd_terms,
        dmd_formula_plain: dmd_formula.to_plain(),
        dmd_formula_latex: dmd_formula.to_latex(),
        parallel: None,
        notes,
    })
}

fn analyze_parallel_with_context<'ctx>(
    ctx: ContextRef<'ctx>,
    model: &SemanticProgram,
    options: AnalysisOptions,
) -> DmdResult<AnalysisReport> {
    let (private_model, thread_var) = build_private_trace_model(model)?;
    let private_lowering = Lowering::new(&private_model);
    let timestamp_space = private_lowering.get_timestamp_space(ctx)?;
    let access_map = private_lowering
        .get_access_map(ctx, &options)?
        .intersect_domain(timestamp_space.clone())?;
    let space = timestamp_space.get_space()?;
    let thread_dim = timestamp_space
        .find_dim_by_name(DimType::Out, &thread_var)
        .ok_or_else(|| DmdError::analysis("missing synthetic thread dimension in PRI model"))?;
    let lt = same_thread_lex_order(
        ctx,
        space.clone(),
        timestamp_space.clone(),
        thread_dim,
        false,
    )?;
    let le = same_thread_lex_order(
        ctx,
        space.clone(),
        timestamp_space.clone(),
        thread_dim,
        true,
    )?;
    let access_rev = access_map.clone().reverse()?;
    let same_element = access_map.clone().apply_range(access_rev)?;
    let immediate_next = same_element.intersect(lt.clone())?.lexmin()?;
    let immediate_prev = immediate_next.reverse()?;
    let interval = immediate_prev.apply_range(lt)?;
    let ri_relation = interval.intersect(le.reverse()?)?;
    let ri_values = ri_relation.clone().cardinality()?;
    let ri_distribution = DistributionProcessor::new(ri_values).collect()?;
    let ri_rendered = render_distribution(&ri_distribution)?;
    let (ri_entries, filtered_ri_region_count) = filter_rendered_entries(ri_rendered.entries)?;
    let parallel = build_parallel_analysis(model, &ri_entries)?
        .ok_or_else(|| DmdError::analysis("parallel analysis expected an annotated loop"))?;
    let approximation_arg = options.approximation_method.as_barvinok_arg().to_string();
    let mut notes = vec![
        format!(
            "Barvinok runs inside a context initialized with `{}`.",
            approximation_arg
        ),
        "Reuse intervals are computed from the static per-thread access relation induced by the parallel schedule, not from the original sequential RI stream.".to_string(),
        "The static schedule is modeled as chunk-size-1 round-robin assignment over the parallel loop's ordinal iterations.".to_string(),
        "Parallel loop mode stops after PRI collection and CRI law generation; sequential RD/DMD statistics are skipped to keep the analysis fast.".to_string(),
    ];
    if filtered_ri_region_count > 0 {
        notes.push(format!(
            "Filtered {filtered_ri_region_count} RI region(s) whose domains add equality or upper-bound constraints on named scaling dimensions."
        ));
    }

    Ok(AnalysisReport {
        options,
        total_accesses_plain: String::new(),
        total_accesses_latex: String::new(),
        warm_accesses_plain: String::new(),
        warm_accesses_latex: String::new(),
        compulsory_accesses_plain: String::new(),
        compulsory_accesses_latex: String::new(),
        timestamp_space: String::new(),
        access_map: String::new(),
        ri_distribution: ri_entries.into_iter().map(|entry| entry.entry).collect(),
        rd_distribution: Vec::new(),
        dmd_terms: Vec::new(),
        dmd_formula_plain: String::new(),
        dmd_formula_latex: String::new(),
        parallel: Some(parallel),
        notes,
    })
}

fn same_thread_lex_order<'ctx>(
    ctx: ContextRef<'ctx>,
    space: Space<'ctx>,
    timestamp_space: Set<'ctx>,
    thread_dim: u32,
    inclusive: bool,
) -> DmdResult<Map<'ctx>> {
    let map = if inclusive {
        Map::lex_le(space)?
    } else {
        Map::lex_lt(space)?
    }
    .intersect_domain(timestamp_space.clone())?
    .intersect_range(timestamp_space)?;
    let local_space = LocalSpace::try_from(map.get_space()?)?;
    Ok(map.add_constraint(
        Constraint::new_equality(local_space)?
            .set_coefficient_val(DimType::In, thread_dim as i32, Value::int_from_si(ctx, 1)?)?
            .set_coefficient_val(
                DimType::Out,
                thread_dim as i32,
                Value::int_from_si(ctx, -1)?,
            )?,
    )?)
}

fn build_private_trace_model(model: &SemanticProgram) -> DmdResult<(SemanticProgram, String)> {
    let mut used_names = collect_names(model);
    let mut substitution = HashMap::new();
    let thread_var = fresh_name("__autolala_tid", &mut used_names);
    let body =
        rewrite_parallel_block(&model.body, &mut used_names, &mut substitution, &thread_var)?;
    let program = Program {
        params: model.params.clone(),
        arrays: model
            .arrays
            .iter()
            .map(|array| crate::ast::ArrayDecl {
                name: array.name.clone(),
                extents: array.extents.clone(),
            })
            .collect(),
        body,
    };
    let private_model = validate_program(program)?;
    Ok((private_model, thread_var))
}

fn collect_names(model: &SemanticProgram) -> HashSet<String> {
    fn collect_expr(expr: &Expr, used: &mut HashSet<String>) {
        match expr {
            Expr::Int(_) => {}
            Expr::Var(name) => {
                used.insert(name.clone());
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::FloorDiv(lhs, rhs) => {
                collect_expr(lhs, used);
                collect_expr(rhs, used);
            }
            Expr::Neg(expr) => collect_expr(expr, used),
        }
    }

    fn collect_block(block: &Block, used: &mut HashSet<String>) {
        for stmt in &block.statements {
            match stmt {
                Stmt::For(for_loop) => {
                    used.insert(for_loop.var.clone());
                    collect_expr(&for_loop.lower, used);
                    collect_expr(&for_loop.upper, used);
                    if let Some(parallel) = &for_loop.parallel {
                        collect_expr(&parallel.threads, used);
                    }
                    collect_block(&for_loop.body, used);
                }
                Stmt::If(if_stmt) => {
                    for condition in &if_stmt.conditions {
                        collect_expr(&condition.lhs, used);
                        collect_expr(&condition.rhs, used);
                    }
                    collect_block(&if_stmt.then_branch, used);
                    if let Some(else_branch) = &if_stmt.else_branch {
                        collect_block(else_branch, used);
                    }
                }
                Stmt::Access(access) => {
                    used.insert(access.array.clone());
                    for index in &access.indices {
                        collect_expr(index, used);
                    }
                }
            }
        }
    }

    let mut used = model.params.iter().cloned().collect::<HashSet<_>>();
    for array in &model.arrays {
        used.insert(array.name.clone());
        for extent in &array.extents {
            collect_expr(extent, &mut used);
        }
    }
    collect_block(&model.body, &mut used);
    used
}

fn fresh_name(prefix: &str, used_names: &mut HashSet<String>) -> String {
    if used_names.insert(prefix.to_string()) {
        return prefix.to_string();
    }

    let mut index = 0usize;
    loop {
        let candidate = format!("{prefix}_{index}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

fn rewrite_parallel_block(
    block: &Block,
    used_names: &mut HashSet<String>,
    substitution: &mut HashMap<String, Expr>,
    thread_var: &str,
) -> DmdResult<Block> {
    let statements = block
        .statements
        .iter()
        .map(|stmt| rewrite_parallel_stmt(stmt, used_names, substitution, thread_var))
        .collect::<DmdResult<Vec<_>>>()?;
    Ok(Block::new(statements))
}

fn rewrite_parallel_stmt(
    stmt: &Stmt,
    used_names: &mut HashSet<String>,
    substitution: &mut HashMap<String, Expr>,
    thread_var: &str,
) -> DmdResult<Stmt> {
    match stmt {
        Stmt::For(for_loop) => rewrite_parallel_for(for_loop, used_names, substitution, thread_var),
        Stmt::If(if_stmt) => Ok(Stmt::If(crate::ast::IfStmt {
            conditions: if_stmt
                .conditions
                .iter()
                .map(|condition| {
                    Ok(Comparison {
                        lhs: substitute_expr(&condition.lhs, substitution),
                        op: condition.op,
                        rhs: substitute_expr(&condition.rhs, substitution),
                    })
                })
                .collect::<DmdResult<Vec<_>>>()?,
            then_branch: rewrite_parallel_block(
                &if_stmt.then_branch,
                used_names,
                substitution,
                thread_var,
            )?,
            else_branch: if let Some(else_branch) = &if_stmt.else_branch {
                Some(rewrite_parallel_block(
                    else_branch,
                    used_names,
                    substitution,
                    thread_var,
                )?)
            } else {
                None
            },
        })),
        Stmt::Access(access) => Ok(Stmt::Access(crate::ast::Access {
            kind: access.kind,
            array: access.array.clone(),
            indices: access
                .indices
                .iter()
                .map(|index| substitute_expr(index, substitution))
                .collect(),
        })),
    }
}

fn rewrite_parallel_for(
    for_loop: &crate::ast::ForLoop,
    used_names: &mut HashSet<String>,
    substitution: &mut HashMap<String, Expr>,
    thread_var: &str,
) -> DmdResult<Stmt> {
    if for_loop.parallel.is_none() {
        return Ok(Stmt::For(crate::ast::ForLoop {
            parallel: None,
            var: for_loop.var.clone(),
            lower: substitute_expr(&for_loop.lower, substitution),
            upper: substitute_expr(&for_loop.upper, substitution),
            step: for_loop.step,
            body: rewrite_parallel_block(&for_loop.body, used_names, substitution, thread_var)?,
        }));
    }

    let Some(threads) = for_loop
        .parallel
        .as_ref()
        .and_then(|parallel| parallel.threads.as_const_i64())
    else {
        return Err(DmdError::analysis(
            "parallel loop thread count must be constant during PRI lowering",
        ));
    };
    let private_var = fresh_name(&format!("__autolala_private_{}", for_loop.var), used_names);
    let lower = substitute_expr(&for_loop.lower, substitution);
    let upper = substitute_expr(&for_loop.upper, substitution);
    let trip = loop_trip_expr(&lower, &upper, for_loop.step);
    let private_upper = Expr::FloorDiv(
        Box::new(Expr::Add(
            Box::new(Expr::Sub(
                Box::new(trip),
                Box::new(Expr::Var(thread_var.to_string())),
            )),
            Box::new(Expr::Int(threads - 1)),
        )),
        Box::new(Expr::Int(threads)),
    );
    let actual_parallel_value = Expr::Add(
        Box::new(lower),
        Box::new(Expr::Mul(
            Box::new(Expr::Int(for_loop.step)),
            Box::new(Expr::Add(
                Box::new(Expr::Var(thread_var.to_string())),
                Box::new(Expr::Mul(
                    Box::new(Expr::Int(threads)),
                    Box::new(Expr::Var(private_var.clone())),
                )),
            )),
        )),
    );

    substitution.insert(for_loop.var.clone(), actual_parallel_value);
    let rewritten_body =
        rewrite_parallel_block(&for_loop.body, used_names, substitution, thread_var)?;
    substitution.remove(&for_loop.var);

    Ok(Stmt::For(crate::ast::ForLoop {
        parallel: None,
        var: thread_var.to_string(),
        lower: Expr::Int(0),
        upper: Expr::Int(threads),
        step: 1,
        body: Block::new(vec![Stmt::For(crate::ast::ForLoop {
            parallel: None,
            var: private_var,
            lower: Expr::Int(0),
            upper: private_upper,
            step: 1,
            body: rewritten_body,
        })]),
    }))
}

fn loop_trip_expr(lower: &Expr, upper: &Expr, step: i64) -> Expr {
    Expr::FloorDiv(
        Box::new(Expr::Add(
            Box::new(Expr::Sub(Box::new(upper.clone()), Box::new(lower.clone()))),
            Box::new(Expr::Int(step - 1)),
        )),
        Box::new(Expr::Int(step)),
    )
}

fn substitute_expr(expr: &Expr, substitution: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Int(value) => Expr::Int(*value),
        Expr::Var(name) => substitution
            .get(name)
            .cloned()
            .unwrap_or_else(|| Expr::Var(name.clone())),
        Expr::Add(lhs, rhs) => Expr::Add(
            Box::new(substitute_expr(lhs, substitution)),
            Box::new(substitute_expr(rhs, substitution)),
        ),
        Expr::Sub(lhs, rhs) => Expr::Sub(
            Box::new(substitute_expr(lhs, substitution)),
            Box::new(substitute_expr(rhs, substitution)),
        ),
        Expr::Mul(lhs, rhs) => Expr::Mul(
            Box::new(substitute_expr(lhs, substitution)),
            Box::new(substitute_expr(rhs, substitution)),
        ),
        Expr::FloorDiv(lhs, rhs) => Expr::FloorDiv(
            Box::new(substitute_expr(lhs, substitution)),
            Box::new(substitute_expr(rhs, substitution)),
        ),
        Expr::Neg(inner) => Expr::Neg(Box::new(substitute_expr(inner, substitution))),
    }
}

fn build_parallel_analysis(
    model: &SemanticProgram,
    ri_entries: &[RenderedDistributionEntry<'_>],
) -> DmdResult<Option<ParallelAnalysis>> {
    let Some(thread_count) = model.thread_count() else {
        return Ok(None);
    };

    let schedule = "OpenMP static-style round-robin schedule with chunk size 1".to_string();
    let mut cri_entries = Vec::new();
    let mut negative_binomial_regions = 0usize;
    let mut racetrack_regions = 0usize;

    for entry in ri_entries {
        let input_independent_ri = entry.value_expr.as_const_i64().filter(|value| *value > 0);
        for region in &entry.regions {
            let (model_kind, law) = if let Some(source_ri) = input_independent_ri {
                negative_binomial_regions += 1;
                (
                    ParallelModelKind::NegativeBinomial,
                    negative_binomial_law(thread_count, source_ri),
                )
            } else {
                racetrack_regions += 1;
                (
                    ParallelModelKind::Racetrack,
                    racetrack_law(thread_count, entry.value_expr.clone()),
                )
            };
            let law = with_count_law(law, &region.count_expr);
            cri_entries.push(ParallelCriEntry {
                domain_plain: region.region.domain_plain.clone(),
                multiplicity_plain: region.region.count_plain.clone(),
                multiplicity_latex: region.region.count_latex.clone(),
                source_ri_plain: entry.entry.value_plain.clone(),
                source_ri_latex: entry.entry.value_latex.clone(),
                model_kind,
                law,
            });
        }
    }

    Ok(Some(ParallelAnalysis {
        thread_count,
        schedule,
        cri_entries,
        notes: vec![
            format!(
                "Parallel CRI modeling is enabled for `{}` threads.",
                thread_count
            ),
            "Each CRI region is reported as an implicit symbolic law over X rather than as an expanded table.".to_string(),
            "For negative-binomial entries, Prob(X) is a PMF over integer X >= 0.".to_string(),
            "For racetrack entries, Prob(X) is a density over continuous X in [0, 1].".to_string(),
            format!(
                "Generated {negative_binomial_regions} negative-binomial law(s) and {racetrack_regions} racetrack law(s)."
            ),
        ],
    }))
}

fn negative_binomial_law(thread_count: i64, source_ri: i64) -> ParallelCriLaw {
    ParallelCriLaw {
        range_plain: "X >= 0, X integer".to_string(),
        range_latex: "X \\in \\mathbb{Z}_{\\ge 0}".to_string(),
        cri_plain: format!("{source_ri} + X"),
        cri_latex: format!("{source_ri} + X"),
        probability_plain: format!(
            "binom(X + {} - 1, {} - 1) * (1/{thread_count})^{} * ({}/{thread_count})^X",
            source_ri,
            source_ri,
            source_ri,
            thread_count - 1,
        ),
        probability_latex: format!(
            "\\binom{{X + {} - 1}}{{{} - 1}} \\left(\\frac{{1}}{{{thread_count}}}\\right)^{{{}}} \\left(\\frac{{{}}}{{{thread_count}}}\\right)^X",
            source_ri,
            source_ri,
            source_ri,
            thread_count - 1,
        ),
        count_plain: String::new(),
        count_latex: String::new(),
    }
}

fn racetrack_law(thread_count: i64, source_ri: FormulaExpr) -> ParallelCriLaw {
    ParallelCriLaw {
        range_plain: "0 <= X <= 1".to_string(),
        range_latex: "0 \\le X \\le 1".to_string(),
        cri_plain: format!("X * ({})", source_ri.to_plain()),
        cri_latex: format!("X \\cdot \\left({}\\right)", source_ri.to_latex()),
        probability_plain: if thread_count == 2 {
            "1".to_string()
        } else {
            format!("{} * (1 - X)^{}", thread_count - 1, thread_count - 2)
        },
        probability_latex: if thread_count == 2 {
            "1".to_string()
        } else {
            format!(
                "{} \\cdot (1 - X)^{{{}}}",
                thread_count - 1,
                thread_count - 2
            )
        },
        count_plain: String::new(),
        count_latex: String::new(),
    }
}

fn with_count_law(mut law: ParallelCriLaw, multiplicity: &FormulaExpr) -> ParallelCriLaw {
    let multiplicity_plain = multiplicity.to_plain();
    let multiplicity_latex = multiplicity.to_latex();
    law.count_plain = if multiplicity_plain == "1" {
        law.probability_plain.clone()
    } else {
        format!("({multiplicity_plain}) * ({})", law.probability_plain)
    };
    law.count_latex = if multiplicity_latex == "1" {
        law.probability_latex.clone()
    } else {
        format!(
            "\\left({multiplicity_latex}\\right) \\cdot \\left({}\\right)",
            law.probability_latex
        )
    };
    law
}

struct RenderedDistribution<'a> {
    entries: Vec<RenderedDistributionEntry<'a>>,
    count_exprs: Vec<FormulaExpr>,
}

struct RenderedDistributionEntry<'a> {
    entry: DistributionEntry,
    value_expr: FormulaExpr,
    regions: Vec<RenderedDistributionRegion<'a>>,
}

struct RenderedDistributionRegion<'a> {
    region: DistributionRegion,
    count_expr: FormulaExpr,
    domain: Set<'a>,
}

fn filter_rendered_entries<'a>(
    entries: Vec<RenderedDistributionEntry<'a>>,
) -> DmdResult<(Vec<RenderedDistributionEntry<'a>>, usize)> {
    let mut filtered_region_count = 0usize;
    let mut kept_entries = Vec::new();

    for mut entry in entries {
        let mut kept_regions = Vec::new();
        for region in entry.regions {
            if region_scales(&region.domain)? {
                kept_regions.push(region);
            } else {
                filtered_region_count += 1;
            }
        }
        if kept_regions.is_empty() {
            continue;
        }
        entry.entry.regions = kept_regions
            .iter()
            .map(|region| region.region.clone())
            .collect::<Vec<_>>();
        entry.regions = kept_regions;
        kept_entries.push(entry);
    }

    Ok((kept_entries, filtered_region_count))
}

fn render_distribution<'a>(items: &[DistributionItem<'a>]) -> DmdResult<RenderedDistribution<'a>> {
    let mut entries = Vec::new();
    let mut count_exprs = Vec::new();

    for item in items {
        let value_formatter = FormulaFormatter::new(item.value.get_space()?);
        let value_expr = value_formatter.quasi_polynomial(item.value.clone())?;
        let mut regions = Vec::new();
        item.cardinality.foreach_piece(|qpoly, domain| {
            let formatter = FormulaFormatter::new(qpoly.get_space()?);
            let count_expr = formatter.quasi_polynomial(qpoly)?;
            count_exprs.push(count_expr.clone());
            regions.push(RenderedDistributionRegion {
                region: DistributionRegion {
                    domain_plain: format_domain(&domain),
                    count_plain: count_expr.to_plain(),
                    count_latex: count_expr.to_latex(),
                },
                count_expr,
                domain,
            });
            Ok(())
        })?;
        entries.push(RenderedDistributionEntry {
            entry: DistributionEntry {
                value_plain: value_expr.to_plain(),
                value_latex: value_expr.to_latex(),
                regions: regions.iter().map(|region| region.region.clone()).collect(),
            },
            value_expr,
            regions,
        });
    }

    Ok(RenderedDistribution {
        entries,
        count_exprs,
    })
}

fn region_scales(domain: &Set<'_>) -> DmdResult<bool> {
    let basic_sets = domain.get_basic_set_list()?;
    if basic_sets.is_empty() {
        return Ok(false);
    }

    for basic_set in basic_sets.iter() {
        if basic_set_scales(&basic_set)? {
            return Ok(true);
        }
    }

    Ok(false)
}

fn basic_set_scales(basic_set: &BasicSet<'_>) -> DmdResult<bool> {
    for constraint in basic_set.clone().get_constraints()?.iter() {
        if equality_hits_named_scaling_dimension(&constraint, basic_set)?
            || upper_bound_hits_named_scaling_dimension(&constraint, basic_set)?
        {
            return Ok(false);
        }
    }

    Ok(true)
}

fn equality_hits_named_scaling_dimension(
    constraint: &Constraint<'_>,
    basic_set: &BasicSet<'_>,
) -> DmdResult<bool> {
    if !constraint.is_equality()? {
        return Ok(false);
    }

    named_scaling_dims(basic_set)?
        .into_iter()
        .try_fold(false, |found, (dim_type, pos)| {
            if found {
                return Ok(true);
            }
            Ok(constraint.involves_dims(dim_type, pos, 1)?)
        })
}

fn upper_bound_hits_named_scaling_dimension(
    constraint: &Constraint<'_>,
    basic_set: &BasicSet<'_>,
) -> DmdResult<bool> {
    for (dim_type, pos) in named_scaling_dims(basic_set)? {
        if constraint.is_upper_bound(dim_type, pos)? {
            return Ok(true);
        }
    }

    Ok(false)
}

fn named_scaling_dims(basic_set: &BasicSet<'_>) -> DmdResult<Vec<(DimType, u32)>> {
    let mut dims = Vec::new();

    for dim_type in [DimType::Param, DimType::Out] {
        let dim_count = basic_set.dim(dim_type)?;
        for pos in 0..dim_count {
            if basic_set.get_dim_name(dim_type, pos)?.is_some() {
                dims.push((dim_type, pos));
            }
        }
    }

    Ok(dims)
}

fn render_piecewise(pw: PiecewiseQuasiPolynomial<'_>) -> DmdResult<FormulaExpr> {
    let mut pieces = Vec::new();
    pw.foreach_piece(|qpoly, domain| {
        let formatter = FormulaFormatter::new(qpoly.get_space()?);
        let expr = formatter.quasi_polynomial(qpoly)?;
        pieces.push((expr, format_domain(&domain)));
        Ok(())
    })?;

    match pieces.len() {
        0 => Ok(FormulaExpr::zero()),
        1 => Ok(pieces
            .into_iter()
            .next()
            .map(|(expr, _)| expr)
            .unwrap_or_else(FormulaExpr::zero)),
        _ => {
            let plain = pieces
                .iter()
                .map(|(expr, region)| format!("[{region}] => {}", expr.to_plain()))
                .collect::<Vec<_>>();
            let latex = pieces
                .iter()
                .map(|(expr, region)| {
                    format!("\\left[{}\\right] \\Rightarrow {}", region, expr.to_latex())
                })
                .collect::<Vec<_>>();
            Ok(FormulaExpr::raw(join_add(plain), join_add(latex)))
        }
    }
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
                .find(|piece: &&mut DistributionPiece<'a>| {
                    qpoly.plain_is_equal(&piece.value).unwrap_or(false)
                })
            {
                existing.domain = existing.domain.clone().union(domain)?;
            } else {
                pieces.push(DistributionPiece {
                    domain,
                    value: qpoly,
                });
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
        let mut aligned = set
            .clone()
            .insert_dims(DimType::Out, dims, longest_dims - dims)?;
        for dim in dims..longest_dims {
            aligned = aligned.add_constraint(
                Constraint::new_equality(local_space.clone())?.set_coefficient_si(
                    DimType::Out,
                    dim as i32,
                    1,
                )?,
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
                Constraint::new_equality(local_space.clone())?.set_coefficient_si(
                    DimType::In,
                    dim as i32,
                    1,
                )?,
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

    const NESTED: &str = r#"
params N, M;
array A[N, M];
array B[M];

for i in 0 .. N {
    for j in 0 .. M {
        read A[i, j];
        read B[j];
    }
}
"#;

    const FOUR_ACCESS_MATMUL: &str = r#"
params M, N, K;
array A[M, K];
array B[K, N];
array C[M, N];

for i in 0 .. M {
    for j in 0 .. N {
        for k in 0 .. K {
            read C[i, j];
            read A[i, k];
            read B[k, j];
            write C[i, j];
        }
    }
}
"#;

    const PARALLEL_CONSTANT_RI: &str = r#"
params N;
array A[N];

parallel(4) for i in 0 .. N {
    read A[0];
}
"#;

    const PARALLEL_SYMBOLIC_RI: &str = r#"
params N, M;
array A[M];

parallel(4) for i in 0 .. N {
    for j in 0 .. M {
        read A[j];
    }
}
"#;

    const PARALLEL_PRI_DIFFERS_FROM_SEQUENTIAL_RI: &str = r#"
params N;
array A[N];

parallel(4) for i in 0 .. N {
    read A[i / 4];
}
"#;

    #[test]
    fn repeated_single_access_has_unit_rd() {
        let report =
            analyze_source(SIMPLE, AnalysisOptions::default()).expect("analysis should succeed");
        assert!(!report.rd_distribution.is_empty());
        assert!(
            report
                .rd_distribution
                .iter()
                .any(|entry| entry.value_plain == "1")
        );
        assert!(report.warm_accesses_plain.contains("N"));
    }

    #[test]
    fn dmd_formula_uses_clean_subtractions() {
        let report =
            analyze_source(SIMPLE, AnalysisOptions::default()).expect("analysis should succeed");
        assert!(!report.dmd_formula_plain.contains("+ -"));
        let nested = analyze_source(NESTED, AnalysisOptions::default())
            .expect("nested analysis should succeed");
        assert!(nested.dmd_formula_latex.contains("\\sqrt"));
    }

    #[test]
    fn default_analysis_uses_scale_approximation_context() {
        let report =
            analyze_source(SIMPLE, AnalysisOptions::default()).expect("analysis should succeed");
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("--approximation-method=scale"))
        );
        assert_eq!(
            report.options.approximation_method,
            ApproximationMethod::Scale
        );
    }

    #[test]
    fn dmd_filters_non_scaling_domains() {
        let report = analyze_source(FOUR_ACCESS_MATMUL, AnalysisOptions::default())
            .expect("analysis should succeed");

        assert!(
            report
                .dmd_terms
                .iter()
                .all(|term| !term.domain_plain.contains(" = "))
        );
        assert!(
            report
                .dmd_terms
                .iter()
                .all(|term| !term.domain_plain.contains("<="))
        );
        assert!(
            report
                .dmd_terms
                .iter()
                .any(|term| term.domain_plain == "M >= 2 and N >= 3 and K >= 3")
        );
        assert!(
            !report
                .dmd_terms
                .iter()
                .any(|term| term.domain_plain.contains("K = 2"))
        );
        assert!(
            !report
                .dmd_terms
                .iter()
                .any(|term| term.domain_plain.contains("0 < k <= -2 + K"))
        );
        assert!(report.notes.iter().any(|note| note.contains("Filtered")));
    }

    #[test]
    fn ri_distribution_filters_non_scaling_domains() {
        let report = analyze_source(FOUR_ACCESS_MATMUL, AnalysisOptions::default())
            .expect("analysis should succeed");

        assert!(
            report
                .ri_distribution
                .iter()
                .flat_map(|entry| entry.regions.iter())
                .all(|region| !region.domain_plain.contains(" = "))
        );
        assert!(
            report
                .ri_distribution
                .iter()
                .flat_map(|entry| entry.regions.iter())
                .all(|region| !region.domain_plain.contains("<="))
        );
        assert!(
            !report
                .ri_distribution
                .iter()
                .flat_map(|entry| entry.regions.iter())
                .any(|region| region.domain_plain.contains("K = 2"))
        );
        assert!(report.notes.iter().any(|note| note.contains("RI region")));
    }

    #[test]
    fn analysis_is_safe_under_parallel_calls() {
        std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        analyze_source(FOUR_ACCESS_MATMUL, AnalysisOptions::default())
                            .expect("analysis should succeed")
                    })
                })
                .collect::<Vec<_>>();

            for handle in handles {
                let report = handle.join().expect("worker thread should not panic");
                assert!(!report.dmd_formula_plain.is_empty());
            }
        });
    }

    #[test]
    fn parallel_loops_emit_negative_binomial_laws_for_input_independent_ri() {
        let report = analyze_source(PARALLEL_CONSTANT_RI, AnalysisOptions::default())
            .expect("analysis should succeed");
        let parallel = report
            .parallel
            .expect("parallel analysis should be present");
        assert_eq!(parallel.thread_count, 4);
        assert!(report.rd_distribution.is_empty());
        assert!(report.dmd_formula_plain.is_empty());
        assert!(parallel.cri_entries.iter().any(|entry| {
            entry.model_kind == ParallelModelKind::NegativeBinomial
                && entry.source_ri_plain == "1"
                && entry.law.cri_plain == "1 + X"
                && entry.law.probability_plain == "binom(X + 1 - 1, 1 - 1) * (1/4)^1 * (3/4)^X"
        }));
    }

    #[test]
    fn parallel_loops_emit_racetrack_laws_for_symbolic_ri() {
        let report = analyze_source(PARALLEL_SYMBOLIC_RI, AnalysisOptions::default())
            .expect("analysis should succeed");
        let parallel = report
            .parallel
            .expect("parallel analysis should be present");
        assert!(parallel.cri_entries.iter().any(|entry| {
            entry.model_kind == ParallelModelKind::Racetrack
                && entry.source_ri_plain.contains('M')
                && entry.law.cri_plain.contains('M')
                && entry.law.probability_plain == "3 * (1 - X)^2"
        }));
        assert!(
            parallel
                .notes
                .iter()
                .any(|note| note.contains("racetrack law"))
        );
    }

    #[test]
    fn parallel_pri_is_computed_from_private_trace_not_sequential_ri() {
        let sequential = analyze_source(
            r#"
params N;
array A[N];

for i in 0 .. N {
    read A[i / 4];
}
"#,
            AnalysisOptions::default(),
        )
        .expect("sequential analysis should succeed");
        assert!(
            sequential
                .ri_distribution
                .iter()
                .flat_map(|entry| entry.regions.iter())
                .any(|region| region.count_plain == "N - 1"),
            "sequential RI should observe the repeated A[i / 4] accesses"
        );

        let parallel = analyze_source(
            PARALLEL_PRI_DIFFERS_FROM_SEQUENTIAL_RI,
            AnalysisOptions::default(),
        )
        .expect("parallel analysis should succeed");
        let parallel_analysis = parallel
            .parallel
            .expect("parallel analysis should be present");

        assert!(parallel.ri_distribution.is_empty());
        assert!(parallel_analysis.cri_entries.is_empty());
        assert!(parallel.notes.iter().any(|note| {
            note.contains("per-thread access relation")
                || note.contains("original sequential RI stream")
        }));
    }

    #[test]
    fn parallel_ri_laws_filter_non_scaling_domains() {
        let report = analyze_source(
            r#"
params M, N, K;
array A[M, K];
array B[K, N];
array C[M, N];

parallel(4) for i in 0 .. M {
    for j in 0 .. N {
        for k in 0 .. K {
            read C[i, j];
            read A[i, k];
            read B[k, j];
            write C[i, j];
        }
    }
}
"#,
            AnalysisOptions::default(),
        )
        .expect("analysis should succeed");
        let parallel = report
            .parallel
            .expect("parallel analysis should be present");

        assert!(
            parallel
                .cri_entries
                .iter()
                .all(|entry| !entry.domain_plain.contains(" = "))
        );
        assert!(
            parallel
                .cri_entries
                .iter()
                .all(|entry| !entry.domain_plain.contains("<="))
        );
        assert!(
            !parallel
                .cri_entries
                .iter()
                .any(|entry| entry.domain_plain.contains("K = 1"))
        );
        assert!(report.notes.iter().any(|note| note.contains("RI region")));
    }
}
