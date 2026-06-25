//! Translate an attribute-tagged `affine.for` loop nest into the AutoLALA DSL
//! ("the language"), reusing `dmd-core`'s AST as the target representation.
//!
//! Supported MLIR constructs:
//!   * `affine.for`  -> `for v in lo .. hi step s { ... }`
//!   * `affine.if`   -> `if c0 && c1 { ... } else { ... }`
//!   * `affine.load` -> `read  A[..];`
//!   * `affine.store`-> `write A[..];`
//!   * pure arithmetic (`arith.*`, `math.*`, `complex.*`) is ignored — only the
//!     memory-access structure matters for locality analysis.
//!
//! Anything else (non-affine memory ops, `scf.*`, `func.call`, `affine.apply`,
//! `min`/`max` bounds, `mod`/`ceildiv`, loop-carried reductions, ...) is
//! rejected with an [`ExtractError`] carrying the operation's source span.

use std::collections::HashMap;

use dmd_core::ast::{
    Access, AccessKind, ArrayDecl, Block, Comparison, ComparisonOp, Expr, ForLoop, IfStmt, Program,
    Stmt,
};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;
use melior::ir::{Module, OperationRef, RegionLike, Value, ValueLike};

use crate::error::{ExtractError, ExtractResult};
use crate::ffi::{
    AffineExpr, AffineExprKind, AffineMap, IntegerSet, SourceSpan, integer_attr_value,
    operation_span, shaped_type_shape,
};

/// Translate the loop tagged with `attr` in `module` into a DSL [`Program`].
pub fn extract_program(module: &Module<'_>, attr: &str) -> ExtractResult<Program> {
    let mut tagged = Vec::new();
    collect_tagged(module.as_operation(), attr, &mut tagged);

    let target = match tagged.first() {
        Some(op) => *op,
        None => {
            return Err(ExtractError::new(format!(
                "no operation tagged with attribute `{attr}` was found"
            ))
            .with_help(format!(
                "annotate the loop to extract, e.g. `affine.for ... {{ ... }} {{{attr}}}`"
            )));
        }
    };

    if tagged.len() > 1 {
        eprintln!(
            "warning: {} operations are tagged with `{attr}`; extracting the first one",
            tagged.len()
        );
    }

    let name = operation_name(target)?;
    if name != "affine.for" {
        return Err(ExtractError::at(
            format!(
                "the `{attr}` tag must be placed on an `affine.for` loop, but it is on `{name}`"
            ),
            operation_span(target),
        ));
    }

    let mut translator = Translator::new();
    let loop_nest = translator.translate_for(target)?;
    Ok(Program {
        params: translator.params,
        arrays: translator.arrays,
        body: Block::new(vec![Stmt::For(loop_nest)]),
    })
}

/// Recursively collect every operation carrying the named attribute.
fn collect_tagged<'c, 'a>(
    op: OperationRef<'c, 'a>,
    attr: &str,
    out: &mut Vec<OperationRef<'c, 'a>>,
) {
    if op.has_attribute(attr) {
        out.push(op);
    }
    for region in op.regions() {
        let mut block = region.first_block();
        while let Some(current) = block {
            let mut inner = current.first_operation();
            while let Some(child) = inner {
                collect_tagged(child, attr, out);
                inner = child.next_in_block();
            }
            block = current.next_in_region();
        }
    }
}

fn operation_name(op: OperationRef<'_, '_>) -> ExtractResult<String> {
    op.name()
        .as_string_ref()
        .as_str()
        .map(str::to_owned)
        .map_err(|_| ExtractError::at("operation has a non-UTF-8 name", operation_span(op)))
}

/// True for dialects whose operations have no memory effect and only feed the
/// (already folded) arithmetic of a kernel body, so they are safe to ignore.
fn is_ignorable_compute(name: &str) -> bool {
    const PURE_DIALECTS: [&str; 3] = ["arith.", "math.", "complex."];
    PURE_DIALECTS.iter().any(|prefix| name.starts_with(prefix))
}

/// Mutable translation state: identifier assignment and array discovery.
struct Translator {
    params: Vec<String>,
    arrays: Vec<ArrayDecl>,
    /// SSA value pointer -> induction-variable name, for IVs inside the nest.
    induction_vars: HashMap<usize, String>,
    /// SSA value pointer -> synthesized symbolic parameter name.
    symbols: HashMap<usize, String>,
    /// memref SSA value pointer -> array name.
    array_names: HashMap<usize, String>,
    induction_counter: usize,
    symbol_counter: usize,
    array_counter: usize,
}

impl Translator {
    fn new() -> Self {
        Self {
            params: Vec::new(),
            arrays: Vec::new(),
            induction_vars: HashMap::new(),
            symbols: HashMap::new(),
            array_names: HashMap::new(),
            induction_counter: 0,
            symbol_counter: 0,
            array_counter: 0,
        }
    }

    /// Map an operand SSA value to a DSL identifier: an induction variable if it
    /// is one of the enclosing extracted loops, otherwise a fresh symbolic
    /// parameter (loop-invariant within the nest).
    fn resolve_value(&mut self, value: &Value<'_, '_>) -> String {
        let key = value.to_raw().ptr as usize;
        if let Some(name) = self.induction_vars.get(&key) {
            return name.clone();
        }
        if let Some(name) = self.symbols.get(&key) {
            return name.clone();
        }
        let name = format!("p{}", self.symbol_counter);
        self.symbol_counter += 1;
        self.params.push(name.clone());
        self.symbols.insert(key, name.clone());
        name
    }

    fn resolve_values(&mut self, values: &[Value<'_, '_>]) -> Vec<String> {
        values
            .iter()
            .map(|value| self.resolve_value(value))
            .collect()
    }

    /// Look up (or create) the DSL array for a memref SSA value, deriving its
    /// declared extents from the memref's shape.
    fn array_for(&mut self, memref: &Value<'_, '_>) -> ExtractResult<String> {
        let key = memref.to_raw().ptr as usize;
        if let Some(name) = self.array_names.get(&key) {
            return Ok(name.clone());
        }

        let name = array_letter(self.array_counter);
        let span = None; // memref values carry no useful op span here
        let shape = shaped_type_shape(memref.r#type()).ok_or_else(|| {
            ExtractError::at(
                "memory access targets a value that is not a ranked memref/shaped type",
                span,
            )
        })?;
        if shape.is_empty() {
            return Err(ExtractError::new(
                "zero-rank memref accesses cannot be represented (arrays need at least one extent)",
            ));
        }

        let mut extents = Vec::with_capacity(shape.len());
        for (dimension, extent) in shape.iter().enumerate() {
            match extent {
                Some(size) => extents.push(Expr::Int(*size)),
                None => {
                    // Dynamic dimension: introduce a fresh symbolic extent. It is
                    // independent of any loop bound because the memref type does
                    // not record which SSA value supplies the size.
                    let param = format!("{name}_d{dimension}");
                    self.params.push(param.clone());
                    extents.push(Expr::Var(param));
                }
            }
        }

        self.arrays.push(ArrayDecl {
            name: name.clone(),
            extents,
        });
        self.array_names.insert(key, name.clone());
        self.array_counter += 1;
        Ok(name)
    }

    fn translate_for(&mut self, op: OperationRef<'_, '_>) -> ExtractResult<ForLoop> {
        let span = operation_span(op);

        if op.result_count() != 0 {
            return Err(ExtractError::at(
                "affine.for produces results (loop-carried values / reduction), which the language cannot model",
                span,
            )
            .with_help("extract a loop nest without iter_args"));
        }

        let lower_map = self.bound_map(op, "lowerBoundMap", span)?;
        let upper_map = self.bound_map(op, "upperBoundMap", span)?;
        let step_attr = op
            .attribute("step")
            .map_err(|_| ExtractError::at("affine.for is missing its `step` attribute", span))?;
        let step = integer_attr_value(step_attr)
            .ok_or_else(|| ExtractError::at("affine.for `step` is not an integer", span))?;

        if lower_map.num_results() != 1 {
            return Err(ExtractError::at(
                "affine.for lower bound is a max() of several expressions, which is not representable",
                span,
            )
            .with_help("the language supports a single affine lower bound per loop"));
        }
        if upper_map.num_results() != 1 {
            return Err(ExtractError::at(
                "affine.for upper bound is a min() of several expressions, which is not representable",
                span,
            )
            .with_help("the language supports a single affine upper bound per loop"));
        }

        let lower_inputs = lower_map.num_inputs();
        let upper_inputs = upper_map.num_inputs();
        let operands = self.operands(op)?;
        if operands.len() != lower_inputs + upper_inputs {
            return Err(ExtractError::at(
                "affine.for has unexpected operands (loop-carried values are not supported)",
                span,
            ));
        }
        let (lower_operands, upper_operands) = operands.split_at(lower_inputs);

        let lower = self.lower_map_result(&lower_map, lower_operands, span)?;
        let upper = self.lower_map_result(&upper_map, upper_operands, span)?;

        // The induction variable enters scope only after the bounds are read.
        let body_block = op
            .region(0)
            .ok()
            .and_then(|region| region.first_block())
            .ok_or_else(|| ExtractError::at("affine.for has no body block", span))?;
        let induction = body_block
            .argument(0)
            .map_err(|_| ExtractError::at("affine.for is missing its induction variable", span))?;
        let var = format!("i{}", self.induction_counter);
        self.induction_counter += 1;
        self.induction_vars
            .insert(induction.to_raw().ptr as usize, var.clone());

        let body = Block::new(self.translate_block(op)?);

        Ok(ForLoop {
            parallel: None,
            var,
            lower,
            upper,
            step,
            body,
        })
    }

    fn bound_map<'c>(
        &self,
        op: OperationRef<'c, '_>,
        name: &str,
        span: Option<SourceSpan>,
    ) -> ExtractResult<AffineMap<'c>> {
        let attr = op
            .attribute(name)
            .map_err(|_| ExtractError::at(format!("affine.for is missing `{name}`"), span))?;
        AffineMap::from_attribute(attr)
            .ok_or_else(|| ExtractError::at(format!("`{name}` is not an affine map"), span))
    }

    /// Resolve the operands of a single-result-ish map and translate its only
    /// (first) result expression. Used for loop bounds.
    fn lower_map_result(
        &mut self,
        map: &AffineMap<'_>,
        operands: &[Value<'_, '_>],
        span: Option<SourceSpan>,
    ) -> ExtractResult<Expr> {
        let names = self.resolve_values(operands);
        let (dims, symbols) = names.split_at(map.num_dims());
        affine_expr_to_dsl(map.result(0), dims, symbols, span)
    }

    fn translate_load_store(
        &mut self,
        op: OperationRef<'_, '_>,
        is_write: bool,
    ) -> ExtractResult<Access> {
        let span = operation_span(op);
        let map_attr = op
            .attribute("map")
            .map_err(|_| ExtractError::at("affine access is missing its `map` attribute", span))?;
        let map = AffineMap::from_attribute(map_attr)
            .ok_or_else(|| ExtractError::at("affine access `map` is not an affine map", span))?;

        // Operand layout: load = [memref, indices...]; store = [value, memref, indices...].
        let memref_index = if is_write { 1 } else { 0 };
        let first_index = memref_index + 1;
        let memref = op
            .operand(memref_index)
            .map_err(|_| ExtractError::at("affine access is missing its memref operand", span))?;
        let array = self.array_for(&memref)?;

        let operand_count = op.operand_count();
        let mut index_operands = Vec::with_capacity(operand_count.saturating_sub(first_index));
        for i in first_index..operand_count {
            index_operands.push(
                op.operand(i)
                    .map_err(|_| ExtractError::at("invalid affine access operand", span))?,
            );
        }
        if index_operands.len() != map.num_inputs() {
            return Err(ExtractError::at(
                "affine access map operand count does not match its index operands",
                span,
            ));
        }

        let names = self.resolve_values(&index_operands);
        let (dims, symbols) = names.split_at(map.num_dims());
        let mut indices = Vec::with_capacity(map.num_results());
        for result in 0..map.num_results() {
            indices.push(affine_expr_to_dsl(map.result(result), dims, symbols, span)?);
        }

        Ok(Access {
            kind: if is_write {
                AccessKind::Write
            } else {
                AccessKind::Read
            },
            array,
            indices,
        })
    }

    fn translate_if(&mut self, op: OperationRef<'_, '_>) -> ExtractResult<IfStmt> {
        let span = operation_span(op);
        let cond_attr = op
            .attribute("condition")
            .map_err(|_| ExtractError::at("affine.if is missing its `condition`", span))?;
        let set = IntegerSet::from_attribute(cond_attr)
            .ok_or_else(|| ExtractError::at("affine.if `condition` is not an integer set", span))?;

        let operands = self.operands(op)?;
        if operands.len() != set.num_dims() + set.num_symbols() {
            return Err(ExtractError::at(
                "affine.if operand count does not match its condition",
                span,
            ));
        }
        let names = self.resolve_values(&operands);
        let (dims, symbols) = names.split_at(set.num_dims());

        let constraint_count = set.num_constraints();
        if constraint_count == 0 {
            return Err(ExtractError::at(
                "affine.if with an empty constraint set is not supported",
                span,
            ));
        }
        let mut conditions = Vec::with_capacity(constraint_count);
        for index in 0..constraint_count {
            let lhs = affine_expr_to_dsl(set.constraint(index), dims, symbols, span)?;
            // MLIR encodes constraints as `expr >= 0` (inequality) or `expr == 0`
            // (equality); the language expresses these as `expr >= 0` / `expr == 0`.
            let comparison_op = if set.constraint_is_eq(index) {
                ComparisonOp::Eq
            } else {
                ComparisonOp::Ge
            };
            conditions.push(Comparison {
                lhs,
                op: comparison_op,
                rhs: Expr::Int(0),
            });
        }

        let then_block = op
            .region(0)
            .ok()
            .and_then(|region| region.first_block())
            .ok_or_else(|| ExtractError::at("affine.if has no `then` region", span))?;
        let then_branch = Block::new(self.translate_block_of(then_block)?);

        let else_branch = if op.region_count() > 1 {
            match op.region(1).ok().and_then(|region| region.first_block()) {
                Some(block) => Some(Block::new(self.translate_block_of(block)?)),
                None => None,
            }
        } else {
            None
        };

        Ok(IfStmt {
            conditions,
            then_branch,
            else_branch,
        })
    }

    /// Translate the single body block of `op`'s first region.
    fn translate_block(&mut self, op: OperationRef<'_, '_>) -> ExtractResult<Vec<Stmt>> {
        let block = op
            .region(0)
            .ok()
            .and_then(|region| region.first_block())
            .ok_or_else(|| ExtractError::at("operation has no body block", operation_span(op)))?;
        self.translate_block_of(block)
    }

    fn translate_block_of(
        &mut self,
        block: melior::ir::BlockRef<'_, '_>,
    ) -> ExtractResult<Vec<Stmt>> {
        let mut statements = Vec::new();
        let mut current = block.first_operation();
        while let Some(op) = current {
            let name = operation_name(op)?;
            match name.as_str() {
                "affine.for" => statements.push(Stmt::For(self.translate_for(op)?)),
                "affine.load" => {
                    statements.push(Stmt::Access(self.translate_load_store(op, false)?))
                }
                "affine.store" => {
                    statements.push(Stmt::Access(self.translate_load_store(op, true)?))
                }
                "affine.if" => statements.push(Stmt::If(self.translate_if(op)?)),
                "affine.yield" => {} // region terminator, no DSL counterpart
                other if is_ignorable_compute(other) => {} // pure arithmetic
                other => {
                    return Err(ExtractError::at(
                        format!(
                            "operation `{other}` is not supported inside an extractable affine loop nest"
                        ),
                        operation_span(op),
                    )
                    .with_help(
                        "only affine.for / affine.if / affine.load / affine.store and pure \
                         arithmetic (arith.*, math.*) are understood; lower or remove this \
                         operation before extraction",
                    ));
                }
            }
            current = op.next_in_block();
        }
        Ok(statements)
    }

    fn operands<'c, 'a>(&self, op: OperationRef<'c, 'a>) -> ExtractResult<Vec<Value<'c, 'a>>> {
        let count = op.operand_count();
        let mut operands = Vec::with_capacity(count);
        for i in 0..count {
            operands.push(
                op.operand(i)
                    .map_err(|_| ExtractError::at("invalid operand index", operation_span(op)))?,
            );
        }
        Ok(operands)
    }
}

/// Translate an MLIR affine expression into a DSL [`Expr`]. `dims[k]` / `syms[k]`
/// are the already-resolved identifier names for `d_k` / `s_k`.
fn affine_expr_to_dsl(
    expr: AffineExpr<'_>,
    dims: &[String],
    syms: &[String],
    span: Option<SourceSpan>,
) -> ExtractResult<Expr> {
    match expr.kind() {
        AffineExprKind::Constant => Ok(Expr::Int(expr.constant_value())),
        AffineExprKind::Dim => {
            let position = expr.position();
            dims.get(position)
                .map(|name| Expr::Var(name.clone()))
                .ok_or_else(|| ExtractError::at("affine dimension index out of range", span))
        }
        AffineExprKind::Symbol => {
            let position = expr.position();
            syms.get(position)
                .map(|name| Expr::Var(name.clone()))
                .ok_or_else(|| ExtractError::at("affine symbol index out of range", span))
        }
        AffineExprKind::Add => {
            let lhs = affine_expr_to_dsl(expr.lhs(), dims, syms, span)?;
            let rhs = affine_expr_to_dsl(expr.rhs(), dims, syms, span)?;
            Ok(Expr::Add(Box::new(lhs), Box::new(rhs)))
        }
        AffineExprKind::Mul => {
            let lhs = affine_expr_to_dsl(expr.lhs(), dims, syms, span)?;
            let rhs = affine_expr_to_dsl(expr.rhs(), dims, syms, span)?;
            if lhs.as_const_i64().is_none() && rhs.as_const_i64().is_none() {
                return Err(ExtractError::at(
                    "non-affine multiplication (neither operand is a constant)",
                    span,
                )
                .with_help("the language only allows multiplication by an integer constant"));
            }
            Ok(Expr::Mul(Box::new(lhs), Box::new(rhs)))
        }
        AffineExprKind::FloorDiv => {
            let lhs = affine_expr_to_dsl(expr.lhs(), dims, syms, span)?;
            let rhs = affine_expr_to_dsl(expr.rhs(), dims, syms, span)?;
            if rhs.as_const_i64().is_none() {
                return Err(ExtractError::at(
                    "floor division with a non-constant divisor is not supported",
                    span,
                ));
            }
            Ok(Expr::FloorDiv(Box::new(lhs), Box::new(rhs)))
        }
        AffineExprKind::Mod => Err(ExtractError::at(
            "the `mod` operator cannot be expressed in the target language",
            span,
        )
        .with_help("rewrite the access without modulo, or extract a tiled form that avoids it")),
        AffineExprKind::CeilDiv => Err(ExtractError::at(
            "the `ceildiv` operator cannot be expressed in the target language",
            span,
        )
        .with_help("the language only supports floor division (`/`)")),
    }
}

/// `0 -> "A"`, `25 -> "Z"`, `26 -> "arr26"`, ...
fn array_letter(index: usize) -> String {
    if index < 26 {
        char::from(b'A' + index as u8).to_string()
    } else {
        format!("arr{index}")
    }
}
