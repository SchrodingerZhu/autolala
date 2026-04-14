use crate::ast::{Access, ArrayDecl, Block, Comparison, Expr, ForLoop, IfStmt, Program, Stmt};
use crate::error::{DmdError, DmdResult};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ArrayInfo {
    pub id: usize,
    pub name: String,
    pub rank: usize,
    pub extents: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub struct ParallelLoopInfo {
    pub var: String,
    pub threads: i64,
}

#[derive(Debug, Clone)]
pub struct SemanticProgram {
    pub params: Vec<String>,
    pub arrays: Vec<ArrayInfo>,
    pub body: Block,
    pub parallel_loop: Option<ParallelLoopInfo>,
}

impl SemanticProgram {
    pub fn array_id(&self, name: &str) -> Option<usize> {
        self.arrays
            .iter()
            .find(|array| array.name == name)
            .map(|array| array.id)
    }

    pub fn array_rank(&self, name: &str) -> Option<usize> {
        self.arrays
            .iter()
            .find(|array| array.name == name)
            .map(|array| array.rank)
    }

    pub fn max_access_rank(&self) -> usize {
        fn block_max(block: &Block) -> usize {
            block
                .statements
                .iter()
                .map(stmt_max)
                .max()
                .unwrap_or_default()
        }

        fn stmt_max(stmt: &Stmt) -> usize {
            match stmt {
                Stmt::For(for_loop) => block_max(&for_loop.body),
                Stmt::If(if_stmt) => {
                    let then_rank = block_max(&if_stmt.then_branch);
                    let else_rank = if_stmt
                        .else_branch
                        .as_ref()
                        .map(block_max)
                        .unwrap_or_default();
                    then_rank.max(else_rank)
                }
                Stmt::Access(access) => access.indices.len(),
            }
        }

        block_max(&self.body)
    }

    pub fn thread_count(&self) -> Option<i64> {
        self.parallel_loop.as_ref().map(|parallel| parallel.threads)
    }
}

pub fn validate_program(program: Program) -> DmdResult<SemanticProgram> {
    let mut params_seen = HashSet::new();
    for param in &program.params {
        if !params_seen.insert(param.clone()) {
            return Err(DmdError::semantic(format!(
                "duplicate parameter declaration `{param}`"
            )));
        }
    }

    let mut arrays = Vec::with_capacity(program.arrays.len());
    let mut array_names = HashSet::new();
    for (id, array) in program.arrays.iter().enumerate() {
        validate_array_decl(array, &params_seen)?;
        if !array_names.insert(array.name.clone()) {
            return Err(DmdError::semantic(format!(
                "duplicate array declaration `{}`",
                array.name
            )));
        }

        arrays.push(ArrayInfo {
            id,
            name: array.name.clone(),
            rank: array.extents.len(),
            extents: array.extents.clone(),
        });
    }

    let array_map = arrays
        .iter()
        .map(|array| (array.name.as_str(), array.rank))
        .collect::<HashMap<_, _>>();

    let mut loop_scope = Vec::new();
    let mut parallel_loop = None;
    validate_block(
        &program.body,
        &params_seen,
        &array_map,
        &mut loop_scope,
        &mut parallel_loop,
    )?;

    if arrays.is_empty() {
        return Err(DmdError::semantic(
            "at least one array declaration is required",
        ));
    }

    if max_access_rank(&program.body) == 0 {
        return Err(DmdError::semantic(
            "the program does not contain any accesses",
        ));
    }

    Ok(SemanticProgram {
        params: program.params,
        arrays,
        body: program.body,
        parallel_loop,
    })
}

fn validate_array_decl(array: &ArrayDecl, params: &HashSet<String>) -> DmdResult<()> {
    if array.extents.is_empty() {
        return Err(DmdError::semantic(format!(
            "array `{}` must declare at least one extent",
            array.name
        )));
    }

    for extent in &array.extents {
        validate_expr(extent, params, &[])?;
    }

    Ok(())
}

fn validate_block(
    block: &Block,
    params: &HashSet<String>,
    arrays: &HashMap<&str, usize>,
    loop_scope: &mut Vec<String>,
    parallel_loop: &mut Option<ParallelLoopInfo>,
) -> DmdResult<()> {
    for stmt in &block.statements {
        match stmt {
            Stmt::For(for_loop) => {
                validate_for(for_loop, params, arrays, loop_scope, parallel_loop)?
            }
            Stmt::If(if_stmt) => validate_if(if_stmt, params, arrays, loop_scope, parallel_loop)?,
            Stmt::Access(access) => validate_access(access, params, arrays, loop_scope)?,
        }
    }

    Ok(())
}

fn validate_for(
    for_loop: &ForLoop,
    params: &HashSet<String>,
    arrays: &HashMap<&str, usize>,
    loop_scope: &mut Vec<String>,
    parallel_loop: &mut Option<ParallelLoopInfo>,
) -> DmdResult<()> {
    if for_loop.step <= 0 {
        return Err(DmdError::semantic(format!(
            "loop `{}` must have a positive step",
            for_loop.var
        )));
    }

    if params.contains(&for_loop.var) || loop_scope.iter().any(|name| name == &for_loop.var) {
        return Err(DmdError::semantic(format!(
            "loop variable `{}` shadows an existing binding",
            for_loop.var
        )));
    }

    validate_expr(&for_loop.lower, params, loop_scope)?;
    validate_expr(&for_loop.upper, params, loop_scope)?;
    if let Some(spec) = &for_loop.parallel {
        validate_expr(&spec.threads, params, loop_scope)?;
        let Some(threads) = spec.threads.as_const_i64() else {
            return Err(DmdError::semantic(
                "parallel loop thread count must be an input-independent constant",
            ));
        };
        if threads < 2 {
            return Err(DmdError::semantic(
                "parallel loop thread count must be at least two",
            ));
        }
        if parallel_loop.is_some() {
            return Err(DmdError::semantic(
                "only one parallel loop is currently supported",
            ));
        }
        *parallel_loop = Some(ParallelLoopInfo {
            var: for_loop.var.clone(),
            threads,
        });
    }

    loop_scope.push(for_loop.var.clone());
    let result = validate_block(&for_loop.body, params, arrays, loop_scope, parallel_loop);
    loop_scope.pop();
    result
}

fn validate_if(
    if_stmt: &IfStmt,
    params: &HashSet<String>,
    arrays: &HashMap<&str, usize>,
    loop_scope: &mut Vec<String>,
    parallel_loop: &mut Option<ParallelLoopInfo>,
) -> DmdResult<()> {
    if if_stmt.conditions.is_empty() {
        return Err(DmdError::semantic(
            "if statements require at least one affine comparison",
        ));
    }

    for condition in &if_stmt.conditions {
        validate_comparison(condition, params, loop_scope)?;
    }

    validate_block(
        &if_stmt.then_branch,
        params,
        arrays,
        loop_scope,
        parallel_loop,
    )?;
    if let Some(else_branch) = &if_stmt.else_branch {
        validate_block(else_branch, params, arrays, loop_scope, parallel_loop)?;
    }

    Ok(())
}

fn validate_access(
    access: &Access,
    params: &HashSet<String>,
    arrays: &HashMap<&str, usize>,
    loop_scope: &[String],
) -> DmdResult<()> {
    let Some(expected_rank) = arrays.get(access.array.as_str()) else {
        return Err(DmdError::semantic(format!(
            "access references undeclared array `{}`",
            access.array
        )));
    };

    if access.indices.len() != *expected_rank {
        return Err(DmdError::semantic(format!(
            "array `{}` expects {} indices, got {}",
            access.array,
            expected_rank,
            access.indices.len()
        )));
    }

    for index in &access.indices {
        validate_expr(index, params, loop_scope)?;
    }

    Ok(())
}

fn validate_comparison(
    comparison: &Comparison,
    params: &HashSet<String>,
    loop_scope: &[String],
) -> DmdResult<()> {
    validate_expr(&comparison.lhs, params, loop_scope)?;
    validate_expr(&comparison.rhs, params, loop_scope)?;
    Ok(())
}

fn validate_expr(expr: &Expr, params: &HashSet<String>, loop_scope: &[String]) -> DmdResult<()> {
    match expr {
        Expr::Int(_) => Ok(()),
        Expr::Var(name) => {
            if params.contains(name) || loop_scope.iter().any(|item| item == name) {
                Ok(())
            } else {
                Err(DmdError::semantic(format!(
                    "unknown identifier `{name}` in affine expression"
                )))
            }
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            validate_expr(lhs, params, loop_scope)?;
            validate_expr(rhs, params, loop_scope)
        }
        Expr::Mul(lhs, rhs) => {
            validate_expr(lhs, params, loop_scope)?;
            validate_expr(rhs, params, loop_scope)?;
            if lhs.as_const_i64().is_none() && rhs.as_const_i64().is_none() {
                return Err(DmdError::semantic(
                    "non-affine multiplication is not supported; one side must be constant",
                ));
            }
            Ok(())
        }
        Expr::FloorDiv(lhs, rhs) => {
            validate_expr(lhs, params, loop_scope)?;
            validate_expr(rhs, params, loop_scope)?;
            let Some(divisor) = rhs.as_const_i64() else {
                return Err(DmdError::semantic(
                    "floor division requires a constant divisor",
                ));
            };
            if divisor == 0 {
                return Err(DmdError::semantic("division by zero in affine expression"));
            }
            Ok(())
        }
        Expr::Neg(expr) => validate_expr(expr, params, loop_scope),
    }
}

fn max_access_rank(block: &Block) -> usize {
    block
        .statements
        .iter()
        .map(|stmt| match stmt {
            Stmt::For(for_loop) => max_access_rank(&for_loop.body),
            Stmt::If(if_stmt) => {
                let then_rank = max_access_rank(&if_stmt.then_branch);
                let else_rank = if_stmt
                    .else_branch
                    .as_ref()
                    .map(max_access_rank)
                    .unwrap_or_default();
                then_rank.max(else_rank)
            }
            Stmt::Access(access) => access.indices.len(),
        })
        .max()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::validate_program;
    use crate::parse_program;

    #[test]
    fn rejects_non_affine_multiplication() {
        let source = r#"
params N;
array A[N];

for i in 0 .. N {
    read A[i * i];
}
"#;
        let program = parse_program(source).expect("parser should succeed");
        let error = validate_program(program)
            .expect_err("semantic validation should reject nonlinear terms");
        assert!(format!("{error}").contains("non-affine multiplication"));
    }

    #[test]
    fn accepts_affine_program() {
        let source = r#"
params N, M;
array A[N, M];

for i in 0 .. N {
    for j in 0 .. M step 2 {
        read A[i, j];
    }
}
"#;
        let program = parse_program(source).expect("parser should succeed");
        let model = validate_program(program).expect("semantic validation should succeed");
        assert_eq!(model.max_access_rank(), 2);
        assert_eq!(model.array_rank("A"), Some(2));
        assert!(model.parallel_loop.is_none());
    }

    #[test]
    fn accepts_parallel_loop_with_constant_thread_count() {
        let source = r#"
params N;
array A[N];

parallel(2 * 2) for i in 0 .. N {
    read A[i];
}
"#;
        let program = parse_program(source).expect("parser should succeed");
        let model = validate_program(program).expect("semantic validation should succeed");
        assert_eq!(model.thread_count(), Some(4));
        assert_eq!(
            model
                .parallel_loop
                .as_ref()
                .map(|parallel| parallel.var.as_str()),
            Some("i")
        );
    }

    #[test]
    fn rejects_parallel_loop_with_symbolic_thread_count() {
        let source = r#"
params N, T;
array A[N];

parallel(T) for i in 0 .. N {
    read A[i];
}
"#;
        let program = parse_program(source).expect("parser should succeed");
        let error = validate_program(program)
            .expect_err("semantic validation should reject symbolic thread counts");
        assert!(format!("{error}").contains("input-independent constant"));
    }
}
