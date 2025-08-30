use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use indicatif::ParallelProgressIterator;
use melior::ir::{BlockLike, Module, OperationRef, RegionLike};
use palc::Parser;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use raffine::affine::{AffineExpr, AffineMap};
use raffine::tree::{Tree, ValID};
use raffine::{Context, DominanceInfo};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use tracing::{debug, info, trace};

struct CProgramEmitter<W: Write> {
    writer: BufWriter<W>,
    indent: usize,
}

impl<W: Write> CProgramEmitter<W> {
    fn new(writer: W) -> Self {
        CProgramEmitter {
            writer: BufWriter::new(writer),
            indent: 1,
        }
    }

    fn emit_indent(&mut self) -> Result<()> {
        for _ in 0..self.indent {
            write!(self.writer, "\t")?;
        }
        Ok(())
    }

    fn emit(mut self, tree: &Tree) -> Result<()> {
        writeln!(self.writer, "extern \"C\" void _start() {{")?;
        self.emit_tree(tree)?;
        self.emit_indent()?;

        // x86_64: exit(0) => eax=60 (sys_exit), edi=0, syscall
        #[cfg(target_arch = "x86_64")]
        {
            writeln!(
                self.writer,
                r#"asm volatile("xor %edi, %edi\n\tmov $60, %eax\n\tsyscall");"#
            )?;
        }

        // aarch64 (Linux): exit(0) => x8=93 (sys_exit), x0=0, svc #0
        #[cfg(target_arch = "aarch64")]
        {
            writeln!(
                self.writer,
                r#"asm volatile("mov x0, #0\n\tmov x8, #93\n\tsvc #0");"#
            )?;
        }

        Ok(writeln!(self.writer, "}}")?)
    }

    fn emit_tree(&mut self, tree: &Tree) -> Result<()> {
        match tree {
            Tree::For {
                lower_bound,
                upper_bound,
                lower_bound_operands,
                upper_bound_operands,
                step,
                ivar,
                body,
            } => {
                let ValID::IVar(ivar) = *ivar else {
                    return Err(anyhow!("expected ivar in for loop, found: {ivar}"));
                };
                self.emit_indent()?;
                write!(self.writer, "for (int ivar_{ivar} = ")?;
                self.emit_affine_map(lower_bound, lower_bound_operands)?;
                write!(self.writer, "; ivar_{ivar} < ")?;
                self.emit_affine_map(upper_bound, upper_bound_operands)?;
                write!(self.writer, "; ivar_{ivar} += {step}",)?;
                writeln!(self.writer, ") {{")?;
                self.indent += 1;
                self.emit_tree(body)?;
                self.indent -= 1;
                self.emit_indent()?;
                writeln!(self.writer, "}}")?;
            }
            Tree::Block(trees) => {
                self.emit_indent()?;
                writeln!(self.writer, "{{")?;
                self.indent += 1;
                for tree in trees.iter() {
                    self.emit_tree(tree)?;
                    writeln!(self.writer)?;
                }
                self.indent -= 1;
                self.emit_indent()?;
                writeln!(self.writer, "}}")?;
            }
            Tree::Access {
                memref,
                map,
                operands,
                ..
            } => {
                let ValID::Memref(array) = *memref else {
                    return Err(anyhow!("expected memref access, found: {memref}"));
                };
                self.emit_indent()?;
                self.emit_access(array, operands, *map)?;
            }
            Tree::If {
                condition,
                operands,
                then,
                r#else,
            } => {
                self.emit_indent()?;
                write!(self.writer, "if ")?;
                self.emit_integer_set(condition, operands)?;
                writeln!(self.writer, " {{")?;
                self.indent += 1;
                self.emit_tree(then)?;
                self.indent -= 1;
                self.emit_indent()?;
                writeln!(self.writer, "}}")?;
                if let Some(else_block) = r#else {
                    self.emit_indent()?;
                    writeln!(self.writer, "else {{")?;
                    self.indent += 1;
                    self.emit_tree(else_block)?;
                    self.indent -= 1;
                    self.emit_indent()?;
                    writeln!(self.writer, "}}")?;
                }
            }
        }
        Ok(())
    }

    fn emit_integer_set(
        &mut self,
        set: &raffine::affine::IntegerSet,
        operands: &[ValID],
    ) -> Result<()> {
        write!(self.writer, "(")?;
        for i in 0..set.num_constraints() {
            if i > 0 {
                write!(self.writer, " && ")?;
            }
            let expr = set.get_constraint(i as isize);
            write!(self.writer, "((")?;
            let operands = operands
                .iter()
                .map(|&operand| match operand {
                    ValID::IVar(x) | ValID::Symbol(x) | ValID::Memref(x) => x,
                })
                .collect::<Vec<_>>();
            self.emit_affine_expr(expr, &operands)?;
            if set.is_constraint_equal(i as isize) {
                write!(self.writer, ") == 0)")?;
            } else {
                write!(self.writer, ") >= 0)")?; // need to confirm this
            }
        }
        write!(self.writer, ")")?;
        Ok(())
    }

    fn emit_affine_map(
        &mut self,
        map: &raffine::affine::AffineMap,
        operands: &[ValID],
    ) -> Result<()> {
        let operands = operands
            .iter()
            .map(|&operand| match operand {
                ValID::IVar(x) | ValID::Symbol(x) | ValID::Memref(x) => x,
            })
            .collect::<Vec<_>>();
        for i in 0..map.num_results() {
            let expr = map
                .get_result_expr(i as isize)
                .ok_or_else(|| anyhow!("invalid affine map: result {i} does not exist in map"))?;
            if i > 0 {
                write!(self.writer, "][")?;
            }
            self.emit_affine_expr(expr, &operands)?;
        }
        Ok(())
    }

    fn emit_access(&mut self, array: usize, operands: &[ValID], map: AffineMap) -> Result<()> {
        write!(self.writer, "{{ ARRAY_{array}[")?;
        self.emit_affine_map(&map, operands)?;
        write!(
            self.writer,
            r#"] = 0; __asm__ __volatile__ ("" ::: "memory"); }}"#
        )?;
        Ok(())
    }

    fn emit_affine_expr(&mut self, expr: AffineExpr, operands: &[usize]) -> Result<()> {
        match expr.get_kind() {
            raffine::affine::AffineExprKind::Add
            | raffine::affine::AffineExprKind::FloorDiv
            | raffine::affine::AffineExprKind::Mul
            | raffine::affine::AffineExprKind::Mod => {
                let operator = match expr.get_kind() {
                    raffine::affine::AffineExprKind::Add => "+",
                    raffine::affine::AffineExprKind::FloorDiv => "/",
                    raffine::affine::AffineExprKind::Mul => "*",
                    raffine::affine::AffineExprKind::Mod => "%",
                    _ => unreachable!(),
                };
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow!("addition should have lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow!("addition should have rhs"))?;
                write!(self.writer, "(")?;
                self.emit_affine_expr(lhs, operands)?;
                write!(self.writer, " {operator}")?;
                self.emit_affine_expr(rhs, operands)?;
                write!(self.writer, ")")?;
            }

            raffine::affine::AffineExprKind::Dim | raffine::affine::AffineExprKind::Symbol => {
                let operand = expr
                    .get_position()
                    .ok_or_else(|| anyhow!("dimension expression should have position"))?
                    as usize;
                let target = *operands
                    .get(operand)
                    .ok_or_else(|| anyhow!("invalid operand index"))?;
                let prefix = if matches!(expr.get_kind(), raffine::affine::AffineExprKind::Symbol) {
                    "SYM"
                } else {
                    "ivar"
                };
                write!(self.writer, "{prefix}_{target}",)?;
            }
            raffine::affine::AffineExprKind::CeilDiv => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow!("ceil division should have lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow!("ceil division should have rhs"))?;
                let decomposed = (lhs + lhs % rhs) / rhs;
                self.emit_affine_expr(decomposed, operands)?;
            }
            raffine::affine::AffineExprKind::Constant => {
                let value = expr
                    .get_value()
                    .ok_or_else(|| anyhow!("constant expression should have value"))?;
                if value < 0 {
                    write!(self.writer, "(")?;
                }
                write!(self.writer, "{value}")?;
                if value < 0 {
                    write!(self.writer, ")")?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Parser)]
struct Cli {
    #[arg(long, short)]
    input: PathBuf,

    /// name of the target function to extract
    /// if not specified, the program will try to find first function
    #[arg(short = 'f', long)]
    target_function: Option<String>,

    /// target affine loop attribute
    /// if not specified, the program will try to find first affine loop in the function
    #[arg(short = 'l', long)]
    target_affine_loop: Option<String>,

    /// valgrind path
    #[arg(long, default_value = "valgrind")]
    valgrind_path: String,

    /// block size (D1)
    #[arg(long, default_value = "64", short = 'B')]
    d1_block_size: usize,

    /// associativity of the cache
    /// if not specified, the program will assume fully associative
    #[arg(long, short = 'A')]
    d1_associativity: Option<usize>,

    /// number of blocks upper bound
    #[arg(long, short = 'C')]
    d1_cache_size: usize,

    /// block size (LL)
    #[arg(long, default_value = "64", short = 'b')]
    ll_block_size: usize,

    /// associativity of the second level cache
    #[arg(long, short = 'a')]
    ll_associativity: usize,

    /// cache size of the second level cache
    #[arg(long, short = 'c')]
    ll_cache_size: usize,

    /// Do batched run from block size to cache size, stepping by block size
    #[arg(long)]
    batched: bool,

    /// Database file to store the results
    #[arg(long, short = 'd', default_value = "/tmp/cachegrind.db")]
    database: PathBuf,
}

#[derive(Debug, Clone)]
struct Record {
    program: String,
    d1_cache_size: usize,
    d1_associativity: usize,
    d1_block_size: usize,
    ll_associativity: usize,
    ll_cache_size: usize,
    ll_block_size: usize,
    d1_miss_count: usize,
    ll_miss_count: usize,
    total_access: usize,
    process_time: usize,
}

fn extract_target<'ctx>(
    module: &'ctx Module<'ctx>,
    options: &Cli,
    context: &'ctx Context,
    dom: &'ctx DominanceInfo<'ctx>,
) -> anyhow::Result<&'ctx Tree<'ctx>> {
    let body = module.body();
    fn locate_function<'a, 'b, F>(
        cursor: Option<OperationRef<'a, 'b>>,
        options: &'_ Cli,
        conti: F,
    ) -> anyhow::Result<&'a Tree<'a>>
    where
        F: for<'c> FnOnce(OperationRef<'a, 'c>) -> anyhow::Result<&'a Tree<'a>>,
    {
        let Some(op) = cursor else {
            return Err(anyhow!("No operation found"));
        };
        if op.name().as_string_ref().as_str()? == "func.func" {
            if let Some(name) = options.target_function.as_deref() {
                let sym_name = op.attribute("sym_name")?;
                debug!("Checking function: {}", sym_name);
                if sym_name.to_string().trim_matches('"') == name {
                    debug!("Found target function: {}", name);
                    return conti(op);
                }
            } else {
                return conti(op);
            }
        }
        locate_function(op.next_in_block(), options, conti)
    }
    fn locate_loop<'a, 'b, F>(
        cursor: Option<OperationRef<'a, 'b>>,
        options: &'_ Cli,
        conti: F,
    ) -> anyhow::Result<&'a Tree<'a>>
    where
        F: for<'c> FnOnce(OperationRef<'a, 'c>) -> anyhow::Result<&'a Tree<'a>>,
    {
        let Some(op) = cursor else {
            return Err(anyhow!("No operation found"));
        };
        if op.name().as_string_ref().as_str()? == "affine.for" {
            if let Some(name) = options.target_affine_loop.as_deref() {
                if op.has_attribute(name) {
                    debug!("Found target affine loop: {}", name);
                    return conti(op);
                }
            } else {
                return conti(op);
            }
        }
        locate_loop(op.next_in_block(), options, conti)
    }

    let cursor = body.first_operation();
    locate_function(cursor, options, move |func| {
        let cursor = func
            .region(0)?
            .first_block()
            .ok_or_else(|| anyhow!("function does not have block"))?
            .first_operation();
        locate_loop(cursor, options, move |for_loop| {
            Ok(context.build_tree(for_loop, dom)?)
        })
    })
}

impl Record {
    fn insert(&self, database: &Pool<SqliteConnectionManager>) {
        while pool_get_retry(database)
            .execute(
                r#"INSERT INTO records (
                    program,
                    d1_block_size,
                    d1_associativity,
                    d1_cache_size,
                    ll_block_size,
                    ll_associativity,
                    ll_cache_size,
                    d1_miss_count,
                    ll_miss_count,
                    total_access,
                    process_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(program, d1_block_size, d1_associativity, d1_cache_size, ll_block_size, ll_associativity, ll_cache_size) DO UPDATE SET
                    d1_miss_count = excluded.d1_miss_count,
                    ll_miss_count = excluded.ll_miss_count,
                    total_access = excluded.total_access,
                    process_time = excluded.process_time"#,
                rusqlite::params![
                    self.program,
                    self.d1_block_size,
                    self.d1_associativity,
                    self.d1_cache_size,
                    self.ll_block_size,
                    self.ll_associativity,
                    self.ll_cache_size,
                    self.d1_miss_count,
                    self.ll_miss_count,
                    self.total_access,
                    self.process_time,
                ],
            )
            .is_err()
        {}
    }
}

fn pool_get_retry(
    pool: &Pool<SqliteConnectionManager>,
) -> r2d2::PooledConnection<SqliteConnectionManager> {
    loop {
        match pool.get() {
            Ok(conn) => return conn,
            Err(e) => {
                debug!("Failed to get connection from pool: {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let rcontext = Context::new();
    let args = Cli::parse();
    let source = std::fs::read_to_string(&args.input).unwrap();
    let module = Module::parse(rcontext.mlir_context(), &source).unwrap();
    let dom = DominanceInfo::new(&module);
    let target_tree = extract_target(&module, &args, &rcontext, &dom).unwrap();
    trace!("Extracted target tree: {:#}", target_tree);
    let workdir = tempfile::tempdir().unwrap();
    let program_path = workdir.path().join("program.cxx");
    let mut program_file = std::fs::File::create(&program_path).unwrap();
    if let Ok(attr) = module.as_operation().attribute("simulation.prologue") {
        let string = attr.to_string();
        let unescaped = unescaper::unescape(string.trim_matches('"')).unwrap();
        write!(program_file, "// Simulation Prologue:\n{unescaped}\n\n").unwrap();
    } else {
        trace!("No simulation prologue found");
    }
    let emitter = CProgramEmitter::new(program_file);
    emitter.emit(target_tree).unwrap();
    info!("C program emitted:{}", {
        std::fs::read_to_string(&program_path).unwrap()
    });
    let output_path = workdir.path().join("test.exe");
    std::process::Command::new("clang++")
        .arg(&program_path)
        .arg("-o")
        .arg(&output_path)
        .args([
            "-static",
            "-nostdlib",
            "-fno-stack-protector",
            "-fno-pic",
            "-Os",
            "-ffreestanding",
        ])
        .current_dir(workdir.path())
        .status()
        .expect("Failed to compile C program");
    let manager = r2d2_sqlite::SqliteConnectionManager::file(&args.database);
    let pool = r2d2::Pool::new(manager).unwrap();
    pool.get()
        .unwrap()
        .execute(
            r#"CREATE TABLE IF NOT EXISTS records (
            program TEXT NOT NULL,
            d1_block_size INTEGER NOT NULL,
            d1_associativity INTEGER NOT NULL,
            d1_cache_size INTEGER NOT NULL,
            ll_block_size INTEGER NOT NULL,
            ll_associativity INTEGER NOT NULL,
            ll_cache_size INTEGER NOT NULL,
            d1_miss_count INTEGER NOT NULL,
            ll_miss_count INTEGER NOT NULL,
            total_access INTEGER NOT NULL,
            process_time INTEGER NOT NULL,
            PRIMARY KEY (program, d1_block_size, d1_associativity, d1_cache_size, ll_block_size, ll_associativity, ll_cache_size)
        )"#,
            (),
        )
        .unwrap();

    let program = args
        .input
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let range = if args.batched {
        let block_size = args.d1_block_size;
        let cache_size = args.d1_cache_size;
        (block_size..=cache_size)
            .step_by(block_size)
            .collect::<Vec<_>>()
    } else {
        vec![args.d1_cache_size]
    };
    range.into_par_iter().progress().for_each(|cache_size| {
        let associativity = args
            .d1_associativity
            .unwrap_or_else(|| cache_size / args.d1_block_size);
        let d1_string = format!(
            "--D1={},{},{}",
            cache_size, associativity, args.d1_block_size,
        );
        let ll_string = format!(
            "--LL={},{},{}",
            args.ll_cache_size, args.ll_associativity, args.ll_block_size,
        );
        let start = std::time::Instant::now();
        let output = std::process::Command::new(&args.valgrind_path)
            .arg("--tool=cachegrind")
            .arg("--cache-sim=yes")
            .arg("-v")
            .arg(d1_string)
            .arg(ll_string)
            .arg(&output_path)
            .current_dir(workdir.path())
            .output()
            .unwrap();
        let process_time = start.elapsed().as_nanos() as usize;
        let output = String::from_utf8_lossy(&output.stderr);
        info!("Valgrind output:\n{output}");
        let mut total_access = 0usize;
        let mut d1_miss_count = 0usize;
        let mut ll_miss_count = 0usize;
        for line in output.lines() {
            if line.contains("D refs:") {
                if let Some(value) = line.split(':').nth(1).and_then(|s| s.split('(').next()) {
                    total_access = value.trim().replace(",", "").parse().unwrap_or(0);
                }
            } else if line.contains("D1  misses:") {
                if let Some(value) = line.split(':').nth(1).and_then(|s| s.split('(').next()) {
                    d1_miss_count = value.trim().replace(",", "").parse().unwrap_or(0);
                }
            } else if line.contains("LLd misses:")
                && let Some(value) = line.split(':').nth(1).and_then(|s| s.split('(').next())
            {
                ll_miss_count = value.trim().replace(",", "").parse().unwrap_or(0);
            }
        }

        let record = Record {
            program: program.clone(),
            d1_cache_size: cache_size,
            d1_associativity: associativity,
            d1_block_size: args.d1_block_size,
            ll_associativity: args.ll_associativity,
            ll_cache_size: args.ll_cache_size,
            ll_block_size: args.ll_block_size,
            d1_miss_count,
            ll_miss_count,
            total_access,
            process_time,
        };
        record.insert(&pool);
    });
}
