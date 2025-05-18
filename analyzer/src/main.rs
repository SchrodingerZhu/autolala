use anyhow::{Ok, anyhow};
use barvinok::ContextRef as BContext;
use clap::Parser;
use melior::Context as MContext;
use melior::ir::{BlockLike, Module, OperationRef, RegionLike};
use raffine::Context as RContext;
use raffine::affine::{AffineExpr, AffineMap};
use raffine::{DominanceInfo, tree::Tree};
use std::{collections::HashMap, io::Read, path::PathBuf};
use symbolica::coefficient;
use symbolica::domains::rational_polynomial::RationalPolynomial;
use tracing::{debug, error};
use tracing_subscriber::EnvFilter;
mod utils;
use raffine::tree::ValID;
use symbolica::domains::{Field, rational_polynomial::RationalPolynomialField};

use utils::{Poly, convert_affine_map, create_table};
struct AnalysisContext<'a> {
    rcontext: RContext,
    bcontext: BContext<'a>,
}

impl<'a> AnalysisContext<'a>
where
    Self: 'a,
{
    fn start<F, R>(f: F) -> R
    where
        F: for<'x> FnOnce(AnalysisContext<'x>) -> R,
    {
        let rcontext = RContext::new();
        barvinok::Context::new().scope(move |bcontext| {
            let context = AnalysisContext { rcontext, bcontext };
            f(context)
        })
    }
    fn rcontext(&self) -> &RContext {
        &self.rcontext
    }
    fn bcontext(&self) -> BContext<'a> {
        self.bcontext
    }
    fn mcontext(&self) -> &MContext {
        self.rcontext.mlir_context()
    }
}

#[derive(Debug, Parser)]
struct Options {
    /// The input file to process
    /// If not specified, the input will be read from stdin
    #[clap(short, long)]
    input: Option<PathBuf>,

    /// The output file to write
    /// If not specified, the output will be written to stdout
    #[clap(short, long)]
    output: Option<PathBuf>,

    /// name of the target function to extract
    /// if not specified, the program will try to find first function
    #[clap(short = 'f', long)]
    target_function: Option<String>,

    /// target affine loop attribute
    /// if not specified, the program will try to find first affine loop in the function
    #[clap(short = 'l', long)]
    target_affine_loop: Option<String>,
}

fn extract_target<'bctx, 'ctx, 'dom>(
    module: &'ctx Module<'ctx>,
    options: &Options,
    context: &'ctx AnalysisContext<'bctx>,
    dom: &'ctx DominanceInfo<'ctx>,
) -> anyhow::Result<&'ctx Tree<'ctx>>
where
    'bctx: 'ctx,
{
    let body = module.body();
    fn locate_function<'a, 'b, F>(
        cursor: Option<OperationRef<'a, 'b>>,
        options: &'_ Options,
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
        options: &'_ Options,
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
            Ok(context.rcontext().build_tree(for_loop, dom)?)
        })
    })
}

fn isize_to_poly<'a>(value: isize, context: &AnalysisContext<'a>) -> Poly {
    let expr = AffineExpr::new_constant(context.rcontext().mlir_context(), value as i64);
    let map = AffineMap::new(
        context.rcontext().mlir_context(),
        0, // num_dims
        0, // num_symbols
        &[expr],
    );
    let step_converted = convert_affine_map(map, &[]).unwrap();
    step_converted[0].clone()
}

// reuse interval distribution without block
fn get_reuse_interval_distribution<'a, 'b: 'a>(
    ri_dist: &mut Vec<(Poly, Poly)>,
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
) {
    match tree {
        Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands,
            upper_bound_operands,
            body,
            ivar,
            step,
        } => {
            let lower_bound_converted =
                convert_affine_map(*lower_bound, lower_bound_operands).unwrap();
            let upper_bound_converted =
                convert_affine_map(*upper_bound, upper_bound_operands).unwrap();
            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);
            let tmp = upper_bound_converted[0].clone() - lower_bound_converted[0].clone();
            let trip_count = field.div(&tmp, &isize_to_poly(*step, context));
            if reuse_factors.is_empty() {
                reuse_factors.insert(0x5eabed, isize_to_poly(1, context));
            }

            for (_key, value) in reuse_factors.iter_mut() {
                *value = &*value * &trip_count.clone();
            }

            let ValID::IVar(id) = ivar else {
                unreachable!("not possible ")
            };
            reuse_factors.insert(*id, isize_to_poly(1, context));
            trip_counts.insert(*id, trip_count.clone());

            get_reuse_interval_distribution(
                ri_dist,
                body,
                reuse_factors,
                trip_counts,
                ref_count,
                context,
            )
        }
        Tree::Block(trees) => {
            for subtree in trees.iter() {
                get_reuse_interval_distribution(
                    ri_dist,
                    subtree,
                    reuse_factors,
                    trip_counts,
                    trees.len(),
                    context,
                );
            }
        }
        Tree::Access { map, operands, .. } => {
            let mut reference_vector = vec![0; reuse_factors.len()];
            let converted_map = convert_affine_map(*map, operands);

            if let polys = converted_map.unwrap() {
                for poly in polys {
                    for (_var_pos, var) in poly.numerator.variables.iter().enumerate() {
                        if var.to_string().starts_with('i') {
                            let index_str = &var.to_string()[1..];
                            let index = index_str.parse::<usize>().unwrap();
                            reference_vector[index + 1] = 1;
                        }
                    }
                }
            }

            let mut shrinked_ref_vec = vec![];
            for i in 0..reference_vector.len() - 1 {
                if reference_vector[i] != reference_vector[i + 1] {
                    if i == 0 {
                        shrinked_ref_vec.push(0x5eabed);
                    } else {
                        shrinked_ref_vec.push(i - 1);
                    }
                }
            }
            if reference_vector[reference_vector.len() - 1] == 0 {
                shrinked_ref_vec.push(reference_vector.len() - 2);
            }

            let mut ri_value = isize_to_poly(0, context);
            let mut ri_portion = isize_to_poly(1, context);
            let mut ri_portion_sum = isize_to_poly(0, context);
            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);

            for i in 0..reference_vector.len() - 1 {
                if reference_vector[i + 1] == 0 {
                    ri_portion = field.div(&ri_portion, &trip_counts.get(&i).unwrap());
                }
            }
            let mut coefficient = 1;
            let n_ref = isize_to_poly(ref_count as isize, context);

            for (place, i) in (shrinked_ref_vec.iter().rev()).enumerate() {
                let tmp = &*reuse_factors.get(i).unwrap() * &isize_to_poly(coefficient, context);
                ri_value = &ri_value + &tmp;
                if coefficient == 1 {
                    if place != shrinked_ref_vec.len() - 1 {
                        ri_dist.push((
                            &ri_value * &n_ref,
                            field.div(&(&ri_portion - &ri_portion_sum), &n_ref),
                        ));
                        ri_portion_sum = &ri_portion_sum + &ri_portion;
                        ri_portion = &ri_portion * trip_counts.get(i).unwrap()
                    } else {
                        ri_dist.push((
                            &ri_value * &n_ref,
                            field.div(&(&isize_to_poly(1, context) - &ri_portion_sum), &n_ref),
                        ));
                    }
                }
                coefficient *= -1;
            }
        }
        Tree::If { .. } => {}
    }
}

fn main_entry() -> anyhow::Result<()> {
    let options = Options::parse();

    let mut reader = match options.input.as_ref() {
        Some(path) => {
            debug!("Opening input file: {}", path.display());
            Box::new(std::fs::File::open(path)?) as Box<dyn Read>
        }
        None => {
            debug!("Reading from stdin");
            Box::new(std::io::stdin()) as Box<dyn Read>
        }
    };

    let _writer = match options.output.as_ref() {
        Some(path) => {
            debug!("Opening output file: {}", path.display());
            Box::new(std::fs::File::create(path)?) as Box<dyn std::io::Write>
        }
        None => {
            debug!("Writing to stdout");
            Box::new(std::io::stdout()) as Box<dyn std::io::Write>
        }
    };
    AnalysisContext::start(|context| {
        let mut source = String::new();
        let bytes = reader.read_to_string(&mut source)?;

        debug!("Read {} bytes", bytes);

        let module = Module::parse(context.mcontext(), &source)
            .ok_or_else(|| anyhow!("Failed to parse module"))?;

        debug!("Parsed module: {}", module.as_operation());

        let dom = DominanceInfo::new(&module);

        let tree = extract_target(&module, &options, &context, &dom)?;
        debug!("Extracted tree: {}", tree);

        let total_space = utils::get_space(&context, tree)?;

        debug!("Total space: {:?}", total_space);

        let nesting_level = utils::get_nesting_level(tree);
        debug!("Nesting level: {:?}", nesting_level);

        //  utils::walk_tree_print_converted_affine_map(tree, 0)?;
        let mut ri_dist = vec![];
        let mut rf = HashMap::new();
        let mut tc = HashMap::new();
        get_reuse_interval_distribution(&mut ri_dist, tree, &mut rf, &mut tc, 0, &context);
        let table = create_table(&ri_dist);

        println!("Reuse interval distribution:\n{}", table);
        Ok(())
    })
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .init();
    let res = main_entry();
    if let Err(e) = res {
        let bt = e.backtrace();
        error!("{:#}", e);
        match bt.status() {
            std::backtrace::BacktraceStatus::Disabled => {
                error!("Backtrace is disabled -- rerun with RUST_BACKTRACE=1 to get a backtrace")
            }
            std::backtrace::BacktraceStatus::Captured => error!("Stack backtrace:\n{bt}"),
            _ => panic!("unsupported backtrace"),
        }
        std::process::exit(1);
    }
}
