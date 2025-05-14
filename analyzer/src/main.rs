use anyhow::{Ok, anyhow};
use barvinok::ContextRef as BContext;
use clap::Parser;
use melior::Context as MContext;
use melior::ir::{BlockLike, Module, OperationRef, RegionLike};
use raffine::Context as RContext;
use raffine::{DominanceInfo, tree::Tree};
use std::{io::Read, path::PathBuf};
use tracing::{debug, error};
use tracing_subscriber::EnvFilter;
mod isl;
mod utils;

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

    #[allow(unused)]
    let writer = match options.output.as_ref() {
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
        let context = &context;
        let mut source = String::new();
        let bytes = reader.read_to_string(&mut source)?;

        debug!("Read {} bytes", bytes);

        let module = Module::parse(context.mcontext(), &source)
            .ok_or_else(|| anyhow!("Failed to parse module"))?;

        debug!("Parsed module: {}", module.as_operation());

        let dom = DominanceInfo::new(&module);

        let tree = extract_target(&module, &options, context, &dom)?;

        debug!("Extracted tree: {}", tree);

        let total_space = isl::get_space(context, tree)?;

        debug!("Total space: {:?}", total_space);

        let nesting_level = utils::get_nesting_level(tree);
        debug!("Nesting level: {:?}", nesting_level);

        utils::walk_tree_print_converted_affine_map(tree, 0)?;

        if std::env::var("ISL_DEBUG").is_ok() {
            let space = isl::get_timestamp_space(
                total_space.get_dim(barvinok::DimType::Param)? as usize,
                0,
                context,
                tree,
                &mut Vec::new(),
            )?;
            let access_map = isl::get_access_map(
                total_space.get_dim(barvinok::DimType::Param)? as usize,
                0,
                context,
                tree,
                &mut Vec::new(),
            )?;
            let access_map = access_map.intersect_domain(space.clone())?;
            debug!("Timestamp space: {:?}", space);
            debug!("Access map: {:?}", access_map);
        }

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
