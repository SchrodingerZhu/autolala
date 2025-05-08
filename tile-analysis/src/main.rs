use anyhow::{Ok, anyhow};
use clap::Parser;
use melior::ir::{BlockLike, Module, OperationRef, RegionLike};
use raffine::{Context, DominanceInfo, tree::Tree};
use std::{io::Read, path::PathBuf};
use tracing::debug;
use tracing_subscriber::EnvFilter;

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

fn extract_target<'a>(
    module: &Module<'a>,
    options: &Options,
    context: &'a Context,
    dom: &'a DominanceInfo<'a>,
) -> anyhow::Result<&'a Tree<'a>> {
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
            Ok(context.build_tree(for_loop, dom)?)
        })
    })
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .init();
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

    let context = raffine::Context::new();

    let mut source = String::new();
    let bytes = reader.read_to_string(&mut source)?;

    debug!("Read {} bytes", bytes);

    let module = Module::parse(context.mlir_context(), &source)
        .ok_or_else(|| anyhow!("Failed to parse module"))?;

    debug!("Parsed module: {}", module.as_operation());

    let dom = DominanceInfo::new(&module);

    let tree = extract_target(&module, &options, &context, &dom)?;

    debug!("Extracted tree: {}", tree);

    Ok(())
}
