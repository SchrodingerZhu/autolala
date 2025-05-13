use anyhow::Result;
use barvinok::space::Space;
use raffine::tree::Tree;

use crate::{AnalysisContext, utils::get_max_param_ivar};

pub fn get_space<'a, 'b: 'a>(context: &AnalysisContext<'b>, tree: &Tree<'a>) -> Result<Space<'b>> {
    let (max_param, max_ivar) = get_max_param_ivar(tree)?;
    let space = Space::set(context.bcontext(), max_param as u32, max_ivar as u32)?;
    Ok(space)
}
