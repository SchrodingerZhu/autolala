use anyhow::Result;
use barvinok::space::Space;
use raffine::tree::Tree;
use raffine::tree::ValID;

use crate::AnalysisContext;

fn get_max_param_ivar<'a>(tree: &Tree<'a>) -> Result<(usize, usize)> {
    let mut max_param = 0;
    let mut max_ivar = 0;
    match tree {
        Tree::For {
            lower_bound_operands,
            upper_bound_operands,
            ivar,
            body,
            ..
        } => {
            let ValID::IVar(n) = ivar else {
                return Err(anyhow::anyhow!("invalid tree"));
            };
            max_ivar = max_ivar.max(*n);
            for id in lower_bound_operands
                .iter()
                .chain(upper_bound_operands.iter())
            {
                match id {
                    ValID::Symbol(n) => {
                        max_param = max_param.max(*n);
                    }
                    ValID::IVar(n) => {
                        max_ivar = max_ivar.max(*n);
                    }
                    _ => {}
                }
            }
            let (param, ivar) = get_max_param_ivar(body)?;
            max_param = max_param.max(param);
            max_ivar = max_ivar.max(ivar);
        }
        Tree::Block(blk) => {
            for subtree in blk.iter() {
                let (param, ivar) = get_max_param_ivar(subtree)?;
                max_param = max_param.max(param);
                max_ivar = max_ivar.max(ivar);
            }
        }
        Tree::Access { operands, .. } => {
            for id in operands.iter() {
                match id {
                    ValID::Symbol(n) => {
                        max_param = max_param.max(*n);
                    }
                    ValID::IVar(n) => {
                        max_ivar = max_ivar.max(*n);
                    }
                    _ => {}
                }
            }
        }
        Tree::If { .. } => return Err(anyhow::anyhow!("not implemented for conditional branch")),
    }
    Ok((max_param, max_ivar))
}

pub fn get_space<'a, 'b: 'a>(context: &AnalysisContext<'b>, tree: &Tree<'a>) -> Result<Space<'b>> {
    let (max_param, max_ivar) = get_max_param_ivar(tree)?;
    let space = Space::set(context.bcontext(), max_param as u32, max_ivar as u32)?;
    Ok(space)
}

/// Return levels of nesting if the loop is perfectly nested.
pub fn get_nesting_level(tree: &Tree) -> Option<usize> {
    match tree {
        Tree::For { body, .. } => get_nesting_level(body).map(|level| level + 1),
        Tree::Block(trees) => {
            if trees.iter().all(|t| matches!(t, Tree::Access { .. })) {
                Some(0)
            } else if trees.len() != 1 {
                None
            } else {
                let Some(t) = trees.first() else {
                    return None;
                };
                get_nesting_level(t)
            }
        }
        Tree::Access { .. } => Some(0),
        Tree::If { .. } => None,
    }
}
