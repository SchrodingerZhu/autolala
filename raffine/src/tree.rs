use crate::affine::{AffineMap, IntegerSet};
use melior::ir::Value;
pub enum Tree<'a> {
    For {
        lower_bound: AffineMap<'a>,
        upper_bound: AffineMap<'a>,
        step: isize,
        body: &'a Tree<'a>,
    },
    Block(&'a [Tree<'a>]),
    Access {
        target: Value<'a, 'a>,
        indices: AffineMap<'a>,
        is_write: bool,
    },
    If {
        condition: IntegerSet<'a>,
        then: &'a Tree<'a>,
        r#else: Option<&'a Tree<'a>>,
    }
}