use crate::{
    Context,
    affine::{AffineMap, IntegerSet},
};

pub enum Tree<'a> {
    For {
        lower_bound: AffineMap<'a>,
        upper_bound: AffineMap<'a>,
        step: isize,
        body: &'a Tree<'a>,
    },
    Block(&'a [&'a Tree<'a>]),
    Access {
        target_id: usize,
        indices: AffineMap<'a>,
        is_write: bool,
    },
    If {
        condition: IntegerSet<'a>,
        then: &'a Tree<'a>,
        r#else: Option<&'a Tree<'a>>,
    },
}

#[allow(unused)]
impl Context {
    fn build_if<'a>(
        &'a self,
        condition: IntegerSet<'a>,
        then: &'a Tree<'a>,
        r#else: Option<&'a Tree<'a>>,
    ) -> &'a Tree<'a> {
        self.arena.alloc(Tree::If {
            condition,
            then,
            r#else,
        })
    }
    fn build_for<'a>(
        &'a self,
        lower_bound: AffineMap<'a>,
        upper_bound: AffineMap<'a>,
        step: isize,
        body: &'a Tree<'a>,
    ) -> &'a Tree<'a> {
        self.arena.alloc(Tree::For {
            lower_bound,
            upper_bound,
            step,
            body,
        })
    }
    fn build_block<'a, I>(&'a self, body: I) -> &'a Tree<'a>
    where
        I: IntoIterator<Item = &'a Tree<'a>>,
        I::IntoIter: ExactSizeIterator,
    {
        let inner = self.arena.alloc_slice_fill_iter(body.into_iter());
        self.arena.alloc(Tree::Block(inner))
    }
    fn build_access<'a>(
        &'a self,
        target_id: usize,
        indices: AffineMap<'a>,
        is_write: bool,
    ) -> &'a Tree<'a> {
        self.arena.alloc(Tree::Access {
            target_id,
            indices,
            is_write,
        })
    }
}
