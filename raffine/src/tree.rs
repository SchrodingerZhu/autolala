use melior::{
    dialect::ods::affine::{AffineForOperation, AffineLoadOperation},
    ir::{BlockLike, BlockRef, Operation, RegionLike},
};

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
        let inner = self.arena.alloc_slice_fill_iter(body);
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

    pub fn build_tree_from_loop<'a>(
        &'a self,
        entry: &'_ AffineForOperation<'a>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from loop: {}", unsafe {
            std::mem::transmute::<&AffineForOperation, &Operation>(entry)
        });
        let lower_bound = entry.lower_bound_map()?;
        let upper_bound = entry.upper_bound_map()?;
        let lower_bound = AffineMap::from_attr(lower_bound)
            .ok_or(crate::Error::InvalidLoopNest("invalid lowerbound"))?;
        let upper_bound = AffineMap::from_attr(upper_bound)
            .ok_or(crate::Error::InvalidLoopNest("invalid upperbound"))?;
        let step = entry.step()?.signed_value() as isize;
        let region = entry.region()?;
        let body = region
            .first_block()
            .ok_or(crate::Error::InvalidLoopNest("invalid loop body"))?;
        if body.next_in_region().is_some() {
            return Err(crate::Error::InvalidLoopNest("invalid loop body"));
        }
        let body = self.build_tree_from_block(body)?;
        Ok(self.build_for(lower_bound, upper_bound, step, body))
    }

    fn build_tree_from_load<'a>(
        &'a self,
        entry: &'_ AffineLoadOperation<'a>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from load: {}", unsafe {
            std::mem::transmute::<&AffineLoadOperation, &Operation>(entry)
        });
        todo!()
    }

    fn build_tree_from_store<'a>(
        &'a self,
        entry: &'_ AffineLoadOperation<'a>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from store: {}", unsafe {
            std::mem::transmute::<&AffineLoadOperation, &Operation>(entry)
        });
        todo!()
    }

    fn build_tree_from_block<'a>(
        &'a self,
        entry: BlockRef<'a, '_>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from block: {}", entry);
        let mut subtrees = Vec::new();
        let mut cursor = entry.first_operation().map(|x| unsafe { x.to_ref() });
        while let Some(op) = cursor {
            let name = op.name();
            let name = name
                .as_string_ref()
                .as_str()
                .map_err(|_| crate::Error::InvalidLoopNest("invalid operation name"))?;
            match name {
                "affine.for" => {
                    let for_op =
                        unsafe { std::mem::transmute::<&Operation, &AffineForOperation>(op) };
                    let tree = self.build_tree_from_loop(for_op)?;
                    subtrees.push(tree);
                }
                "affine.load" => {
                    let load_op =
                        unsafe { std::mem::transmute::<&Operation, &AffineLoadOperation>(op) };
                    let tree = self.build_tree_from_load(load_op)?;
                    subtrees.push(tree);
                }
                "affine.store" => {
                    let store_op =
                        unsafe { std::mem::transmute::<&Operation, &AffineLoadOperation>(op) };
                    let tree = self.build_tree_from_store(store_op)?;
                    subtrees.push(tree);
                }
                _ => tracing::trace!("ignored operation: {}", op),
            }
            cursor = op.next_in_block().map(|x| unsafe { x.to_ref() });
        }
        if subtrees.is_empty() {
            return Err(crate::Error::InvalidLoopNest("empty block"));
        }
        if subtrees.len() == 1 {
            return Ok(subtrees[0]);
        }
        let block = self.build_block(subtrees.into_iter());
        Ok(block)
    }
}
