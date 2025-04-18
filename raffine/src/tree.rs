use melior::{
    dialect::ods::affine::{
        AffineForOperation, AffineIfOperation, AffineLoadOperation, AffineStoreOperation,
    },
    ir::{BlockLike, BlockRef, Operation, RegionLike, ValueLike},
};

use crate::{
    Context,
    affine::{AffineMap, IntegerSet},
};

#[derive(Debug)]
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
        map: AffineMap<'a>,
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
        map: AffineMap<'a>,
        is_write: bool,
    ) -> &'a Tree<'a> {
        self.arena.alloc(Tree::Access {
            target_id,
            map,
            is_write,
        })
    }

    pub fn build_tree_from_loop<'a>(
        &'a self,
        entry: &'_ AffineForOperation<'a>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from loop: {}", entry.as_operation());
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
        tracing::trace!("building tree from load: {}", entry.as_operation());
        let target = entry.memref()?;
        let target_id = target.to_raw().ptr as usize;
        let map = entry.map()?;
        let map =
            AffineMap::from_attr(map).ok_or(crate::Error::InvalidLoopNest("invalid load map"))?;
        let is_write = false;
        Ok(self.build_access(target_id, map, is_write))
    }

    fn build_tree_from_store<'a>(
        &'a self,
        entry: &'_ AffineStoreOperation<'a>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from store: {}", entry.as_operation());
        let target = entry.memref()?;
        let target_id = target.to_raw().ptr as usize;
        let map = entry.map()?;
        let map =
            AffineMap::from_attr(map).ok_or(crate::Error::InvalidLoopNest("invalid store map"))?;
        let is_write = true;
        Ok(self.build_access(target_id, map, is_write))
    }

    fn build_tree_from_if<'a>(
        &'a self,
        entry: &'_ AffineIfOperation<'a>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from if: {}", entry.as_operation());
        let condition = entry.as_operation();
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
                        unsafe { std::mem::transmute::<&Operation, &AffineStoreOperation>(op) };
                    let tree = self.build_tree_from_store(store_op)?;
                    subtrees.push(tree);
                }
                "affine.if" => {
                    let if_op =
                        unsafe { std::mem::transmute::<&Operation, &AffineIfOperation>(op) };
                    let tree = self.build_tree_from_if(if_op)?;
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

#[cfg(test)]
mod tests {
    use melior::ir::{BlockLike, Module, RegionLike};

    use crate::Context;

    #[test]
    fn build_tree() {
        _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .try_init();
        let context = Context::new();
        let module = r#"
 module {
  func.func @stencil_kernel(%A: memref<100x100xf32>, %B: memref<100x100xf32>) {
    affine.for %i = 1 to 99 {
      affine.for %j = 1 to 99 {
        // Load the neighboring values and the center value
        %top    = affine.load %A[%i - 1, %j] : memref<100x100xf32>
        %bottom = affine.load %A[%i + 1, %j] : memref<100x100xf32>
        %left   = affine.load %A[%i, %j - 1] : memref<100x100xf32>
        %right  = affine.load %A[%i, %j + 1] : memref<100x100xf32>
        %center = affine.load %A[%i, %j] : memref<100x100xf32>

        // Perform the sum of the loaded values
        %sum1 = arith.addf %top, %bottom : f32
        %sum2 = arith.addf %left, %right : f32
        %sum3 = arith.addf %sum1, %sum2 : f32
        %sum4 = arith.addf %sum3, %center : f32

        // Compute the average (sum / 5.0)
        %c5 = arith.constant 5.0 : f32
        %avg = arith.divf %sum4, %c5 : f32

        // Store the result in the output array
        affine.store %avg, %B[%i, %j] : memref<100x100xf32>
      }
    } { slap.extract }
    return
  }
}
"#;
        let module = Module::parse(context.mlir_context(), module).unwrap();
        tracing::debug!("Parsed module: {}", module.body().to_string());
        let body = module.body();
        let op = body.first_operation().unwrap();
        let body = op.region(0).unwrap();
        let body = body.first_block().unwrap();
        let first_op = body.first_operation().unwrap();
        let op: &melior::dialect::ods::affine::AffineForOperation =
            unsafe { std::mem::transmute(first_op.to_ref()) };
        let tree = context.build_tree_from_loop(op).unwrap();
        tracing::debug!("Tree: {:?}", tree);
    }
}
