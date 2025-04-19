use melior::ir::{BlockLike, BlockRef, OperationRef, RegionLike};

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

impl std::fmt::Display for Tree<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tree::For {
                lower_bound,
                upper_bound,
                step,
                body,
            } => write!(
                f,
                "for {} to {} step {} {}",
                lower_bound, upper_bound, step, body
            ),
            Tree::Block(body) => write!(
                f,
                "{{ {} }}",
                body.iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Tree::Access {
                target_id,
                map,
                is_write: true,
            } => write!(f, "store {}[{}]", target_id, map),
            Tree::Access {
                target_id,
                map,
                is_write: false,
            } => write!(f, "load {}[{}]", target_id, map),
            Tree::If {
                condition,
                then,
                r#else,
            } => write!(
                f,
                "if {} {{ {} }} else {{ {} }}",
                condition,
                then,
                r#else.map_or("None".to_string(), |e| e.to_string())
            ),
        }
    }
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
        entry: OperationRef<'a, '_>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from loop: {}", entry);
        let lower_bound = crate::cxx::for_op_get_lower_bound_map(entry)?;
        let upper_bound = crate::cxx::for_op_get_upper_bound_map(entry)?;
        let step = crate::cxx::for_op_get_step(entry)?;
        let region = entry.region(0)?;
        let body = region
            .first_block()
            .ok_or(crate::Error::InvalidLoopNest("invalid loop body"))?;
        if body.next_in_region().is_some() {
            return Err(crate::Error::InvalidLoopNest("invalid loop body"));
        }
        let body = self.build_tree_from_block(body)?;
        Ok(self.build_for(lower_bound, upper_bound, step, body))
    }

    fn build_tree_from_load_store<'a>(
        &'a self,
        is_write: bool,
        entry: OperationRef<'a, '_>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from load: {}", entry);
        let target_id = crate::cxx::load_store_op_get_access_id(entry)?;
        let map = crate::cxx::load_store_op_get_access_map(entry)?;
        Ok(self.build_access(target_id, map, is_write))
    }

    fn build_tree_from_if<'a>(
        &'a self,
        entry: OperationRef<'a, '_>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from if: {}", entry);
        let condition = crate::cxx::if_op_get_condition(entry)?;
        let then_block = crate::cxx::if_op_get_then_block(entry)?;
        let then = self.build_tree_from_block(then_block)?;
        let r#else = if let Some(else_block) = crate::cxx::if_op_get_else_block(entry)? {
            Some(self.build_tree_from_block(else_block)?)
        } else {
            None
        };
        Ok(self.build_if(condition, then, r#else))
    }

    fn build_tree_from_block<'a>(
        &'a self,
        entry: BlockRef<'a, '_>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from block: {}", entry);
        let mut subtrees = Vec::new();
        fn collect_op<'a>(
            this: &'a Context,
            op: OperationRef<'a, '_>,
            subtrees: &mut Vec<&'a Tree<'a>>,
        ) -> Result<(), crate::Error> {
            let name = op.name();
            let name = name
                .as_string_ref()
                .as_str()
                .map_err(|_| crate::Error::InvalidLoopNest("invalid operation name"))?;
            'dispatch: {
                let res = match name {
                    "affine.for" => this.build_tree_from_loop(op)?,
                    "affine.load" => this.build_tree_from_load_store(false, op)?,
                    "affine.store" => this.build_tree_from_load_store(true, op)?,
                    "affine.if" => this.build_tree_from_if(op)?,
                    _ => {
                        tracing::trace!("ignored operation: {}", op);
                        break 'dispatch;
                    }
                };
                subtrees.push(res);
            }
            if let Some(next) = op.next_in_block() {
                return collect_op(this, next, subtrees);
            }
            Ok(())
        }
        if let Some(op) = entry.first_operation() {
            collect_op(self, op, &mut subtrees)?;
        }
        if subtrees.is_empty() {
            return Err(crate::Error::InvalidLoopNest("empty block"));
        }
        if subtrees.len() == 1 {
            return Ok(subtrees[0]);
        }
        let block = self.build_block(subtrees);
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
        println!("First operation: {}", first_op);
        let tree = context.build_tree_from_loop(first_op).unwrap();
        tracing::debug!("Tree: {:#}", tree);
    }
}
