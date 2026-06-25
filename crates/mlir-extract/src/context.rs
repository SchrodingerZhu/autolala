//! MLIR context construction with all upstream dialects loaded, so that
//! `affine`, `memref`, `arith`, `func`, ... parse out of the box.

use melior::Context;
use melior::dialect::DialectRegistry;
use melior::utility::register_all_dialects;

/// Build a fresh MLIR context with every available dialect registered and
/// loaded.
pub fn new_context() -> Context {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}
