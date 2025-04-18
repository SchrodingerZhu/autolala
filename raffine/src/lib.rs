use melior::Context;
pub mod affine;

pub fn initialize_mlir_context() -> Context {
    let context = melior::Context::new();
    let registery = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registery);
    context.append_dialect_registry(&registery);
    context.load_all_available_dialects();
    tracing::debug!("Loaded all available dialects ({} in total)", context.loaded_dialect_count());
    context
}

#[cfg(test)]
mod tests {
    use melior::ir::Module;


    #[test]
    fn mlir_loads_affine_dialect() {
        _ = tracing_subscriber::fmt::try_init();
        let context = super::initialize_mlir_context();
        let dialect = context.get_or_load_dialect("affine");
        tracing::debug!("Loaded dialect: {}", dialect.namespace().unwrap());
        assert_eq!(dialect.namespace().unwrap(), "affine");
    }

    #[test]
    fn mlir_parse_module() {
        let context = super::initialize_mlir_context();
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
        let module = Module::parse(&context, module).unwrap();
        tracing::debug!("Parsed module: {}", module.body().to_string());
    }
}
