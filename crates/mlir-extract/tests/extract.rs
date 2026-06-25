//! End-to-end tests for the MLIR -> DSL extractor.
//!
//! These link MLIR through `melior`, so they require the `mlir-sys` build
//! environment (the workspace `.cargo/config.toml` points it at the apt
//! LLVM/MLIR 22 install).

use mlir_extract::extract_dsl_from_source;

/// Translate `source`, asserting success, and assert the emitted DSL parses and
/// validates as a `dmd-core` program (round-trip correctness).
fn extract_ok(source: &str) -> String {
    let dsl = extract_dsl_from_source(source, "dmd.extract")
        .unwrap_or_else(|error| panic!("extraction failed: {error}"));
    let program = dmd_core::parse_program(&dsl)
        .unwrap_or_else(|error| panic!("generated DSL did not parse: {error}\n---\n{dsl}"));
    dmd_core::validate_program(program)
        .unwrap_or_else(|error| panic!("generated DSL did not validate: {error}\n---\n{dsl}"));
    dsl
}

const MATMUL: &str = r#"
module {
  func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>,
                    %M: index, %N: index, %K: index) {
    affine.for %i = 0 to %M {
      affine.for %j = 0 to %N {
        affine.for %k = 0 to %K {
          %a = affine.load %A[%i, %k] : memref<?x?xf32>
          %b = affine.load %B[%k, %j] : memref<?x?xf32>
          %c = affine.load %C[%i, %j] : memref<?x?xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %c, %p : f32
          affine.store %s, %C[%i, %j] : memref<?x?xf32>
        }
      }
    } { dmd.extract }
    return
  }
}
"#;

#[test]
fn translates_symbolic_matmul() {
    let dsl = extract_ok(MATMUL);
    // Three nested loops, three arrays, and the canonical matmul access set.
    assert_eq!(dsl.matches("for ").count(), 3);
    assert!(dsl.contains("array A["));
    assert!(dsl.contains("array B["));
    assert!(dsl.contains("array C["));
    assert!(dsl.contains("write C[i0, i1];"));
}

#[test]
fn static_shapes_become_constant_extents() {
    let source = r#"
module {
  func.func @stencil(%A: memref<100x100xf32>, %B: memref<100x100xf32>) {
    affine.for %i = 1 to 99 {
      affine.for %j = 1 to 99 {
        %c = affine.load %A[%i - 1, %j + 1] : memref<100x100xf32>
        affine.store %c, %B[%i, %j] : memref<100x100xf32>
      }
    } { dmd.extract }
    return
  }
}
"#;
    let dsl = extract_ok(source);
    assert!(dsl.contains("array A[100, 100];"));
    // Affine subtraction/addition is rendered with `-`/`+`.
    assert!(dsl.contains("read A[i0 - 1, i1 + 1];"));
    assert!(dsl.contains("for i0 in 1 .. 99"));
}

#[test]
fn step_and_if_conditions() {
    let source = r#"
#cond = affine_set<(d0, d1) : (d0 - d1 >= 0, d0 + d1 - 1 == 0)>
module {
  func.func @tri(%A: memref<64x64xf32>, %B: memref<64xf32>) {
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 step 2 {
        affine.if #cond(%i, %j) {
          %v = affine.load %A[%i, %j] : memref<64x64xf32>
          affine.store %v, %B[%i] : memref<64xf32>
        }
      }
    } { dmd.extract }
    return
  }
}
"#;
    let dsl = extract_ok(source);
    assert!(dsl.contains("step 2"));
    assert!(dsl.contains("if i0 - i1 >= 0 && i0 + i1 - 1 == 0 {"));
}

#[test]
fn custom_attribute_name() {
    let source = r#"
module {
  func.func @f(%A: memref<8xf32>) {
    affine.for %i = 0 to 8 {
      %v = affine.load %A[%i] : memref<8xf32>
      affine.store %v, %A[%i] : memref<8xf32>
    } { my.marker }
    return
  }
}
"#;
    let dsl = extract_dsl_from_source(source, "my.marker").expect("custom attr should be honored");
    assert!(dsl.contains("array A[8];"));
}

#[test]
fn rejects_unsupported_operation_with_span() {
    let source = r#"
module {
  func.func @bad(%A: memref<10xf32>, %N: index) {
    affine.for %i = 0 to %N {
      %v = memref.load %A[%i] : memref<10xf32>
      affine.store %v, %A[%i] : memref<10xf32>
    } { dmd.extract }
    return
  }
}
"#;
    let error =
        extract_dsl_from_source(source, "dmd.extract").expect_err("memref.load is unsupported");
    assert!(error.message.contains("memref.load"));
    assert!(
        error.span.is_some(),
        "unsupported op should carry a source span"
    );
}

#[test]
fn rejects_mod_in_index() {
    let source = r#"
module {
  func.func @bad(%A: memref<10xf32>, %N: index) {
    affine.for %i = 0 to %N {
      %v = affine.load %A[%i mod 4] : memref<10xf32>
      affine.store %v, %A[%i] : memref<10xf32>
    } { dmd.extract }
    return
  }
}
"#;
    let error = extract_dsl_from_source(source, "dmd.extract").expect_err("mod is unsupported");
    assert!(error.message.contains("mod"));
}

#[test]
fn rejects_min_bound() {
    let source = r#"
#ub = affine_map<()[s0] -> (s0, 10)>
module {
  func.func @bad(%A: memref<100xf32>, %N: index) {
    affine.for %i = 0 to min #ub()[%N] {
      %v = affine.load %A[%i] : memref<100xf32>
      affine.store %v, %A[%i] : memref<100xf32>
    } { dmd.extract }
    return
  }
}
"#;
    let error =
        extract_dsl_from_source(source, "dmd.extract").expect_err("min bound is unsupported");
    assert!(error.message.contains("min()"));
}

#[test]
fn rejects_missing_tag() {
    let source = r#"
module {
  func.func @f(%A: memref<8xf32>) {
    affine.for %i = 0 to 8 {
      %v = affine.load %A[%i] : memref<8xf32>
      affine.store %v, %A[%i] : memref<8xf32>
    }
    return
  }
}
"#;
    let error = extract_dsl_from_source(source, "dmd.extract").expect_err("no tag present");
    assert!(error.message.contains("no operation tagged"));
}

#[test]
fn rejects_tag_on_non_loop() {
    let source = r#"
module {
  func.func @f(%A: memref<8xf32>) {
    affine.for %i = 0 to 8 {
      %v = affine.load %A[%i] { dmd.extract } : memref<8xf32>
      affine.store %v, %A[%i] : memref<8xf32>
    }
    return
  }
}
"#;
    let error = extract_dsl_from_source(source, "dmd.extract").expect_err("tag on load");
    assert!(error.message.contains("must be placed on an `affine.for`"));
}
