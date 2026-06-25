module {
  // Fused + unroll-jam over i by 4: pass2 reuses each y[j] load across 4 rows.
  func.func @kernel(%A: memref<?x?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %T: memref<?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0.0 : f64
    affine.for %ii = 0 to %N step 4 {
      // compute T for up to 4 rows
      affine.for %io = 0 to min affine_map<(d0,d1)->(4, d1-d0)>(%ii)[%N] {
        %i = arith.addi %ii, %io : index
        %ti = affine.for %j = 0 to %N iter_args(%acc = %c0) -> (f64) {
          %a = affine.load %A[%i, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %ti, %T[%i] : memref<?xf64>
      }
      return
    }
    return
  }
}
