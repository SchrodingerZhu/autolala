#lb = affine_map<(d0) -> (d0)>
#ub = affine_map<(d0)[s0] -> (d0 + 32, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index)
      attributes {llvm.emit_c_interface} {
    affine.for %ii = 0 to %N step 32 {
      affine.for %kk = 0 to %N step 32 {
        affine.for %jj = 0 to %N step 32 {
          affine.for %i = #lb(%ii) to min #ub(%ii)[%N] {
            affine.for %k = #lb(%kk) to min #ub(%kk)[%N] {
              %a = affine.load %A[%i, %k] : memref<?x?xf64>
              affine.for %j = #lb(%jj) to min #ub(%jj)[%N] {
                %b = affine.load %B[%k, %j] : memref<?x?xf64>
                %c = affine.load %C[%i, %j] : memref<?x?xf64>
                %p = arith.mulf %a, %b : f64
                %s = arith.addf %c, %p : f64
                affine.store %s, %C[%i, %j] : memref<?x?xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}
