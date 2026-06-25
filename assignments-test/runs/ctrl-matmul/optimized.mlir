module {
  func.func @matmul(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>,
                    %M: index, %N: index, %K: index) {
    affine.for %ii = 0 to %M step 32 {
      affine.for %jj = 0 to %N step 32 {
        affine.for %kk = 0 to %K step 32 {
          affine.for %i = 0 to 32 {
            affine.for %j = 0 to 32 {
              affine.for %k = 0 to 32 {
                %a = affine.load %A[%ii + %i, %kk + %k] : memref<?x?xf64>
                %b = affine.load %B[%kk + %k, %jj + %j] : memref<?x?xf64>
                %c = affine.load %C[%ii + %i, %jj + %j] : memref<?x?xf64>
                %p = arith.mulf %a, %b : f64
                %s = arith.addf %c, %p : f64
                affine.store %s, %C[%ii + %i, %jj + %j] : memref<?x?xf64>
              }
            }
          }
        }
      }
    } {dmd.extract}
    return
  }
}
