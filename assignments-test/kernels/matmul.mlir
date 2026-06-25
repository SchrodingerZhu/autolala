module {
  func.func @matmul(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>,
                    %M: index, %N: index, %K: index) {
    affine.for %i = 0 to %M {
      affine.for %j = 0 to %N {
        affine.for %k = 0 to %K {
          %a = affine.load %A[%i, %k] : memref<?x?xf64>
          %b = affine.load %B[%k, %j] : memref<?x?xf64>
          %c = affine.load %C[%i, %j] : memref<?x?xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<?x?xf64>
        }
      }
    } {dmd.extract}
    return
  }
}
