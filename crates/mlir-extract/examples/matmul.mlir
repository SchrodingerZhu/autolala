// Symbolic matmul: C[i,j] += A[i,k] * B[k,j]
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
