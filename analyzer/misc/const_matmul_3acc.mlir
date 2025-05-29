module {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    affine.for %arg6 = 0 to 512 {
      affine.for %arg7 = 0 to 512 {
        affine.for %arg8 = 0 to 512 {
          %0 = affine.load %arg1[%arg6, %arg8] : memref<?x?xf32>
          %1 = affine.load %arg2[%arg8, %arg7] : memref<?x?xf32>
          %2 = arith.mulf %0, %1 : f32
          affine.store %2, %arg0[%arg6, %arg7] : memref<?x?xf32>
        }
      }
    }
    return
  }
}
