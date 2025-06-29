module {
  func.func @matmul(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to %arg3 {
      affine.for %arg7 = 0 to %arg5 {
        affine.for %arg8 = 0 to %arg4 {
          %0 = affine.load %arg1[%arg6, %arg8] : memref<?x?xf64>
          %1 = affine.load %arg2[%arg8, %arg7] : memref<?x?xf64>
          %2 = arith.mulf %0, %1 : f64
          affine.store %2, %arg0[%arg6, %arg7] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
