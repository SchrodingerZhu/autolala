module attributes { "simulation.prologue" = "volatile double ARRAY_0[256][256]; volatile double ARRAY_1[256][256]; volatile double ARRAY_2[256][256];" } {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    affine.for %arg6 = 0 to 256 {
      affine.for %arg7 = 0 to 256 {
        affine.for %arg8 = 0 to 256 {
          %0 = affine.load %arg0[%arg6, %arg8] : memref<?x?xf32>
          %1 = affine.load %arg1[%arg8, %arg7] : memref<?x?xf32>
          %2 = affine.load %arg2[%arg6, %arg7] : memref<?x?xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg6, %arg7] : memref<?x?xf32>
        }
      }
    }
    return
  }
}
