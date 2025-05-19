module {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, 
                   %arg3: index, %arg4: index, %arg5: index, %t0: index, %t1: index, %t2: index) {  // Removed %arg6, %arg7, %arg8
    affine.for %i = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%arg3, %t0] {
      affine.for %j = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%arg5, %t1] {
        affine.for %k = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%arg4, %t2] {
          affine.for %ii = 0 to %t0 {
            affine.for %jj = 0 to %t1 {
              affine.for %kk = 0 to %t2 {
                %0 = affine.load %arg1[%i * symbol(%t0) + %ii, %k * symbol(%t2) + %kk] : memref<?x?xf32>
                %1 = affine.load %arg2[%k * symbol(%t2) + %kk, %j * symbol(%t1) + %jj] : memref<?x?xf32>
                %2 = arith.mulf %0, %1 : f32
                affine.store %2, %arg0[%i * symbol(%t0) + %ii,  %j * symbol(%t1) + %jj] : memref<?x?xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}
