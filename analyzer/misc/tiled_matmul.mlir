module {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, 
                   %arg3: index, %arg4: index, %arg5: index) {  // Removed %arg6, %arg7, %arg8
    affine.for %arg6 = 0 to %arg3 step 16 {
      affine.for %arg7 = 0 to %arg5 step 16 {
        affine.for %arg8 = 0 to %arg4 step 16 {
          affine.for %arg9 = 0 to 16 {
            affine.for %arg10 = 0 to 16 {
              affine.for %arg11 = 0 to 16 {
                %0 = affine.load %arg0[%arg6 + %arg9, %arg8 + %arg11] : memref<?x?xf32>
                %1 = affine.load %arg1[%arg8 + %arg11, %arg7 + %arg10] : memref<?x?xf32>
                %2 = affine.load %arg2[%arg6 + %arg9, %arg7 + %arg10] : memref<?x?xf32>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                affine.store %4, %arg2[%arg6 + %arg9, %arg7 + %arg10] : memref<?x?xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}
