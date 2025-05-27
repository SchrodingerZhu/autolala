module {
  func.func @matmul(%C: memref<?x?xf32>, %A: memref<?x?xf32>, %B: memref<?x?xf32>) {  // Removed %arg6, %arg7, %arg8
    affine.for %i = 0 to 512 step 32 {  
      affine.for %j = 0 to 512 step 32 {
        affine.for %k = 0 to 512 step 32 { 
          affine.for %ii = 0 to 32 { 
            affine.for %jj = 0 to 32 { 
              affine.for %kk = 0 to 32 { 
                %0 = affine.load %A[%i + %ii, %k + %kk] : memref<?x?xf32>
                %1 = affine.load %B[%k + %kk, %j + %jj] : memref<?x?xf32>
                %2 = arith.mulf %0, %1 : f32
                affine.store %2, %C[%i + %ii,  %j + %jj] : memref<?x?xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}
