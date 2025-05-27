module {
  func.func @matmul(%C: memref<?x?xf32>, %A: memref<?x?xf32>, %B: memref<?x?xf32>) {
    affine.for %i = 0 to 512 step 128 {  
      affine.for %j = 0 to 512 step 128 {
        affine.for %k = 0 0 to 512 step 128 { 
          affine.for %ii = 0 to 128 step 32 { 
            affine.for %jj = 0 to 128 step 32 { 
              affine.for %kk = 0 to 128 step 32 {
                affine.for %iii = 0 to 32 { 
                  affine.for %jjj = 0 to 32 { 
                    affine.for %kkk = 0 to 32 { 
                      %0 = affine.load %A[%i + %ii + %iii, %k + %kk + %kkk] : memref<?x?xf32>
                      %1 = affine.load %B[%k + %kk + %kkk, %j + %jj + %jjj] : memref<?x?xf32>
                      %2 = arith.mulf %0, %1 : f32
                      affine.store %2, %C[%i + %ii + %iii,  %j + %jj + %jjj] : memref<?x?xf32>
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
