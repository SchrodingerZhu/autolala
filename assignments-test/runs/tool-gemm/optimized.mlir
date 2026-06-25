module {
  func.func @kernel_gemm(%C: memref<?x?xf64>,
                         %A: memref<?x?xf64>,
                         %B: memref<?x?xf64>, %NI : index, %NJ : index, %NK : index) {
      affine.for %loop_once = 0 to 1 {

        %alpha = arith.constant 1.0 : f64
        %beta = arith.constant 1.0 : f64

        // C[i][j] *= beta  (tiled i,j)
        affine.for %ii = 0 to %NI step 32 {
          affine.for %jj = 0 to %NJ step 32 {
            affine.for %i = 0 to 32 {
              affine.for %j = 0 to 32 {
                %0 = affine.load %C[%ii + %i, %jj + %j] : memref<?x?xf64>
                %1 = arith.mulf %0, %beta : f64
                affine.store %1, %C[%ii + %i, %jj + %j] : memref<?x?xf64>
              }
            }
          }
        }

        // C[i][j] += alpha * A[i][k] * B[k][j]  (tiled i,k,j)
        affine.for %ii = 0 to %NI step 32 {
          affine.for %kk = 0 to %NK step 32 {
            affine.for %jj = 0 to %NJ step 32 {
              affine.for %i = 0 to 32 {
                affine.for %k = 0 to 32 {
                  affine.for %j = 0 to 32 {
                    %2 = affine.load %C[%ii + %i, %jj + %j] : memref<?x?xf64>
                    %3 = affine.load %A[%ii + %i, %kk + %k] : memref<?x?xf64>
                    %4 = affine.load %B[%kk + %k, %jj + %j] : memref<?x?xf64>
                    %5 = arith.mulf %alpha, %3 : f64
                    %6 = arith.mulf %5, %4 : f64
                    %7 = arith.addf %2, %6 : f64
                    affine.store %7, %C[%ii + %i, %jj + %j] : memref<?x?xf64>
                  }
                }
              }
            }
          }
        }
     } {dmd.extract}
    return
  }
}
