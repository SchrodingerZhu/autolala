module {
  func.func @matmul(%C: memref<?x?xf64>, %A: memref<?x?xf64>, %B: memref<?x?xf64>, 
                   %s0: index, %s1: index, %s2: index, %t0: index, %t1: index, %t2: index, %tt0: index, %tt1: index, %tt2: index) {  // Removed %arg6, %arg7, %arg8
    affine.for %i = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%s0, %t0] {  
      affine.for %j = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%s1, %t1] {
        affine.for %k = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%s2, %t2] { 
          affine.for %ii = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%t0, %tt0] { 
            affine.for %jj = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%t1, %tt1] { 
              affine.for %kk = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%t2, %tt2] {
                affine.for %iii = 0 to %tt0 { 
                  affine.for %jjj = 0 to %tt1 { 
                    affine.for %kkk = 0 to %tt2 { 
                      %0 = affine.load %A[%i * symbol(%t0) + %ii * symbol(%tt0) + %iii, %k * symbol(%t2) + %kk * symbol(%tt2) + %kkk] : memref<?x?xf64>
                      %1 = affine.load %B[%k * symbol(%t2) + %kk * symbol(%tt2) + %kkk, %j * symbol(%t1) + %jj * symbol(%tt1) + %jjj] : memref<?x?xf64>
                      %2 = arith.mulf %0, %1 : f64
                      affine.store %2, %C[%i * symbol(%t0) + %ii * symbol(%tt0) + %iii,  %j * symbol(%t1) + %jj * symbol(%tt1) + %jjj] : memref<?x?xf64>
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
