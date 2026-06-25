#id = affine_map<(d0) -> (d0)>
#ub64 = affine_map<(d0)[s0] -> (d0 + 64, s0)>
#ub256 = affine_map<(d0)[s0] -> (d0 + 256, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %D: memref<?x?xf64>, %T: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64

    // T = A * B  (tiled, i-k-j order: inner j is contiguous in B and T)
    affine.for %ii = 0 to %N step 64 {
      affine.for %kk = 0 to %N step 64 {
        affine.for %jj = 0 to %N step 256 {
          affine.for %i = #id(%ii) to min #ub64(%ii)[%N] {
            affine.for %k = #id(%kk) to min #ub64(%kk)[%N] {
              %a = affine.load %A[%i, %k] : memref<?x?xf64>
              affine.for %j = #id(%jj) to min #ub256(%jj)[%N] {
                %b = affine.load %B[%k, %j] : memref<?x?xf64>
                %t = affine.load %T[%i, %j] : memref<?x?xf64>
                %p = arith.mulf %a, %b : f64
                %s = arith.addf %t, %p : f64
                affine.store %s, %T[%i, %j] : memref<?x?xf64>
              }
            }
          }
        }
      }
    }

    // D = 0.9 * D
    affine.for %i = 0 to %N {
      affine.for %j = 0 to %N {
        %d = affine.load %D[%i, %j] : memref<?x?xf64>
        %ds = arith.mulf %d, %b9 : f64
        affine.store %ds, %D[%i, %j] : memref<?x?xf64>
      }
    }

    // D = D + T * C  (tiled, i-k-j order: inner j is contiguous in C and D)
    affine.for %ii = 0 to %N step 64 {
      affine.for %kk = 0 to %N step 64 {
        affine.for %jj = 0 to %N step 256 {
          affine.for %i = #id(%ii) to min #ub64(%ii)[%N] {
            affine.for %k = #id(%kk) to min #ub64(%kk)[%N] {
              %t = affine.load %T[%i, %k] : memref<?x?xf64>
              affine.for %j = #id(%jj) to min #ub256(%jj)[%N] {
                %c = affine.load %C[%k, %j] : memref<?x?xf64>
                %d = affine.load %D[%i, %j] : memref<?x?xf64>
                %p = arith.mulf %t, %c : f64
                %s = arith.addf %d, %p : f64
                affine.store %s, %D[%i, %j] : memref<?x?xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}
