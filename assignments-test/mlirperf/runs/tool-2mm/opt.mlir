#id = affine_map<(d0) -> (d0)>
#ub_i = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#ub_k = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#ub_j = affine_map<(d0)[s0] -> (d0 + 128, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %D: memref<?x?xf64>, %T: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    // Fuse the two matmuls by i-band: for each band of rows, produce T[band]=A[band]*B
    // then immediately consume it into D[band]=0.9*D[band]+T[band]*C while T is hot.
    affine.for %ib = 0 to %N step 32 {
      // ---- mm1: T[ib..] = A[ib..]*B  (i-k-j, tiled in k and j) ----
      affine.for %kk = 0 to %N step 32 {
        affine.for %jj = 0 to %N step 128 {
          affine.for %i = #id(%ib) to min #ub_i(%ib)[%N] {
            affine.for %k = #id(%kk) to min #ub_k(%kk)[%N] {
              %a = affine.load %A[%i, %k] : memref<?x?xf64>
              affine.for %j = #id(%jj) to min #ub_j(%jj)[%N] {
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
      // ---- D[ib..] *= 0.9 ----
      affine.for %i = #id(%ib) to min #ub_i(%ib)[%N] {
        affine.for %j = 0 to %N {
          %d = affine.load %D[%i, %j] : memref<?x?xf64>
          %ds = arith.mulf %d, %b9 : f64
          affine.store %ds, %D[%i, %j] : memref<?x?xf64>
        }
      }
      // ---- mm2: D[ib..] += T[ib..]*C  (i-k-j, tiled) ----
      affine.for %kk = 0 to %N step 32 {
        affine.for %jj = 0 to %N step 128 {
          affine.for %i = #id(%ib) to min #ub_i(%ib)[%N] {
            affine.for %k = #id(%kk) to min #ub_k(%kk)[%N] {
              %t = affine.load %T[%i, %k] : memref<?x?xf64>
              affine.for %j = #id(%jj) to min #ub_j(%jj)[%N] {
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
