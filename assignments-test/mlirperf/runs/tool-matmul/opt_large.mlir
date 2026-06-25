#lb = affine_map<(d0) -> (d0)>
#ub64 = affine_map<(d0)[s0] -> (d0 + 64, s0)>
#ub128 = affine_map<(d0)[s0] -> (d0 + 128, s0)>
#ub256 = affine_map<(d0)[s0] -> (d0 + 256, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    // Large N (1152-1536): 3D cache tiling with i-k-j interchange. The
    // (kk x jj) B-block = 128 x 256 x 8B = 256KB stays L2-resident and is
    // reused across all i in the ii-tile; tiling caps reuse distance to the
    // tile constant (DMD dominant term goes from RD~N to RD~const). j inner
    // is unit-stride for B and C; A[i,k] is register-invariant across j.
    affine.for %ii = 0 to %N step 64 {
      affine.for %kk = 0 to %N step 128 {
        affine.for %jj = 0 to %N step 256 {
          affine.for %i = #lb(%ii) to min #ub64(%ii)[%N] {
            affine.for %k = #lb(%kk) to min #ub128(%kk)[%N] {
              %a = affine.load %A[%i, %k] : memref<?x?xf64>
              affine.for %j = #lb(%jj) to min #ub256(%jj)[%N] {
                %b = affine.load %B[%k, %j] : memref<?x?xf64>
                %c = affine.load %C[%i, %j] : memref<?x?xf64>
                %p = arith.mulf %a, %b : f64
                %s = arith.addf %c, %p : f64
                affine.store %s, %C[%i, %j] : memref<?x?xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}
