#lb = affine_map<(d0) -> (d0)>
#ub_i = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#ub_j = affine_map<(d0)[s0] -> (d0 + 128, s0)>
#ub_k = affine_map<(d0)[s0] -> (d0 + 256, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    affine.for %i = 0 to %N {
      affine.for %j = 0 to %N {
        %c = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c, %b9 : f64
        affine.store %cs, %C[%i,%j] : memref<?x?xf64>
      }
    }
    // Tiled: i_t(32), j_t(128), k_t(256); order i_t,j_t,k_t,i,k,j.
    // Smaller j tile keeps the B k_t x j_t panel (256x128 = 256KB) plus C strip in L2.
    affine.for %it = 0 to %N step 32 {
      affine.for %jt = 0 to %N step 128 {
        affine.for %kt = 0 to %N step 256 {
          affine.for %i = #lb(%it) to min #ub_i(%it)[%N] {
            affine.for %k = #lb(%kt) to min #ub_k(%kt)[%N] {
              %a = affine.load %A[%i,%k] : memref<?x?xf64>
              affine.for %j = #lb(%jt) to min #ub_j(%jt)[%N] {
                %b = affine.load %B[%k,%j] : memref<?x?xf64>
                %c = affine.load %C[%i,%j] : memref<?x?xf64>
                %p = arith.mulf %a, %b : f64
                %s = arith.addf %c, %p : f64
                affine.store %s, %C[%i,%j] : memref<?x?xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}
