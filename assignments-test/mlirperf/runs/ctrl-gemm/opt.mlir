#lb = affine_map<(d0) -> (d0)>
#ub64 = affine_map<(d0)[s0] -> (d0 + 64, s0)>
#ub256 = affine_map<(d0)[s0] -> (d0 + 256, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    // Scale C by 0.9
    affine.for %i = 0 to %N {
      affine.for %j = 0 to %N {
        %c = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c, %b9 : f64
        affine.store %cs, %C[%i,%j] : memref<?x?xf64>
      }
    }
    // Tiled GEMM: tile i (64), j (256), k (256); order i_t, j_t, k_t, i, k, j
    affine.for %it = 0 to %N step 64 {
      affine.for %jt = 0 to %N step 256 {
        affine.for %kt = 0 to %N step 256 {
          affine.for %i = #lb(%it) to min #ub64(%it)[%N] {
            affine.for %k = #lb(%kt) to min #ub256(%kt)[%N] {
              %a = affine.load %A[%i,%k] : memref<?x?xf64>
              affine.for %j = #lb(%jt) to min #ub256(%jt)[%N] {
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
