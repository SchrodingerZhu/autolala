#tri  = affine_map<(d0) -> (d0 + 1)>
#id   = affine_map<(d0) -> (d0)>
#iub  = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#jub  = affine_map<(d0, d1) -> (d0 + 32, d1 + 1)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    // Scale phase.
    affine.for %i = 0 to %N {
      affine.for %j = 0 to #tri(%i) {
        %c = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c, %b9 : f64
        affine.store %cs, %C[%i,%j] : memref<?x?xf64>
      }
    }
    // Compute: tile i and j (k kept full so C[i,j] accumulates in a register).
    // A[jt-block,:] is reused across the whole i-block, A[it-block,:] across the j-block.
    affine.for %it = 0 to %N step 32 {
      affine.for %jt = 0 to min #iub(%it)[%N] step 32 {
        affine.for %i = #id(%it) to min #iub(%it)[%N] {
          affine.for %j = #id(%jt) to min #jub(%jt, %i) {
            %c0 = affine.load %C[%i,%j] : memref<?x?xf64>
            %acc = affine.for %k = 0 to %N
                       iter_args(%c = %c0) -> (f64) {
              %aik = affine.load %A[%i,%k] : memref<?x?xf64>
              %ajk = affine.load %A[%j,%k] : memref<?x?xf64>
              %p = arith.mulf %aik, %ajk : f64
              %s = arith.addf %c, %p : f64
              affine.yield %s : f64
            }
            affine.store %acc, %C[%i,%j] : memref<?x?xf64>
          }
        }
      }
    }
    return
  }
}
