#tri  = affine_map<(d0) -> (d0 + 1)>
#id   = affine_map<(d0) -> (d0)>
#iub  = affine_map<(d0)[s0] -> (d0 + 64, s0)>
#kub  = affine_map<(d0)[s0] -> (d0 + 256, s0)>
#jub  = affine_map<(d0, d1) -> (d0 + 64, d1 + 1)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    // Scale phase: C[i,j] *= 0.9 for j<=i
    affine.for %i = 0 to %N {
      affine.for %j = 0 to #tri(%i) {
        %c = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c, %b9 : f64
        affine.store %cs, %C[%i,%j] : memref<?x?xf64>
      }
    }
    // Compute phase tiled it/jt/kt, inner i-j-k with k reduction in register.
    affine.for %it = 0 to %N step 64 {
      affine.for %jt = 0 to min #iub(%it)[%N] step 64 {
        affine.for %kt = 0 to %N step 256 {
          affine.for %i = #id(%it) to min #iub(%it)[%N] {
            affine.for %j = #id(%jt) to min #jub(%jt, %i) {
              %c0 = affine.load %C[%i,%j] : memref<?x?xf64>
              %acc = affine.for %k = #id(%kt) to min #kub(%kt)[%N]
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
    }
    return
  }
}
