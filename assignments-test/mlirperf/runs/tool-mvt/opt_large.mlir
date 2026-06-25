#lb = affine_map<(d0) -> (d0)>
#ub = affine_map<(d0)[s0] -> (d0 + 512, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %x1: memref<?xf64>, %x2: memref<?xf64>, %y1: memref<?xf64>, %y2: memref<?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    // Fused mvt: each A[i,j] loaded once, updates both x1[i] and x2[j].
    //   x1[i] += A[i,j]*y1[j]            (was pass1)
    //   x2[j] += A[i,j]*y2[i]            (pass2 A[j,i] with index swap)
    // Tiled in both i and j (T=512) so x2[jj..]/y1[jj..] tile stays cache-resident.
    affine.for %ii = 0 to %N step 512 {
      affine.for %jj = 0 to %N step 512 {
        affine.for %i = #lb(%ii) to min #ub(%ii)[%N] {
          %x1i = affine.load %x1[%i] : memref<?xf64>
          %y2i = affine.load %y2[%i] : memref<?xf64>
          %acc = affine.for %j = #lb(%jj) to min #ub(%jj)[%N]
                   iter_args(%s1 = %x1i) -> (f64) {
            %a   = affine.load %A[%i, %j] : memref<?x?xf64>
            %y1j = affine.load %y1[%j] : memref<?xf64>
            %p1  = arith.mulf %a, %y1j fastmath<fast> : f64
            %s1n = arith.addf %s1, %p1 fastmath<fast> : f64
            %x2j = affine.load %x2[%j] : memref<?xf64>
            %p2  = arith.mulf %a, %y2i fastmath<fast> : f64
            %s2n = arith.addf %x2j, %p2 fastmath<fast> : f64
            affine.store %s2n, %x2[%j] : memref<?xf64>
            affine.yield %s1n : f64
          }
          affine.store %acc, %x1[%i] : memref<?xf64>
        }
      }
    }
    return
  }
}
