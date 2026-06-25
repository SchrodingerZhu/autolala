#tri  = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    // Compute phase fused with scale: C[i,j] = 0.9*C[i,j] + sum_k A[i,k]*A[j,k], j<=i.
    // i-j-k order: inner k is a reduction; A[i,:] and A[j,:] are contiguous rows.
    // At small N the A rows fit in L1/L2 so no tiling is needed; the k-reduction
    // vectorizes and C[i,j] stays in a register across the whole k-sweep.
    affine.for %i = 0 to %N {
      affine.for %j = 0 to #tri(%i) {
        %c0 = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c0, %b9 : f64
        %acc = affine.for %k = 0 to %N
                   iter_args(%c = %cs) -> (f64) {
          %aik = affine.load %A[%i,%k] : memref<?x?xf64>
          %ajk = affine.load %A[%j,%k] : memref<?x?xf64>
          %p = arith.mulf %aik, %ajk : f64
          %s = arith.addf %c, %p : f64
          affine.yield %s : f64
        }
        affine.store %acc, %C[%i,%j] : memref<?x?xf64>
      }
    }
    return
  }
}
