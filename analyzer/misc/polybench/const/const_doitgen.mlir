module attributes { "simulation.prologue" = "volatile double ARRAY_0[60]; volatile double ARRAY_1[50][40][60]; volatile double ARRAY_2[60][60];" } {
  func.func @doitgen(%A: memref<50x40x60xf64>, %C4: memref<60x60xf64>, %sum: memref<60xf64>) {
    // Loop over r
    affine.for %r = 0 to 50 {
      // Loop over q
      affine.for %q = 0 to 40 {
        // Initialize sum to zero for each p
        affine.for %p = 0 to 60 {
          // Set sum[p] to 0.0
          %zero = arith.constant 0.0 : f64
          affine.store %zero, %sum[%p] : memref<60xf64>
          affine.for %s = 0 to 60 {
            // Compute sum[p] += A[r][q][s] * C4[s][p]
            %A_val = affine.load %A[%r, %q, %s] : memref<50x40x60xf64>
            %C4_val = affine.load %C4[%s, %p] : memref<60x60xf64>
            %product = arith.mulf %A_val, %C4_val : f64
            %current_sum = affine.load %sum[%p] : memref<60xf64>
            %new_sum = arith.addf %current_sum, %product : f64
            affine.store %new_sum, %sum[%p] : memref<60xf64>
          }
        }
        affine.for %p = 0 to 60 {
          // Assign sum to A[r][q][p]
          %final_value = affine.load %sum[%p] : memref<60xf64>
          affine.store %final_value, %A[%r, %q, %p] : memref<50x40x60xf64>
        }
      }
    }{ slap.extract }
    return
  }
}
