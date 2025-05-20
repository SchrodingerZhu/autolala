module {
  func.func @doitgen(%A: memref<82x82x82xf32>, %C4: memref<82x82xf32>, %sum: memref<82xf32>) {
    // Loop over r
    affine.for %r = 0 to 82 {
      // Loop over q
      affine.for %q = 0 to 82 {
        // Initialize sum to zero for each p
        affine.for %p = 0 to 82 {
          // Set sum[p] to 0.0
          %zero = arith.constant 0.0 : f32
          affine.store %zero, %sum[%p] : memref<82xf32>

          affine.for %s = 0 to 82 {
            // Compute sum[p] += A[r][q][s] * C4[s][p]
            %A_val = affine.load %A[%r, %q, %s] : memref<82x82x82xf32>
            %C4_val = affine.load %C4[%s, %p] : memref<82x82xf32>
            %product = arith.mulf %A_val, %C4_val : f32
            %current_sum = affine.load %sum[%p] : memref<82xf32>
            %new_sum = arith.addf %current_sum, %product : f32
            affine.store %new_sum, %sum[%p] : memref<82xf32>
          }
        }
        affine.for %p = 0 to 82 {
          // Assign sum to A[r][q][p]
          %final_value = affine.load %sum[%p] : memref<82xf32>
          affine.store %final_value, %A[%r, %q, %p] : memref<82x82x82xf32>
        }
      }
    }{ slap.extract }
    return
  }
}
