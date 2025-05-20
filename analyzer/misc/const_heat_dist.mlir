module {
  func.func @heat_distribution(%A: memref<100x100xf32>) {
    // Iterate over 100 rounds
    affine.for %r = 0 to 100 {
      
      // First update even-indexed elements (skipping the boundary elements)
      affine.for %i = 1 to 99 step 2 {
        affine.for %j = 1 to 99 step 2 {
          // Load neighbors (top, bottom, left, right, and center)
          %top    = affine.load %A[%i - 1, %j] : memref<100x100xf32>
          %bottom = affine.load %A[%i + 1, %j] : memref<100x100xf32>
          %left   = affine.load %A[%i, %j - 1] : memref<100x100xf32>
          %right  = affine.load %A[%i, %j + 1] : memref<100x100xf32>
          %center = affine.load %A[%i, %j] : memref<100x100xf32>

          // Compute the sum and average the values for heat distribution
          %c5 = arith.constant 5.0 : f32
          %sum1    = arith.addf %top, %bottom : f32
          %sum2    = arith.addf %sum1, %left : f32
          %sum3    = arith.addf %sum2, %right : f32
          %sum4    = arith.addf %sum3, %center : f32
          %avg     = arith.divf %sum4, %c5 : f32

          // Store the updated value back to matrix A
          affine.store %avg, %A[%i, %j] : memref<100x100xf32>
        }
      }

      // Then update odd-indexed elements (skipping the boundary elements)
      affine.for %i = 2 to 98 step 2 {
        affine.for %j = 2 to 98 step 2 {
          // Load neighbors (top, bottom, left, right, and center)
          %top_odd    = affine.load %A[%i - 1, %j] : memref<100x100xf32>
          %bottom_odd = affine.load %A[%i + 1, %j] : memref<100x100xf32>
          %left_odd   = affine.load %A[%i, %j - 1] : memref<100x100xf32>
          %right_odd  = affine.load %A[%i, %j + 1] : memref<100x100xf32>
          %center_odd = affine.load %A[%i, %j] : memref<100x100xf32>

          // Compute the sum and average the values for heat distribution
          %c5 = arith.constant 5.0 : f32
          %sum1_odd    = arith.addf %top_odd, %bottom_odd : f32
          %sum2_odd    = arith.addf %sum1_odd, %left_odd : f32
          %sum3_odd    = arith.addf %sum2_odd, %right_odd : f32
          %sum4_odd    = arith.addf %sum3_odd, %center_odd : f32
          %avg_odd     = arith.divf %sum4_odd, %c5 : f32

          // Store the updated value back to matrix A
          affine.store %avg_odd, %A[%i, %j] : memref<100x100xf32>
        }
      }
    } { slap.extract }
    return
  }
}
