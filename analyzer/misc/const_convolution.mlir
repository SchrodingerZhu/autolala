module {
  func.func @conv2d_kernel(%input: memref<1024x1024xf32>, %filter: memref<16x16xf32>, %output: memref<1009x1009xf32>) {
    // Loop over the output matrix dimensions (1009x1009)
    affine.for %i = 0 to 1009 {
      affine.for %j = 0 to 1009 {
        // Use affine.parallel to accumulate values into %acc using iter_args
        %zero = arith.constant 0.0 : f32
        %acc = affine.for %fi = 0 to 16 iter_args(%acc = %zero) -> (f32) {
          %acc_inner = affine.for %fj = 0 to 16 iter_args(%acc_inner = %acc) -> (f32) {
            // Load filter value
            %filter_val = affine.load %filter[%fi, %fj] : memref<16x16xf32>

            // Load corresponding input value from the input matrix
            %input_val = affine.load %input[%i + %fi, %j + %fj] : memref<1024x1024xf32>

            // Multiply input value with filter value
            %prod = arith.mulf %input_val, %filter_val : f32

            // Add product to the accumulator
            %new_acc = arith.addf %acc_inner, %prod : f32
            affine.yield %new_acc : f32
          }
          affine.yield %acc_inner : f32
        }

        // Store the accumulated result in the output matrix
        affine.store %acc, %output[%i, %j] : memref<1009x1009xf32>
      }
    } { slap.extract }
    return
  }
}
