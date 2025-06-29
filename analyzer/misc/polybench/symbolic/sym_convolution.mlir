module {
  func.func @conv2d_kernel(%input: memref<?x?xf32>, %filter: memref<?x?xf32>, %output: memref<?x?xf32>, %N : index, %K : index) {
    affine.for %loop_once = 0 to 1 {
      %n_minusk_i = arith.subi %N, %K : index
      affine.for %i = 0 to %n_minusk_i {
        affine.for %j = 0 to %n_minusk_i {
          // Use affine.parallel to accumulate values into %acc using iter_args
          %zero = arith.constant 0.0 : f32
          %acc = affine.for %fi = 0 to %K iter_args(%acc = %zero) -> (f32) {
            %acc_inner = affine.for %fj = 0 to %K iter_args(%acc_inner = %acc) -> (f32) {
              // Load filter value
              %filter_val = affine.load %filter[%fi, %fj] : memref<?x?xf32>

              // Load corresponding input value from the input matrix
              %input_val = affine.load %input[%i + %fi, %j + %fj] : memref<?x?xf32>

              // Multiply input value with filter value
              %prod = arith.mulf %input_val, %filter_val : f32

              // Add product to the accumulator
              %new_acc = arith.addf %acc_inner, %prod : f32
              affine.yield %new_acc : f32
            }
            affine.yield %acc_inner : f32
          }

          // Store the accumulated result in the output matrix
          affine.store %acc, %output[%i, %j] : memref<?x?xf32>
        }
      } 
    }
    return
  }
}
