module {
  func.func @conv2d_kernel(%input: memref<?x?xf32>, %filter: memref<?x?xf32>, %output: memref<?x?xf32>, %N: index, %K: index) {
    // Treat N and K as symbols for affine operations
    affine.for %i = 0 to affine_map<()[s0, s1] -> (s0 - s1 + 1)>()[%N, %K] {
      affine.for %j = 0 to affine_map<()[s0, s1] -> (s0 - s1 + 1)>()[%N, %K] {
        %zero = arith.constant 0.0 : f32
        %acc = affine.for %fi = 0 to %K iter_args(%acc = %zero) -> (f32) {
          %acc_inner = affine.for %fj = 0 to %K iter_args(%acc_inner = %acc) -> (f32) {
            %filter_val = affine.load %filter[%fi, %fj] : memref<?x?xf32>
            %input_val = affine.load %input[%i + %fi, %j + %fj] : memref<?x?xf32>
            %prod = arith.mulf %input_val, %filter_val : f32
            %new_acc = arith.addf %acc_inner, %prod : f32
            affine.yield %new_acc : f32
          }
          affine.yield %acc_inner : f32
        }
        affine.store %acc, %output[%i, %j] : memref<?x?xf32>
      }
    } { slap.extract }
    return
  }
}
