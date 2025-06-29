module {
  func.func @kernel_gemm(%C: memref<?x?xf64>, 
                         %A: memref<?x?xf64>, 
                         %B: memref<?x?xf64>, %NI : index, %NJ : index, %NK : index) {
  affine.for %loop_once = 0 to 1 {
      affine.for %loop_once = 0 to 1 {

      %alpha = arith.constant 1.0 : f64
        %beta = arith.constant 1.0 : f64
 
        affine.for %i = 0 to %NI {
          
          // First inner loop: C[i][j] *= beta
          affine.for %j = 0 to %NJ {
            %0 = affine.load %C[%i, %j] : memref<?x?xf64>
            %1 = arith.mulf %0, %beta : f64
            affine.store %1, %C[%i, %j] : memref<?x?xf64>
          }
          
          // Second inner loop nest: C[i][j] += alpha * A[i][k] * B[k][j]
          affine.for %k = 0 to %NK {
            affine.for %j = 0 to %NJ {
              %2 = affine.load %C[%i, %j] : memref<?x?xf64>
              %3 = affine.load %A[%i, %k] : memref<?x?xf64>
              %4 = affine.load %B[%k, %j] : memref<?x?xf64>
              %5 = arith.mulf %alpha, %3 : f64
              %6 = arith.mulf %5, %4 : f64
              %7 = arith.addf %2, %6 : f64
              affine.store %7, %C[%i, %j] : memref<?x?xf64>
            }
          }
        }
     }
    return
  }
}
