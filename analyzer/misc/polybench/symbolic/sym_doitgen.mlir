module {
  func.func @doitgen(%A: memref<?x?x?xf64>, %C4: memref<?x?xf64>, %sum: memref<?xf64>, %NR : index, %NQ : index, %NP : index) {
    // Loop over r
     affine.for %loop_once = 0 to 1 {

        affine.for %r = 0 to %NR {
          // Loop over q
          affine.for %q = 0 to %NQ {
            // Initialize sum to zero for each p
            affine.for %p = 0 to %NP {
              // Set sum[p] to 0.0
              %zero = arith.constant 0.0 : f64
              affine.store %zero, %sum[%p] : memref<?xf64>

              affine.for %s = 0 to %NP {
                // Compute sum[p] += A[r][q][s] * C4[s][p]
                %A_val = affine.load %A[%r, %q, %s] : memref<?x?x?xf64>
                %C4_val = affine.load %C4[%s, %p] : memref<?x?xf64>
                %product = arith.mulf %A_val, %C4_val : f64
                %current_sum = affine.load %sum[%p] : memref<?xf64>
                %new_sum = arith.addf %current_sum, %product : f64
                affine.store %new_sum, %sum[%p] : memref<?xf64>
              }
            }
            affine.for %p = 0 to %NP {
              // Assign sum to A[r][q][p]
              %final_value = affine.load %sum[%p] : memref<?xf64>
              affine.store %final_value, %A[%r, %q, %p] : memref<?x?x?xf64>
            }
          }
        }
     }
    return
  }
}
