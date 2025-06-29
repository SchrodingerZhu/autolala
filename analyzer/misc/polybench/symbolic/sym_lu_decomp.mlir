module {
  func.func @lu_decomposition(%A: memref<?x?xf64>, %len: index) {
    affine.for %loop_once = 0 to 1 {
      // Iterate over columns (k-th column of A)
      affine.for %k = 0 to %len {
        // Update the upper triangular part (U) in the k-th row
        affine.for %j = affine_map<(d0) -> (d0)> (%k) to %len {
          // A[k, j] remains as U[k, j] for k-th row
          %A_kj = affine.load %A[%k, %j] : memref<?x?xf64>
          affine.store %A_kj, %A[%k, %j] : memref<?x?xf64>
        }

        // Update the lower triangular part (L) in the k-th column
        affine.for %i = affine_map<(d0) -> (d0 + 1)> (%k) to %len {
          // Compute L[i, k] = A[i, k] / U[k, k]
          %A_ik = affine.load %A[%i, %k] : memref<?x?xf64>
          %A_kk = affine.load %A[%k, %k] : memref<?x?xf64>
          %L_ik = arith.divf %A_ik, %A_kk : f64
          affine.store %L_ik, %A[%i, %k] : memref<?x?xf64>

          // Update the rest of the A matrix using the new L and U values
          affine.for %j = affine_map<(d0) -> (d0 + 1)> (%k) to %len {
            // A[i, j] -= L[i, k] * U[k, j]
            %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
            %U_kj = affine.load %A[%k, %j] : memref<?x?xf64>
            %L_ik_new = affine.load %A[%i, %k] : memref<?x?xf64>
            %product = arith.mulf %L_ik_new, %U_kj : f64
            %A_ij_new = arith.subf %A_ij, %product : f64
            affine.store %A_ij_new, %A[%i, %j] : memref<?x?xf64>
          }
        }
      } 
    }
    return
  }
}
