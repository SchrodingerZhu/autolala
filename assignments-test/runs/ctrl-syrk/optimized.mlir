module {
    func.func @kernel_syrk(%C: memref<?x?xf64>, %A: memref<?x?xf64>, %n: index, %m: index) {
        affine.for %i = 0 to %n {
            %alpha = arith.constant 1.0 : f64
            %beta = arith.constant 1.0 : f64

            // C[i][j] *= beta;  (j <= i, triangular lower part of row i)
            affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
                %C_ij = affine.load %C[%i, %j] : memref<?x?xf64>
                %new_C_ij = arith.mulf %C_ij, %beta : f64
                affine.store %new_C_ij, %C[%i, %j] : memref<?x?xf64>
            }

            // Compute: C[i][j] += alpha * A[i][k] * A[j][k].
            // Interchanged from the original (k, j) order to (j, k):
            // for a fixed (i, j) the same C[i][j] cell is loaded/updated/stored
            // across the entire k loop, so it stays resident the whole time
            // (temporal reuse distance = 1) instead of being re-fetched once
            // per k. A[j][k] is also swept contiguously along row j of A.
            affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
                affine.for %k = 0 to %m {
                    %C_ij = affine.load %C[%i, %j] : memref<?x?xf64>
                    %A_ik = affine.load %A[%i, %k] : memref<?x?xf64>
                    %A_jk = affine.load %A[%j, %k] : memref<?x?xf64>
                    %prod1 = arith.mulf %alpha, %A_ik : f64
                    %prod2 = arith.mulf %prod1, %A_jk : f64
                    %new_C_ij = arith.addf %C_ij, %prod2 : f64
                    affine.store %new_C_ij, %C[%i, %j] : memref<?x?xf64>
                }
            }
        } {dmd.extract}
        return
    }
}
