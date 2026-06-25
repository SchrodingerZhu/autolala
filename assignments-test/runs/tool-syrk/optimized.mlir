module {
    func.func @kernel_syrk(%C: memref<?x?xf64>, %A: memref<?x?xf64>, %n: index, %m: index) {
        affine.for %i = 0 to %n {
            %alpha = arith.constant 1.0 : f64
            %beta = arith.constant 1.0 : f64

            // C[i][j] *= beta;  (unchanged init sweep over the triangular j range)
            affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
                %C_ij = affine.load %C[%i, %j] : memref<?x?xf64>
                %new_C_ij = arith.mulf %C_ij, %beta : f64
                affine.store %new_C_ij, %C[%i, %j] : memref<?x?xf64>
            }

            // Loop interchange: j now OUTSIDE k.
            // For each (i, j) we accumulate C[i][j] over the entire k-sweep,
            // keeping C[i][j] resident (reuse-distance ~ register/L1) across all m
            // updates instead of re-streaming the whole triangular C row per k.
            affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
                affine.for %k = 0 to %m {
                    // C[i][j] += alpha * A[i][k] * A[j][k];
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
