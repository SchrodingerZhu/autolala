module {
    func.func @kernel_syrk(%C: memref<?x?xf64>, %A: memref<?x?xf64>, %n: index, %m: index) {
        affine.for %loop_once = 0 to 1 {
            %alpha = arith.constant 1.0 : f64
            %beta = arith.constant 1.0 : f64
            
            // for (i = 0; i < n; i++)
            affine.for %i = 0 to %n {
                // for (j = 0; j <= i; j++) - using affine_map for dependent loop
                affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
                    // C[i][j] *= beta;
                    %C_ij = affine.load %C[%i, %j] : memref<?x?xf64>
                    %new_C_ij = arith.mulf %C_ij, %beta : f64
                    affine.store %new_C_ij, %C[%i, %j] : memref<?x?xf64>
                }
                
                // for (k = 0; k < m; k++)
                affine.for %k = 0 to %m {
                    // for (j = 0; j <= i; j++) - using affine_map for dependent loop
                    affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
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
            }
        }
        return
    }
}
