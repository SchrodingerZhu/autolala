module {
    func.func @kernel_syr2k(%C: memref<?x?xf64>, %A: memref<?x?xf64>, %B: memref<?x?xf64>, %n: index, %m: index) {
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
                        // C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
                        %C_ij = affine.load %C[%i, %j] : memref<?x?xf64>
                        %A_jk = affine.load %A[%j, %k] : memref<?x?xf64>
                        %B_ik = affine.load %B[%i, %k] : memref<?x?xf64>
                        %B_jk = affine.load %B[%j, %k] : memref<?x?xf64>
                        %A_ik = affine.load %A[%i, %k] : memref<?x?xf64>
                        
                        // First term: A[j][k]*alpha*B[i][k]
                        %prod1 = arith.mulf %A_jk, %alpha : f64
                        %term1 = arith.mulf %prod1, %B_ik : f64
                        
                        // Second term: B[j][k]*alpha*A[i][k]
                        %prod2 = arith.mulf %B_jk, %alpha : f64
                        %term2 = arith.mulf %prod2, %A_ik : f64
                        
                        // Sum both terms
                        %sum_terms = arith.addf %term1, %term2 : f64
                        %new_C_ij = arith.addf %C_ij, %sum_terms : f64
                        affine.store %new_C_ij, %C[%i, %j] : memref<?x?xf64>
                    }
                }
            }
        }
        return
    }
}
