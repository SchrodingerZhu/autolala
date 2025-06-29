module {
    func.func @kernel_jacobi_2d(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %tsteps: index, %n: index) {
        affine.for %loop_once = 0 to 1 {   
            %c0_2 = arith.constant 0.2 : f64
            %c1 = arith.constant 1 : index
            %n_minus_1 = arith.subi %n, %c1 : index
            
            // for (t = 0; t < tsteps; t++)
            affine.for %t = 0 to %tsteps {
                // for (i = 1; i < n - 1; i++)
                affine.for %i = 1 to %n_minus_1 {
                    // for (j = 1; j < n - 1; j++)
                    affine.for %j = 1 to %n_minus_1 {
                        // B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
                        %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
                        %A_ij_minus1 = affine.load %A[%i, %j - 1] : memref<?x?xf64>
                        %A_i_jplus1 = affine.load %A[%i, %j + 1] : memref<?x?xf64>
                        %A_iplus1_j = affine.load %A[%i + 1, %j] : memref<?x?xf64>
                        %A_iminus1_j = affine.load %A[%i - 1, %j] : memref<?x?xf64>
                        
                        %sum1 = arith.addf %A_ij, %A_ij_minus1 : f64
                        %sum2 = arith.addf %sum1, %A_i_jplus1 : f64
                        %sum3 = arith.addf %sum2, %A_iplus1_j : f64
                        %sum4 = arith.addf %sum3, %A_iminus1_j : f64
                        %result_B = arith.mulf %c0_2, %sum4 : f64
                        
                        affine.store %result_B, %B[%i, %j] : memref<?x?xf64>
                    }
                }
                
                // for (i = 1; i < n - 1; i++)
                affine.for %i = 1 to %n_minus_1 {
                    // for (j = 1; j < n - 1; j++)
                    affine.for %j = 1 to %n_minus_1 {
                        // A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
                        %B_ij = affine.load %B[%i, %j] : memref<?x?xf64>
                        %B_ij_minus1 = affine.load %B[%i, %j - 1] : memref<?x?xf64>
                        %B_i_jplus1 = affine.load %B[%i, %j + 1] : memref<?x?xf64>
                        %B_iplus1_j = affine.load %B[%i + 1, %j] : memref<?x?xf64>
                        %B_iminus1_j = affine.load %B[%i - 1, %j] : memref<?x?xf64>
                        
                        %sum1 = arith.addf %B_ij, %B_ij_minus1 : f64
                        %sum2 = arith.addf %sum1, %B_i_jplus1 : f64
                        %sum3 = arith.addf %sum2, %B_iplus1_j : f64
                        %sum4 = arith.addf %sum3, %B_iminus1_j : f64
                        %result_A = arith.mulf %c0_2, %sum4 : f64
                        
                        affine.store %result_A, %A[%i, %j] : memref<?x?xf64>
                    }
                }
            }
        }
        return
    }
}
