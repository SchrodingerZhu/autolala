module {
    func.func @kernel_heat_3d(%A: memref<?x?x?xf64>, %B: memref<?x?x?xf64>, %tsteps: index, %n: index) {
        affine.for %loop_once = 0 to 1 {
            %c0_125 = arith.constant 0.125 : f64
            %c2_0 = arith.constant 2.0 : f64
            %c1 = arith.constant 1 : index
            
            %n_minus_1 = arith.subi %n, %c1 : index
            %tsteps_plus_1 = arith.addi %tsteps, %c1 : index
            
            // for (t = 1; t <= tsteps; t++)
            affine.for %t = 1 to %tsteps_plus_1 {
                // First nested loop: compute B from A
                // for (i = 1; i < n-1; i++)
                affine.for %i = 1 to %n_minus_1 {
                    // for (j = 1; j < n-1; j++)
                    affine.for %j = 1 to %n_minus_1 {
                        // for (k = 1; k < n-1; k++)
                        affine.for %k = 1 to %n_minus_1 {
                            // Load all required A values
                            %A_ijk = affine.load %A[%i, %j, %k] : memref<?x?x?xf64>
                            %A_i_plus_1_jk = affine.load %A[%i + 1, %j, %k] : memref<?x?x?xf64>
                            %A_i_minus_1_jk = affine.load %A[%i - 1, %j, %k] : memref<?x?x?xf64>
                            %A_ij_plus_1_k = affine.load %A[%i, %j + 1, %k] : memref<?x?x?xf64>
                            %A_ij_minus_1_k = affine.load %A[%i, %j - 1, %k] : memref<?x?x?xf64>
                            %A_ijk_plus_1 = affine.load %A[%i, %j, %k + 1] : memref<?x?x?xf64>
                            %A_ijk_minus_1 = affine.load %A[%i, %j, %k - 1] : memref<?x?x?xf64>
                            
                            // Compute first term: 0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                            %two_A_ijk_1 = arith.mulf %c2_0, %A_ijk : f64
                            %diff1_1 = arith.subf %A_i_plus_1_jk, %two_A_ijk_1 : f64
                            %diff1_2 = arith.addf %diff1_1, %A_i_minus_1_jk : f64
                            %term1 = arith.mulf %c0_125, %diff1_2 : f64
                            
                            // Compute second term: 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                            %two_A_ijk_2 = arith.mulf %c2_0, %A_ijk : f64
                            %diff2_1 = arith.subf %A_ij_plus_1_k, %two_A_ijk_2 : f64
                            %diff2_2 = arith.addf %diff2_1, %A_ij_minus_1_k : f64
                            %term2 = arith.mulf %c0_125, %diff2_2 : f64
                            
                            // Compute third term: 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                            %two_A_ijk_3 = arith.mulf %c2_0, %A_ijk : f64
                            %diff3_1 = arith.subf %A_ijk_plus_1, %two_A_ijk_3 : f64
                            %diff3_2 = arith.addf %diff3_1, %A_ijk_minus_1 : f64
                            %term3 = arith.mulf %c0_125, %diff3_2 : f64
                            
                            // Sum all terms: term1 + term2 + term3 + A[i][j][k]
                            %sum1 = arith.addf %term1, %term2 : f64
                            %sum2 = arith.addf %sum1, %term3 : f64
                            %result_B = arith.addf %sum2, %A_ijk : f64
                            
                            affine.store %result_B, %B[%i, %j, %k] : memref<?x?x?xf64>
                        }
                    }
                }
                
                // Second nested loop: compute A from B
                // for (i = 1; i < n-1; i++)
                affine.for %i = 1 to %n_minus_1 {
                    // for (j = 1; j < n-1; j++)
                    affine.for %j = 1 to %n_minus_1 {
                        // for (k = 1; k < n-1; k++)
                        affine.for %k = 1 to %n_minus_1 {
                            // Load all required B values
                            %B_ijk = affine.load %B[%i, %j, %k] : memref<?x?x?xf64>
                            %B_i_plus_1_jk = affine.load %B[%i + 1, %j, %k] : memref<?x?x?xf64>
                            %B_i_minus_1_jk = affine.load %B[%i - 1, %j, %k] : memref<?x?x?xf64>
                            %B_ij_plus_1_k = affine.load %B[%i, %j + 1, %k] : memref<?x?x?xf64>
                            %B_ij_minus_1_k = affine.load %B[%i, %j - 1, %k] : memref<?x?x?xf64>
                            %B_ijk_plus_1 = affine.load %B[%i, %j, %k + 1] : memref<?x?x?xf64>
                            %B_ijk_minus_1 = affine.load %B[%i, %j, %k - 1] : memref<?x?x?xf64>
                            
                            // Compute first term: 0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                            %two_B_ijk_1 = arith.mulf %c2_0, %B_ijk : f64
                            %diff1_1_B = arith.subf %B_i_plus_1_jk, %two_B_ijk_1 : f64
                            %diff1_2_B = arith.addf %diff1_1_B, %B_i_minus_1_jk : f64
                            %term1_B = arith.mulf %c0_125, %diff1_2_B : f64
                            
                            // Compute second term: 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                            %two_B_ijk_2 = arith.mulf %c2_0, %B_ijk : f64
                            %diff2_1_B = arith.subf %B_ij_plus_1_k, %two_B_ijk_2 : f64
                            %diff2_2_B = arith.addf %diff2_1_B, %B_ij_minus_1_k : f64
                            %term2_B = arith.mulf %c0_125, %diff2_2_B : f64
                            
                            // Compute third term: 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                            %two_B_ijk_3 = arith.mulf %c2_0, %B_ijk : f64
                            %diff3_1_B = arith.subf %B_ijk_plus_1, %two_B_ijk_3 : f64
                            %diff3_2_B = arith.addf %diff3_1_B, %B_ijk_minus_1 : f64
                            %term3_B = arith.mulf %c0_125, %diff3_2_B : f64
                            
                            // Sum all terms: term1 + term2 + term3 + B[i][j][k]
                            %sum1_B = arith.addf %term1_B, %term2_B : f64
                            %sum2_B = arith.addf %sum1_B, %term3_B : f64
                            %result_A = arith.addf %sum2_B, %B_ijk : f64
                            
                            affine.store %result_A, %A[%i, %j, %k] : memref<?x?x?xf64>
                        }
                    }
                }
            } 
        }
        return
    }
}
