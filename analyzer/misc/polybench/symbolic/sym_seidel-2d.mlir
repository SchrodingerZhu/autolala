module {
    func.func @kernel_seidel_2d(%A: memref<?x?xf64>, %tsteps: index, %n: index) {

        affine.for %loop_once = 0 to 1 {
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %divisor = arith.constant 9.0 : f64
            
            %tsteps_minus_1 = arith.subi %tsteps, %c1 : index
            %n_minus_2 = arith.subi %n, %c2 : index
            %tsteps_upper = arith.addi %tsteps_minus_1, %c1 : index
            %n_upper = arith.addi %n_minus_2, %c1 : index
            
            // for (t = 0; t <= tsteps - 1; t++)
            affine.for %t = 0 to %tsteps_upper {
                // for (i = 1; i <= n - 2; i++)
                affine.for %i = 1 to %n_upper {
                    // for (j = 1; j <= n - 2; j++)
                    affine.for %j = 1 to %n_upper {
                        // Load all 9 neighboring values
                        %A_i_minus_1_j_minus_1 = affine.load %A[%i - 1, %j - 1] : memref<?x?xf64>
                        %A_i_minus_1_j = affine.load %A[%i - 1, %j] : memref<?x?xf64>
                        %A_i_minus_1_j_plus_1 = affine.load %A[%i - 1, %j + 1] : memref<?x?xf64>
                        %A_i_j_minus_1 = affine.load %A[%i, %j - 1] : memref<?x?xf64>
                        %A_i_j = affine.load %A[%i, %j] : memref<?x?xf64>
                        %A_i_j_plus_1 = affine.load %A[%i, %j + 1] : memref<?x?xf64>
                        %A_i_plus_1_j_minus_1 = affine.load %A[%i + 1, %j - 1] : memref<?x?xf64>
                        %A_i_plus_1_j = affine.load %A[%i + 1, %j] : memref<?x?xf64>
                        %A_i_plus_1_j_plus_1 = affine.load %A[%i + 1, %j + 1] : memref<?x?xf64>
                        
                        // Sum all 9 values
                        %sum1 = arith.addf %A_i_minus_1_j_minus_1, %A_i_minus_1_j : f64
                        %sum2 = arith.addf %sum1, %A_i_minus_1_j_plus_1 : f64
                        %sum3 = arith.addf %sum2, %A_i_j_minus_1 : f64
                        %sum4 = arith.addf %sum3, %A_i_j : f64
                        %sum5 = arith.addf %sum4, %A_i_j_plus_1 : f64
                        %sum6 = arith.addf %sum5, %A_i_plus_1_j_minus_1 : f64
                        %sum7 = arith.addf %sum6, %A_i_plus_1_j : f64
                        %sum8 = arith.addf %sum7, %A_i_plus_1_j_plus_1 : f64
                        
                        // Divide by 9.0
                        %result = arith.divf %sum8, %divisor : f64
                        
                        // Store the result
                        affine.store %result, %A[%i, %j] : memref<?x?xf64>
                    }
                }
            } 
        }
        return
    }
}
