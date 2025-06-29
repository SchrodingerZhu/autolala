module {
    func.func @kernel_jacobi_1d(%A: memref<?xf64>, %B: memref<?xf64>, %tsteps: index, %n: index) {
        affine.for %loop_once = 0 to 1 {
            %coeff = arith.constant 0.33333 : f64
            %c1 = arith.constant 1 : index
            
            %n_minus_1 = arith.subi %n, %c1 : index
            
            // for (t = 0; t < tsteps; t++)
            affine.for %t = 0 to %tsteps {
                // for (i = 1; i < n - 1; i++)
                affine.for %i = 1 to %n_minus_1 {
                    // B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
                    %A_i_minus_1 = affine.load %A[%i - 1] : memref<?xf64>
                    %A_i = affine.load %A[%i] : memref<?xf64>
                    %A_i_plus_1 = affine.load %A[%i + 1] : memref<?xf64>
                    
                    %sum1 = arith.addf %A_i_minus_1, %A_i : f64
                    %sum2 = arith.addf %sum1, %A_i_plus_1 : f64
                    %result_B = arith.mulf %coeff, %sum2 : f64
                    affine.store %result_B, %B[%i] : memref<?xf64>
                }
                
                // for (i = 1; i < n - 1; i++)
                affine.for %i = 1 to %n_minus_1 {
                    // A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
                    %B_i_minus_1 = affine.load %B[%i - 1] : memref<?xf64>
                    %B_i = affine.load %B[%i] : memref<?xf64>
                    %B_i_plus_1 = affine.load %B[%i + 1] : memref<?xf64>
                    
                    %sum1 = arith.addf %B_i_minus_1, %B_i : f64
                    %sum2 = arith.addf %sum1, %B_i_plus_1 : f64
                    %result_A = arith.mulf %coeff, %sum2 : f64
                    affine.store %result_A, %A[%i] : memref<?xf64>
                }
            } 
        }
        return
    }
}
