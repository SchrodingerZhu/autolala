module {
    func.func @kernel_trmm(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %m: index, %n: index) {

        affine.for %loop_once = 0 to 1 {
            %alpha = arith.constant 1.0 : f64
            
            // for (i = 0; i < m; i++)
            affine.for %i = 0 to %m {
                // for (j = 0; j < n; j++)
                affine.for %j = 0 to %n {
                    // for (k = i+1; k < m; k++) - using affine_map for dependent loop
                    affine.for %k = affine_map<(d0) -> (d0 + 1)> (%i) to %m {
                        // B[i][j] += A[k][i] * B[k][j];
                        %B_ij = affine.load %B[%i, %j] : memref<?x?xf64>
                        %A_ki = affine.load %A[%k, %i] : memref<?x?xf64>
                        %B_kj = affine.load %B[%k, %j] : memref<?x?xf64>
                        %prod = arith.mulf %A_ki, %B_kj : f64
                        %new_B_ij = arith.addf %B_ij, %prod : f64
                        affine.store %new_B_ij, %B[%i, %j] : memref<?x?xf64>
                    }
                    
                    // B[i][j] = alpha * B[i][j];
                    %B_ij_final = affine.load %B[%i, %j] : memref<?x?xf64>
                    %result = arith.mulf %alpha, %B_ij_final : f64
                    affine.store %result, %B[%i, %j] : memref<?x?xf64>
                }
            } 
        }    
       return    
    }
}
