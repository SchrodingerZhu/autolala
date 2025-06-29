module {
    func.func @kernel_lu(%A: memref<?x?xf64>, %n: index) {
        
        affine.for %loop_once = 0 to 1 {
            affine.for %i = 0 to %n {
                // for (j = 0; j < i; j++) - using affine_map for dependent loop
                affine.for %j = 0 to affine_map<(d0) -> (d0)> (%i) {
                    // for (k = 0; k < j; k++) - using affine_map for dependent loop
                    affine.for %k = 0 to affine_map<(d0) -> (d0)> (%j) {
                        // A[i][j] -= A[i][k] * A[k][j];
                        %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
                        %A_ik = affine.load %A[%i, %k] : memref<?x?xf64>
                        %A_kj = affine.load %A[%k, %j] : memref<?x?xf64>
                        %prod = arith.mulf %A_ik, %A_kj : f64
                        %new_A_ij = arith.subf %A_ij, %prod : f64
                        affine.store %new_A_ij, %A[%i, %j] : memref<?x?xf64>
                    }
                    
                    // A[i][j] /= A[j][j];
                    %A_ij_final = affine.load %A[%i, %j] : memref<?x?xf64>
                    %A_jj = affine.load %A[%j, %j] : memref<?x?xf64>
                    %result = arith.divf %A_ij_final, %A_jj : f64
                    affine.store %result, %A[%i, %j] : memref<?x?xf64>
                }
                
                // for (j = i; j < n; j++) - using affine_map for dependent loop
                affine.for %j = affine_map<(d0) -> (d0)> (%i) to %n {
                    // for (k = 0; k < i; k++) - using affine_map for dependent loop
                    affine.for %k = 0 to affine_map<(d0) -> (d0)> (%i) {
                        // A[i][j] -= A[i][k] * A[k][j];
                        %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
                        %A_ik = affine.load %A[%i, %k] : memref<?x?xf64>
                        %A_kj = affine.load %A[%k, %j] : memref<?x?xf64>
                        %prod = arith.mulf %A_ik, %A_kj : f64
                        %new_A_ij = arith.subf %A_ij, %prod : f64
                        affine.store %new_A_ij, %A[%i, %j] : memref<?x?xf64>
                    }
                }
            } 
        }
        return
    }
}
