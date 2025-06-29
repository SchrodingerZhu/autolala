module {
    func.func @kernel_trisolv(%L: memref<?x?xf64>, %x: memref<?xf64>, %b: memref<?xf64>, %n: index) {
        
        // for (i = 0; i < n; i++)
        affine.for %i = 0 to %n {
            // x[i] = b[i];
            %b_i = affine.load %b[%i] : memref<?xf64>
            affine.store %b_i, %x[%i] : memref<?xf64>
            
            // for (j = 0; j < i; j++) - using affine_map for dependent loop
            affine.for %j = 0 to affine_map<(d0) -> (d0)> (%i) {
                // x[i] -= L[i][j] * x[j];
                %x_i = affine.load %x[%i] : memref<?xf64>
                %L_ij = affine.load %L[%i, %j] : memref<?x?xf64>
                %x_j = affine.load %x[%j] : memref<?xf64>
                %prod = arith.mulf %L_ij, %x_j : f64
                %new_x_i = arith.subf %x_i, %prod : f64
                affine.store %new_x_i, %x[%i] : memref<?xf64>
            }
            
            // x[i] = x[i] / L[i][i];
            %x_i_final = affine.load %x[%i] : memref<?xf64>
            %L_ii = affine.load %L[%i, %i] : memref<?x?xf64>
            %result = arith.divf %x_i_final, %L_ii : f64
            affine.store %result, %x[%i] : memref<?xf64>
        } 
        
        return
    }
}
