module {
    func.func @kernel_2mm(%tmp: memref<?x?xf64>, %A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %D: memref<?x?xf64>, %ni: index, %nj: index, %nk: index, %nl: index) {
        affine.for %loop_once = 0 to 1 {

            %alpha = arith.constant 1.0 : f64
            %beta = arith.constant 1.0 : f64
            %c0 = arith.constant 0.0 : f64
            
            // First matrix multiplication: tmp = alpha * A * B
            // for (i = 0; i < ni; i++)
            affine.for %i = 0 to %ni {
                // for (j = 0; j < nj; j++)
                affine.for %j = 0 to %nj {
                    // tmp[i][j] = 0.0;
                    affine.store %c0, %tmp[%i, %j] : memref<?x?xf64>
                    
                    // for (k = 0; k < nk; k++)
                    affine.for %k = 0 to %nk {
                        // tmp[i][j] += alpha * A[i][k] * B[k][j];
                        %tmp_ij = affine.load %tmp[%i, %j] : memref<?x?xf64>
                        %A_ik = affine.load %A[%i, %k] : memref<?x?xf64>
                        %B_kj = affine.load %B[%k, %j] : memref<?x?xf64>
                        %prod1 = arith.mulf %alpha, %A_ik : f64
                        %prod2 = arith.mulf %prod1, %B_kj : f64
                        %new_tmp_ij = arith.addf %tmp_ij, %prod2 : f64
                        affine.store %new_tmp_ij, %tmp[%i, %j] : memref<?x?xf64>
                    }
                }
            }
            
            // Second matrix multiplication: D = tmp * C + beta * D
            // for (i = 0; i < ni; i++)
            affine.for %i = 0 to %ni {
                // for (j = 0; j < nl; j++)
                affine.for %j = 0 to %nl {
                    // D[i][j] *= beta;
                    %D_ij = affine.load %D[%i, %j] : memref<?x?xf64>
                    %scaled_D_ij = arith.mulf %D_ij, %beta : f64
                    affine.store %scaled_D_ij, %D[%i, %j] : memref<?x?xf64>
                    
                    // for (k = 0; k < nj; k++)
                    affine.for %k = 0 to %nj {
                        // D[i][j] += tmp[i][k] * C[k][j];
                        %D_ij_current = affine.load %D[%i, %j] : memref<?x?xf64>
                        %tmp_ik = affine.load %tmp[%i, %k] : memref<?x?xf64>
                        %C_kj = affine.load %C[%k, %j] : memref<?x?xf64>
                        %prod = arith.mulf %tmp_ik, %C_kj : f64
                        %new_D_ij = arith.addf %D_ij_current, %prod : f64
                        affine.store %new_D_ij, %D[%i, %j] : memref<?x?xf64>
                    }
                }
            } 
        }
        return
    }
}
