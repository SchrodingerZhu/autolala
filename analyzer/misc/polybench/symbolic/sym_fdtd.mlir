module {
    func.func @kernel_fdtd_2d(%ex: memref<?x?xf64>, %ey: memref<?x?xf64>, %hz: memref<?x?xf64>, %fict: memref<?xf64>, %tmax: index, %nx: index, %ny: index) {
    affine.for %loop_once = 0 to 1 {
        %c0_5 = arith.constant 0.5 : f64
        %c0_7 = arith.constant 0.7 : f64
        %c1 = arith.constant 1 : index
        
        // for(t = 0; t < tmax; t++)
        affine.for %t = 0 to %tmax {
            // for (j = 0; j < ny; j++)
            //   ey[0][j] = _fict_[t];
            affine.for %j = 0 to %ny {
                %fict_t = affine.load %fict[%t] : memref<?xf64>
                affine.store %fict_t, %ey[0, %j] : memref<?x?xf64>
            }
            
            // for (i = 1; i < nx; i++)
            //   for (j = 0; j < ny; j++)
            //     ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
            affine.for %i = 1 to %nx {
                affine.for %j = 0 to %ny {
                    %ey_ij = affine.load %ey[%i, %j] : memref<?x?xf64>
                    %hz_ij = affine.load %hz[%i, %j] : memref<?x?xf64>
                    %hz_i_minus_1_j = affine.load %hz[%i - 1, %j] : memref<?x?xf64>
                    %diff = arith.subf %hz_ij, %hz_i_minus_1_j : f64
                    %scaled_diff = arith.mulf %c0_5, %diff : f64
                    %new_ey_ij = arith.subf %ey_ij, %scaled_diff : f64
                    affine.store %new_ey_ij, %ey[%i, %j] : memref<?x?xf64>
                }
            }
            
            // for (i = 0; i < nx; i++)
            //   for (j = 1; j < ny; j++)
            //     ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
            affine.for %i = 0 to %nx {
                affine.for %j = 1 to %ny {
                    %ex_ij = affine.load %ex[%i, %j] : memref<?x?xf64>
                    %hz_ij = affine.load %hz[%i, %j] : memref<?x?xf64>
                    %hz_i_j_minus_1 = affine.load %hz[%i, %j - 1] : memref<?x?xf64>
                    %diff = arith.subf %hz_ij, %hz_i_j_minus_1 : f64
                    %scaled_diff = arith.mulf %c0_5, %diff : f64
                    %new_ex_ij = arith.subf %ex_ij, %scaled_diff : f64
                    affine.store %new_ex_ij, %ex[%i, %j] : memref<?x?xf64>
                }
            }
            
            // for (i = 0; i < nx - 1; i++)
            //   for (j = 0; j < ny - 1; j++)
            //     hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
            affine.for %i = 0 to affine_map<(d0) -> (d0 - 1)> (%nx) {
                affine.for %j = 0 to affine_map<(d0) -> (d0 - 1)> (%ny) {
                    %hz_ij = affine.load %hz[%i, %j] : memref<?x?xf64>
                    %ex_i_j_plus_1 = affine.load %ex[%i, %j + 1] : memref<?x?xf64>
                    %ex_ij = affine.load %ex[%i, %j] : memref<?x?xf64>
                    %ey_i_plus_1_j = affine.load %ey[%i + 1, %j] : memref<?x?xf64>
                    %ey_ij = affine.load %ey[%i, %j] : memref<?x?xf64>
                    
                    %ex_diff = arith.subf %ex_i_j_plus_1, %ex_ij : f64
                    %ey_diff = arith.subf %ey_i_plus_1_j, %ey_ij : f64
                    %total_diff = arith.addf %ex_diff, %ey_diff : f64
                    %scaled_total = arith.mulf %c0_7, %total_diff : f64
                    %new_hz_ij = arith.subf %hz_ij, %scaled_total : f64
                    affine.store %new_hz_ij, %hz[%i, %j] : memref<?x?xf64>
                    }
                }
            } 
        }
        return
    }
}
