module {
    func.func @kernel_correlation(%data: memref<?x?xf64>, %corr: memref<?x?xf64>, %mean: memref<?xf64>, %stddev: memref<?xf64>, %m: index, %n: index, %float_n: f64) {
        affine.for %loop_once = 0 to 1 {
            %c0 = arith.constant 0.0 : f64
            %c1 = arith.constant 1.0 : f64
            %eps = arith.constant 0.1 : f64
            %c1_index = arith.constant 1 : index
            
            // Step 1: Calculate mean for each column
            // for (j = 0; j < m; j++)
            affine.for %j = 0 to %m {
                // mean[j] = 0.0;
                affine.store %c0, %mean[%j] : memref<?xf64>
                
                // for (i = 0; i < n; i++)
                affine.for %i = 0 to %n {
                    // mean[j] += data[i][j];
                    %mean_j = affine.load %mean[%j] : memref<?xf64>
                    %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
                    %new_mean_j = arith.addf %mean_j, %data_ij : f64
                    affine.store %new_mean_j, %mean[%j] : memref<?xf64>
                }
                
                // mean[j] /= float_n;
                %final_mean_j = affine.load %mean[%j] : memref<?xf64>
                %mean_normalized = arith.divf %final_mean_j, %float_n : f64
                affine.store %mean_normalized, %mean[%j] : memref<?xf64>
            }
            
            // Step 2: Calculate standard deviation for each column
            // for (j = 0; j < m; j++)
            affine.for %j = 0 to %m {
                // stddev[j] = 0.0;
                affine.store %c0, %stddev[%j] : memref<?xf64>
                
                // for (i = 0; i < n; i++)
                affine.for %i = 0 to %n {
                    // stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
                    %stddev_j = affine.load %stddev[%j] : memref<?xf64>
                    %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
                    %mean_j = affine.load %mean[%j] : memref<?xf64>
                    %diff = arith.subf %data_ij, %mean_j : f64
                    %diff_squared = arith.mulf %diff, %diff : f64
                    %new_stddev_j = arith.addf %stddev_j, %diff_squared : f64
                    affine.store %new_stddev_j, %stddev[%j] : memref<?xf64>
                }
                
                // stddev[j] /= float_n;
                %variance = affine.load %stddev[%j] : memref<?xf64>
                %variance_normalized = arith.divf %variance, %float_n : f64
                
                // stddev[j] = sqrt(stddev[j]);
                %stddev_val = math.sqrt %variance_normalized : f64
                
                // stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
                %is_small = arith.cmpf ole, %stddev_val, %eps : f64
                %final_stddev = arith.select %is_small, %c1, %stddev_val : f64
                affine.store %final_stddev, %stddev[%j] : memref<?xf64>
            }
            
            // Step 3: Center and reduce the column vectors
            // Calculate sqrt(float_n) once
            %sqrt_float_n = math.sqrt %float_n : f64
            
            // for (i = 0; i < n; i++)
            affine.for %i = 0 to %n {
                // for (j = 0; j < m; j++)
                affine.for %j = 0 to %m {
                    %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
                    %mean_j = affine.load %mean[%j] : memref<?xf64>
                    %stddev_j = affine.load %stddev[%j] : memref<?xf64>
                    
                    // data[i][j] -= mean[j];
                    %centered = arith.subf %data_ij, %mean_j : f64
                    
                    // data[i][j] /= sqrt(float_n) * stddev[j];
                    %denominator = arith.mulf %sqrt_float_n, %stddev_j : f64
                    %normalized = arith.divf %centered, %denominator : f64
                    affine.store %normalized, %data[%i, %j] : memref<?x?xf64>
                }
            }
            
            // Step 4: Calculate the m x m correlation matrix
            %m_minus_1 = arith.subi %m, %c1_index : index
            
            // for (i = 0; i < m-1; i++)
            affine.for %i = 0 to %m_minus_1 {
                // corr[i][i] = 1.0;
                affine.store %c1, %corr[%i, %i] : memref<?x?xf64>
                
                // for (j = i+1; j < m; j++) - using affine map for dependent loop
                affine.for %j = affine_map<(d0) -> (d0 + 1)> (%i) to %m {
                    // corr[i][j] = 0.0;
                    affine.store %c0, %corr[%i, %j] : memref<?x?xf64>
                    
                    // for (k = 0; k < n; k++)
                    affine.for %k = 0 to %n {
                        // corr[i][j] += (data[k][i] * data[k][j]);
                        %corr_ij = affine.load %corr[%i, %j] : memref<?x?xf64>
                        %data_ki = affine.load %data[%k, %i] : memref<?x?xf64>
                        %data_kj = affine.load %data[%k, %j] : memref<?x?xf64>
                        %product = arith.mulf %data_ki, %data_kj : f64
                        %new_corr_ij = arith.addf %corr_ij, %product : f64
                        affine.store %new_corr_ij, %corr[%i, %j] : memref<?x?xf64>
                    }
                    
                    // corr[j][i] = corr[i][j];
                    %corr_ij_final = affine.load %corr[%i, %j] : memref<?x?xf64>
                    affine.store %corr_ij_final, %corr[%j, %i] : memref<?x?xf64>
                }
            }
            // corr[m-1][m-1] = 1.0;
            affine.store %c1, %corr[%m_minus_1, %m_minus_1] : memref<?x?xf64>
        }
        return
    }
}
