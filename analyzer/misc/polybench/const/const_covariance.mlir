module 
attributes {"simulation.prologue" = "volatile double ARRAY_0[240]; volatile double ARRAY_1[260][240]; volatile double ARRAY_2[260][240];"}
{
    func.func @kernel_covariance(%data: memref<260x240xf64>, %cov: memref<240x240xf64>, %mean: memref<240xf64>, %float_n: f64) {
        %c0 = arith.constant 0.0 : f64
        %c1 = arith.constant 1.0 : f64
        %c240 = arith.constant 240 : index
        %c260 = arith.constant 260 : index
        affine.for %loop_once = 0 to 1 {
        // Step 1: Calculate mean for each column
        // for (j = 0; j < 240; j++)
        affine.for %j = 0 to 240 {
            // mean[j] = 0.0;
            affine.store %c0, %mean[%j] : memref<240xf64>
            
            // for (i = 0; i < 260; i++)
            affine.for %i = 0 to 260 {
                // mean[j] += data[i][j];
                %mean_j = affine.load %mean[%j] : memref<240xf64>
                %data_ij = affine.load %data[%i, %j] : memref<260x240xf64>
                %new_mean_j = arith.addf %mean_j, %data_ij : f64
                affine.store %new_mean_j, %mean[%j] : memref<240xf64>
            }
            
            // mean[j] /= float_n;
            %final_mean_j = affine.load %mean[%j] : memref<240xf64>
            %mean_normalized = arith.divf %final_mean_j, %float_n : f64
            affine.store %mean_normalized, %mean[%j] : memref<240xf64>
        }
        
        // Step 2: Subtract mean from each data point
        // for (i = 0; i < 260; i++)
        affine.for %i = 0 to 260 {
            // for (j = 0; j < 240; j++)
            affine.for %j = 0 to 240 {
                // data[i][j] -= mean[j];
                %data_ij = affine.load %data[%i, %j] : memref<260x240xf64>
                %mean_j = affine.load %mean[%j] : memref<240xf64>
                %centered = arith.subf %data_ij, %mean_j : f64
                affine.store %centered, %data[%i, %j] : memref<260x240xf64>
            }
        }
        
        // Step 3: Calculate covariance matrix
        // Calculate (float_n - 1.0) once
        %float_n_minus_1 = arith.subf %float_n, %c1 : f64
        
        // for (i = 0; i < 240; i++)
        affine.for %i = 0 to 240 {
            // for (j = i; j < 240; j++) - using affine map for dependent loop
            affine.for %j = affine_map<(d0) -> (d0)> (%i) to 240 {
                // cov[i][j] = 0.0;
                affine.store %c0, %cov[%i, %j] : memref<240x240xf64>
                
                // for (k = 0; k < 260; k++)
                affine.for %k = 0 to 260 {
                    // cov[i][j] += data[k][i] * data[k][j];
                    %cov_ij = affine.load %cov[%i, %j] : memref<240x240xf64>
                    %data_ki = affine.load %data[%k, %i] : memref<260x240xf64>
                    %data_kj = affine.load %data[%k, %j] : memref<260x240xf64>
                    %product = arith.mulf %data_ki, %data_kj : f64
                    %new_cov_ij = arith.addf %cov_ij, %product : f64
                    affine.store %new_cov_ij, %cov[%i, %j] : memref<240x240xf64>
                }
                
                // cov[i][j] /= (float_n - 1.0);
                %cov_sum = affine.load %cov[%i, %j] : memref<240x240xf64>
                %cov_normalized = arith.divf %cov_sum, %float_n_minus_1 : f64
                affine.store %cov_normalized, %cov[%i, %j] : memref<240x240xf64>
                
                // cov[j][i] = cov[i][j];
                %cov_ij_final = affine.load %cov[%i, %j] : memref<240x240xf64>
                affine.store %cov_ij_final, %cov[%j, %i] : memref<240x240xf64>
            }
        }
        }{ slap.extract }

        return
    }
}
