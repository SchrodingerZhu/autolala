module {
    func.func @kernel_covariance(%data: memref<?x?xf64>, %cov: memref<?x?xf64>, %mean: memref<?xf64>, %m: index, %n: index, %float_n: f64) {
        affine.for %loop_once = 0 to 1 {

            %c0 = arith.constant 0.0 : f64
            %c1 = arith.constant 1.0 : f64

            // Step 1a: Initialize mean[j] = 0.0 for all columns.
            affine.for %j = 0 to %m {
                affine.store %c0, %mean[%j] : memref<?xf64>
            }

            // Step 1b: Accumulate mean by streaming data row-major (i outer, j inner).
            // for (i = 0; i < n; i++)
            //   for (j = 0; j < m; j++)  mean[j] += data[i][j];
            // Interchange of the original (j-outer, i-inner) loop: data[i][j] is now
            // accessed with unit stride along j, restoring spatial locality.
            affine.for %i = 0 to %n {
                affine.for %j = 0 to %m {
                    %mean_j = affine.load %mean[%j] : memref<?xf64>
                    %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
                    %new_mean_j = arith.addf %mean_j, %data_ij : f64
                    affine.store %new_mean_j, %mean[%j] : memref<?xf64>
                }
            }

            // Step 1c: Normalize mean[j] /= float_n for all columns.
            affine.for %j = 0 to %m {
                %final_mean_j = affine.load %mean[%j] : memref<?xf64>
                %mean_normalized = arith.divf %final_mean_j, %float_n : f64
                affine.store %mean_normalized, %mean[%j] : memref<?xf64>
            }

            // Step 2: Subtract mean from each data point (already row-major).
            affine.for %i = 0 to %n {
                affine.for %j = 0 to %m {
                    %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
                    %mean_j = affine.load %mean[%j] : memref<?xf64>
                    %centered = arith.subf %data_ij, %mean_j : f64
                    affine.store %centered, %data[%i, %j] : memref<?x?xf64>
                }
            }

            // Step 3: Covariance matrix with the k (reduction) loop tiled by 32.
            %float_n_minus_1 = arith.subf %float_n, %c1 : f64

            // for (i = 0; i < m; i++)
            affine.for %i = 0 to %m {
                // for (j = i; j < m; j++)
                affine.for %j = affine_map<(d0) -> (d0)> (%i) to %m {
                    // cov[i][j] = 0.0;
                    affine.store %c0, %cov[%i, %j] : memref<?x?xf64>

                    // Tiled reduction over k: blocks of 32 keep the partial sum and the
                    // loaded data[k][i] / data[k][j] column block resident, shortening the
                    // reuse distance of the cov[i][j] accumulator.
                    affine.for %kk = 0 to %n step 32 {
                        affine.for %k = 0 to 32 {
                            %cov_ij = affine.load %cov[%i, %j] : memref<?x?xf64>
                            %data_ki = affine.load %data[%kk + %k, %i] : memref<?x?xf64>
                            %data_kj = affine.load %data[%kk + %k, %j] : memref<?x?xf64>
                            %product = arith.mulf %data_ki, %data_kj : f64
                            %new_cov_ij = arith.addf %cov_ij, %product : f64
                            affine.store %new_cov_ij, %cov[%i, %j] : memref<?x?xf64>
                        }
                    }

                    // cov[i][j] /= (float_n - 1.0);
                    %cov_sum = affine.load %cov[%i, %j] : memref<?x?xf64>
                    %cov_normalized = arith.divf %cov_sum, %float_n_minus_1 : f64
                    affine.store %cov_normalized, %cov[%i, %j] : memref<?x?xf64>

                    // cov[j][i] = cov[i][j];
                    %cov_ij_final = affine.load %cov[%i, %j] : memref<?x?xf64>
                    affine.store %cov_ij_final, %cov[%j, %i] : memref<?x?xf64>
                }
            }
        } {dmd.extract}
        return
    }
}
