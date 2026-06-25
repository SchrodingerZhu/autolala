module {
    func.func @kernel_covariance(%data: memref<?x?xf64>, %cov: memref<?x?xf64>, %mean: memref<?xf64>, %m: index, %n: index, %float_n: f64) {
        affine.for %loop_once = 0 to 1 {
            %c0 = arith.constant 0.0 : f64
            %c1 = arith.constant 1.0 : f64

            // Step 1: mean
            affine.for %j = 0 to %m {
                affine.store %c0, %mean[%j] : memref<?xf64>
                affine.for %i = 0 to %n {
                    %mean_j = affine.load %mean[%j] : memref<?xf64>
                    %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
                    %new_mean_j = arith.addf %mean_j, %data_ij : f64
                    affine.store %new_mean_j, %mean[%j] : memref<?xf64>
                }
                %final_mean_j = affine.load %mean[%j] : memref<?xf64>
                %mean_normalized = arith.divf %final_mean_j, %float_n : f64
                affine.store %mean_normalized, %mean[%j] : memref<?xf64>
            }

            // Step 2: center
            affine.for %i = 0 to %n {
                affine.for %j = 0 to %m {
                    %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
                    %mean_j = affine.load %mean[%j] : memref<?xf64>
                    %centered = arith.subf %data_ij, %mean_j : f64
                    affine.store %centered, %data[%i, %j] : memref<?x?xf64>
                }
            }

            // Step 3: covariance.
            // Restructure: init+finalize separated from accumulation so that k-tiling is legal.
            %float_n_minus_1 = arith.subf %float_n, %c1 : f64

            // 3a: zero cov[i][j] over the triangle
            affine.for %i = 0 to %m {
                affine.for %j = affine_map<(d0) -> (d0)> (%i) to %m {
                    affine.store %c0, %cov[%i, %j] : memref<?x?xf64>
                }
            }

            // 3b: accumulate with i and k tiled (32x32 data block reused across all j)
            affine.for %kk = 0 to %n step 32 {
                affine.for %ii = 0 to %m step 32 {
                    affine.for %i = 0 to 32 {
                        affine.for %j = affine_map<(d0) -> (d0)> (%i + %ii) to %m {
                            affine.for %k = 0 to 32 {
                                %cov_ij = affine.load %cov[%i + %ii, %j] : memref<?x?xf64>
                                %data_ki = affine.load %data[%k + %kk, %i + %ii] : memref<?x?xf64>
                                %data_kj = affine.load %data[%k + %kk, %j] : memref<?x?xf64>
                                %product = arith.mulf %data_ki, %data_kj : f64
                                %new_cov_ij = arith.addf %cov_ij, %product : f64
                                affine.store %new_cov_ij, %cov[%i + %ii, %j] : memref<?x?xf64>
                            }
                        }
                    }
                }
            }

            // 3c: normalize and mirror
            affine.for %i = 0 to %m {
                affine.for %j = affine_map<(d0) -> (d0)> (%i) to %m {
                    %cov_sum = affine.load %cov[%i, %j] : memref<?x?xf64>
                    %cov_normalized = arith.divf %cov_sum, %float_n_minus_1 : f64
                    affine.store %cov_normalized, %cov[%i, %j] : memref<?x?xf64>
                    %cov_ij_final = affine.load %cov[%i, %j] : memref<?x?xf64>
                    affine.store %cov_ij_final, %cov[%j, %i] : memref<?x?xf64>
                }
            }
        } {dmd.extract}
        return
    }
}
