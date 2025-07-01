module {
  func.func @kernel_correlation(
    %data: memref<?x?xf64>,
    %corr: memref<?x?xf64>,
    %mean: memref<?xf64>,
    %stddev: memref<?xf64>,
    %m: index, %n: index, %float_n: f64) {
    
    affine.for %loop_once = 0 to 1 {
      %c0 = arith.constant 0.0 : f64
      %c1 = arith.constant 1.0 : f64
      %eps = arith.constant 0.1 : f64
      %c1_index = arith.constant 1 : index

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

      // Step 2: stddev
      affine.for %j = 0 to %m {
        affine.store %c0, %stddev[%j] : memref<?xf64>
        affine.for %i = 0 to %n {
          %stddev_j = affine.load %stddev[%j] : memref<?xf64>
          %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
          %mean_j = affine.load %mean[%j] : memref<?xf64>
          %diff = arith.subf %data_ij, %mean_j : f64
          %diff_squared = arith.mulf %diff, %diff : f64
          %new_stddev_j = arith.addf %stddev_j, %diff_squared : f64
          affine.store %new_stddev_j, %stddev[%j] : memref<?xf64>
        }
        %variance = affine.load %stddev[%j] : memref<?xf64>
        %variance_normalized = arith.divf %variance, %float_n : f64
        %stddev_val = math.sqrt %variance_normalized : f64
        %is_small = arith.cmpf ole, %stddev_val, %eps : f64
        %final_stddev = arith.select %is_small, %c1, %stddev_val : f64
        affine.store %final_stddev, %stddev[%j] : memref<?xf64>
      }

      // Step 3: center + reduce
      %sqrt_float_n = math.sqrt %float_n : f64
      affine.for %i = 0 to %n {
        affine.for %j = 0 to %m {
          %data_ij = affine.load %data[%i, %j] : memref<?x?xf64>
          %mean_j = affine.load %mean[%j] : memref<?xf64>
          %stddev_j = affine.load %stddev[%j] : memref<?xf64>
          %centered = arith.subf %data_ij, %mean_j : f64
          %denominator = arith.mulf %sqrt_float_n, %stddev_j : f64
          %normalized = arith.divf %centered, %denominator : f64
          affine.store %normalized, %data[%i, %j] : memref<?x?xf64>
        }
      }

      // Step 4: correlation matrix
      affine.for %i = 0 to affine_map<()[s0] -> (s0 - 1)>()[%m] {
        affine.store %c1, %corr[%i, %i] : memref<?x?xf64>

        // for (j = i+1; j < m; j++)
        affine.for %j = affine_map<(d0) -> (d0 + 1)>(%i) to %m {
          affine.store %c0, %corr[%i, %j] : memref<?x?xf64>
          affine.for %k = 0 to %n {
            %corr_ij = affine.load %corr[%i, %j] : memref<?x?xf64>
            %data_ki = affine.load %data[%k, %i] : memref<?x?xf64>
            %data_kj = affine.load %data[%k, %j] : memref<?x?xf64>
            %product = arith.mulf %data_ki, %data_kj : f64
            %new_corr_ij = arith.addf %corr_ij, %product : f64
            affine.store %new_corr_ij, %corr[%i, %j] : memref<?x?xf64>
          }
          %corr_ij_final = affine.load %corr[%i, %j] : memref<?x?xf64>
          affine.store %corr_ij_final, %corr[%j, %i] : memref<?x?xf64>
        }
      }
      affine.store %c1, %corr[%m - 1, %m - 1] : memref<?x?xf64>
    }

    return
  }
}
