module 
attributes { "simulation.prologue" = "volatile double ARRAY_0[240]; volatile double ARRAY_1[240][260]; volatile double ARRAY_2[240]; volatile double ARRAY_3[240][240];" }
{
  func.func @kernel_correlation(
    %data: memref<260x240xf64>,
    %corr: memref<240x240xf64>,
    %mean: memref<240xf64>,
    %stddev: memref<240xf64>,
    %float_n: f64
  ) {
    affine.for %dummy = 0 to 1 {
      %c0 = arith.constant 0.0 : f64
      %c1 = arith.constant 1.0 : f64
      %eps = arith.constant 0.1 : f64

      // Step 1: Calculate mean for each column
      affine.for %j = 0 to 240 {
        affine.store %c0, %mean[%j] : memref<240xf64>

        affine.for %i = 0 to 260 {
          %mean_j = affine.load %mean[%j] : memref<240xf64>
          %data_ij = affine.load %data[%i, %j] : memref<260x240xf64>
          %new_mean_j = arith.addf %mean_j, %data_ij : f64
          affine.store %new_mean_j, %mean[%j] : memref<240xf64>
        }

        %final_mean_j = affine.load %mean[%j] : memref<240xf64>
        %mean_normalized = arith.divf %final_mean_j, %float_n : f64
        affine.store %mean_normalized, %mean[%j] : memref<240xf64>
      }

      // Step 2: Calculate stddev for each column
      affine.for %j = 0 to 240 {
        affine.store %c0, %stddev[%j] : memref<240xf64>

        affine.for %i = 0 to 260 {
          %stddev_j = affine.load %stddev[%j] : memref<240xf64>
          %data_ij = affine.load %data[%i, %j] : memref<260x240xf64>
          %mean_j = affine.load %mean[%j] : memref<240xf64>
          %diff = arith.subf %data_ij, %mean_j : f64
          %diff_squared = arith.mulf %diff, %diff : f64
          %new_stddev_j = arith.addf %stddev_j, %diff_squared : f64
          affine.store %new_stddev_j, %stddev[%j] : memref<240xf64>
        }

        %variance = affine.load %stddev[%j] : memref<240xf64>
        %variance_normalized = arith.divf %variance, %float_n : f64
        %stddev_val = math.sqrt %variance_normalized : f64
        %is_small = arith.cmpf ole, %stddev_val, %eps : f64
        %final_stddev = arith.select %is_small, %c1, %stddev_val : f64
        affine.store %final_stddev, %stddev[%j] : memref<240xf64>
      }

      // Step 3: Center and reduce the data
      %sqrt_float_n = math.sqrt %float_n : f64
      affine.for %i = 0 to 260 {
        affine.for %j = 0 to 240 {
          %data_ij = affine.load %data[%i, %j] : memref<260x240xf64>
          %mean_j = affine.load %mean[%j] : memref<240xf64>
          %stddev_j = affine.load %stddev[%j] : memref<240xf64>
          %centered = arith.subf %data_ij, %mean_j : f64
          %denominator = arith.mulf %sqrt_float_n, %stddev_j : f64
          %normalized = arith.divf %centered, %denominator : f64
          affine.store %normalized, %data[%i, %j] : memref<260x240xf64>
        }
      }

      // Step 4: Compute correlation matrix
      affine.for %i = 0 to 239 {
        affine.store %c1, %corr[%i, %i] : memref<240x240xf64>
        affine.for %j = affine_map<(d0) -> (d0 + 1)>(%i) to 240 {
          affine.store %c0, %corr[%i, %j] : memref<240x240xf64>
          affine.for %k = 0 to 260 {
            %corr_ij = affine.load %corr[%i, %j] : memref<240x240xf64>
            %data_ki = affine.load %data[%k, %i] : memref<260x240xf64>
            %data_kj = affine.load %data[%k, %j] : memref<260x240xf64>
            %product = arith.mulf %data_ki, %data_kj : f64
            %new_corr_ij = arith.addf %corr_ij, %product : f64
            affine.store %new_corr_ij, %corr[%i, %j] : memref<240x240xf64>
          }
          %corr_ij_final = affine.load %corr[%i, %j] : memref<240x240xf64>
          affine.store %corr_ij_final, %corr[%j, %i] : memref<240x240xf64>
        }
      }

      affine.store %c1, %corr[239, 239] : memref<240x240xf64>
    }

    return
  }
}
