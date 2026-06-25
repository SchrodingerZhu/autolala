module
attributes {
  "simulation.prologue" = "volatile double ARRAY_0[N]; volatile double ARRAY_1[N][N]; volatile double ARRAY_2[N]; double volatile ARRAY_3[N]; double volatile ARRAY_4[N];"
} {
  func.func @kernel_mvt(
    %x1: memref<?xf64>,
    %x2: memref<?xf64>,
    %y_1: memref<?xf64>,
    %y_2: memref<?xf64>,
    %A: memref<?x?xf64>,
    %N: index
  ) {
    affine.for %loop_once = 0 to 1 {
      // First loop nest (tiled): x1[i] = x1[i] + A[i][j] * y_1[j]
      // Tile both i and j by 32 so each 32x32 block of A is loaded once
      // and fully reused; y_1[jj..jj+32] stays cache-resident across the
      // i-tile, and x1[ii..ii+32] is reused across the j sweep.
      affine.for %ii = 0 to %N step 32 {
        affine.for %jj = 0 to %N step 32 {
          affine.for %i = 0 to 32 {
            affine.for %j = 0 to 32 {
              %0 = affine.load %x1[%ii + %i] : memref<?xf64>
              %1 = affine.load %A[%ii + %i, %jj + %j] : memref<?x?xf64>
              %2 = affine.load %y_1[%jj + %j] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %0, %3 : f64
              affine.store %4, %x1[%ii + %i] : memref<?xf64>
            }
          }
        }
      }

      // Second loop nest (tiled): x2[i] = x2[i] + A[j][i] * y_2[j]
      // A is accessed column-major (A[j][i]); tiling converts the long
      // column stride into 32x32 blocks so each block of A is loaded once
      // and reused, x2[ii..ii+32] stays resident across the j sweep, and
      // y_2[jj..jj+32] is reused across the i-tile.
      affine.for %ii = 0 to %N step 32 {
        affine.for %jj = 0 to %N step 32 {
          affine.for %i = 0 to 32 {
            affine.for %j = 0 to 32 {
              %5 = affine.load %x2[%ii + %i] : memref<?xf64>
              %6 = affine.load %A[%jj + %j, %ii + %i] : memref<?x?xf64>
              %7 = affine.load %y_2[%jj + %j] : memref<?xf64>
              %8 = arith.mulf %6, %7 : f64
              %9 = arith.addf %5, %8 : f64
              affine.store %9, %x2[%ii + %i] : memref<?xf64>
            }
          }
        }
      }
    } { dmd.extract }

    return
  }
}
