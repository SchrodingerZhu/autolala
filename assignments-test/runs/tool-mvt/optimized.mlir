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
    // Fused + tiled mvt.
    //   Nest 1:  x1[i] += A[i][j] * y_1[j]        (sum over j)
    //   Nest 2:  x2[i] += A[j][i] * y_2[j]        (sum over j)
    // Rename nest 2 loops (i<->j) -> x2[j] += A[i][j] * y_2[i] (sum over i).
    // Both now traverse A as A[i][j], so the matrix is streamed ONCE instead
    // of twice (second pass previously read A column-major / transposed).
    // 32x32 register/cache tiling bounds the reuse distance of the x2/y vectors.
    affine.for %ii = 0 to %N step 32 {
      affine.for %jj = 0 to %N step 32 {
        affine.for %i = 0 to 32 {
          affine.for %j = 0 to 32 {
            // x1[i] += A[i][j] * y_1[j]
            %0 = affine.load %x1[%ii + %i] : memref<?xf64>
            %1 = affine.load %A[%ii + %i, %jj + %j] : memref<?x?xf64>
            %2 = affine.load %y_1[%jj + %j] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %0, %3 : f64
            affine.store %4, %x1[%ii + %i] : memref<?xf64>

            // x2[j] += A[i][j] * y_2[i]   (== original x2[i] += A[j][i] * y_2[j])
            %5 = affine.load %x2[%jj + %j] : memref<?xf64>
            %6 = affine.load %A[%ii + %i, %jj + %j] : memref<?x?xf64>
            %7 = affine.load %y_2[%ii + %i] : memref<?xf64>
            %8 = arith.mulf %6, %7 : f64
            %9 = arith.addf %5, %8 : f64
            affine.store %9, %x2[%jj + %j] : memref<?xf64>
          }
        }
      }
    } { dmd.extract }

    return
  }
}
