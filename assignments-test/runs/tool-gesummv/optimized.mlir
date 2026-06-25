module {
  func.func @kernel_gesummv(%A: memref<?x?xf64>,
                            %B: memref<?x?xf64>,
                            %tmp: memref<?xf64>,
                            %x: memref<?xf64>,
                            %y: memref<?xf64>, %N : index) {
      // Tile both the i and j loops by 32. The init of tmp[i]/y[i] is hoisted
      // into a separate per-tile loop, and the final alpha/beta combine into a
      // third per-tile loop, so the accumulation can be cleanly tiled in j
      // without affine.if. Within an (ii,jj) tile, the x[jj..jj+32] block is
      // reused across all 32 i rows, and a 32x32 block of A and B stays resident.
      affine.for %ii = 0 to %N step 32 {
        %c0_f64 = arith.constant 0.0 : f64
        %alpha = arith.constant 1.0 : f64
        %beta = arith.constant 1.0 : f64

        // (a) Initialize tmp[i] = 0.0 and y[i] = 0.0 for this i-tile.
        affine.for %i = 0 to 32 {
          affine.store %c0_f64, %tmp[%ii + %i] : memref<?xf64>
          affine.store %c0_f64, %y[%ii + %i] : memref<?xf64>
        }

        // (b) Tiled accumulation: for each j-tile, sweep the 32x32 block.
        affine.for %jj = 0 to %N step 32 {
          affine.for %i = 0 to 32 {
            affine.for %j = 0 to 32 {
              // tmp[i] = A[i][j] * x[j] + tmp[i]
              %0 = affine.load %tmp[%ii + %i] : memref<?xf64>
              %1 = affine.load %A[%ii + %i, %jj + %j] : memref<?x?xf64>
              %2 = affine.load %x[%jj + %j] : memref<?xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %3, %0 : f64
              affine.store %4, %tmp[%ii + %i] : memref<?xf64>

              // y[i] = B[i][j] * x[j] + y[i]
              %5 = affine.load %y[%ii + %i] : memref<?xf64>
              %6 = affine.load %B[%ii + %i, %jj + %j] : memref<?x?xf64>
              %7 = affine.load %x[%jj + %j] : memref<?xf64>
              %8 = arith.mulf %6, %7 : f64
              %9 = arith.addf %8, %5 : f64
              affine.store %9, %y[%ii + %i] : memref<?xf64>
            }
          }
        }

        // (c) Final: y[i] = alpha * tmp[i] + beta * y[i] for this i-tile.
        affine.for %i = 0 to 32 {
          %10 = affine.load %tmp[%ii + %i] : memref<?xf64>
          %11 = affine.load %y[%ii + %i] : memref<?xf64>
          %12 = arith.mulf %alpha, %10 : f64
          %13 = arith.mulf %beta, %11 : f64
          %14 = arith.addf %12, %13 : f64
          affine.store %14, %y[%ii + %i] : memref<?xf64>
        }
      } {dmd.extract}
    return
  }
}
