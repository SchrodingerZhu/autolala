module {
  func.func @kernel_gesummv(%A: memref<?x?xf64>,
                            %B: memref<?x?xf64>,
                            %tmp: memref<?xf64>,
                            %x: memref<?xf64>,
                            %y: memref<?xf64>, %N : index) {
      affine.for %loop_once = 0 to 1 {
      %c0_f64 = arith.constant 0.0 : f64
      %alpha = arith.constant 1.0 : f64
      %beta = arith.constant 1.0 : f64

      // Phase 1: initialize tmp[i] = 0.0 and y[i] = 0.0 for all i
      affine.for %i = 0 to %N {
        affine.store %c0_f64, %tmp[%i] : memref<?xf64>
        affine.store %c0_f64, %y[%i] : memref<?xf64>
      }

      // Phase 2: tiled accumulation.
      // Tile the j (reduction) loop so a block of x[j] (and the touched
      // A[i,j]/B[i,j] columns) is reused across the whole i sweep within
      // each j-tile.  Loop order: jj (tile) -> i -> j.
      affine.for %jj = 0 to %N step 32 {
        affine.for %i = 0 to %N {
          affine.for %j = 0 to 32 {
            // tmp[i] += A[i][j] * x[j]
            %0 = affine.load %tmp[%i] : memref<?xf64>
            %1 = affine.load %A[%i, %jj + %j] : memref<?x?xf64>
            %2 = affine.load %x[%jj + %j] : memref<?xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %3, %0 : f64
            affine.store %4, %tmp[%i] : memref<?xf64>

            // y[i] += B[i][j] * x[j]
            %5 = affine.load %y[%i] : memref<?xf64>
            %6 = affine.load %B[%i, %jj + %j] : memref<?x?xf64>
            %7 = affine.load %x[%jj + %j] : memref<?xf64>
            %8 = arith.mulf %6, %7 : f64
            %9 = arith.addf %8, %5 : f64
            affine.store %9, %y[%i] : memref<?xf64>
          }
        }
      }

      // Phase 3: finalize y[i] = alpha * tmp[i] + beta * y[i] for all i
      affine.for %i = 0 to %N {
        %10 = affine.load %tmp[%i] : memref<?xf64>
        %11 = affine.load %y[%i] : memref<?xf64>
        %12 = arith.mulf %alpha, %10 : f64
        %13 = arith.mulf %beta, %11 : f64
        %14 = arith.addf %12, %13 : f64
        affine.store %14, %y[%i] : memref<?xf64>
      }
     } {dmd.extract}
    return
  }
}
