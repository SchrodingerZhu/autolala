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
      // First loop nest: x1[i] = x1[i] + A[i][j] * y_1[j]
      affine.for %i = 0 to %N {
        affine.for %j = 0 to %N {
          %0 = affine.load %x1[%i] : memref<?xf64>
          %1 = affine.load %A[%i, %j] : memref<?x?xf64>
          %2 = affine.load %y_1[%j] : memref<?xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.addf %0, %3 : f64
          affine.store %4, %x1[%i] : memref<?xf64>
        }
      }

      // Second loop nest: x2[i] = x2[i] + A[j][i] * y_2[j]
      affine.for %i = 0 to %N {
        affine.for %j = 0 to %N {
          %5 = affine.load %x2[%i] : memref<?xf64>
          %6 = affine.load %A[%j, %i] : memref<?x?xf64>
          %7 = affine.load %y_2[%j] : memref<?xf64>
          %8 = arith.mulf %6, %7 : f64
          %9 = arith.addf %5, %8 : f64
          affine.store %9, %x2[%i] : memref<?xf64>
        }
      }
    } { slap.extract }

    return
  }
}
