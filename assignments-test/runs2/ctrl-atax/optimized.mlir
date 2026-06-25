module {
    func.func @kernel_atax(%A: memref<?x?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %tmp: memref<?xf64>, %m: index, %n: index) {
        affine.for %loop_once = 0 to 1 {
            %c0 = arith.constant 0.0 : f64

            // for (i = 0; i < n; i++)
            //   y[i] = 0;
            affine.for %i = 0 to %n {
                affine.store %c0, %y[%i] : memref<?xf64>
            }

            // for (i = 0; i < m; i++)
            affine.for %i = 0 to %m {
                // tmp[i] = 0.0;
                affine.store %c0, %tmp[%i] : memref<?xf64>

                // Phase 1 (tiled over j): tmp[i] += A[i][j] * x[j]
                // j-tiling bounds the x-block / A-row-block working set so the
                // freshly loaded A[i][jj..jj+32] block stays cache-resident and
                // is reused by Phase 2 within the same i iteration.
                affine.for %jj = 0 to %n step 32 {
                    affine.for %j = 0 to 32 {
                        %tmp_i = affine.load %tmp[%i] : memref<?xf64>
                        %A_ij = affine.load %A[%i, %jj + %j] : memref<?x?xf64>
                        %x_j = affine.load %x[%jj + %j] : memref<?xf64>
                        %prod = arith.mulf %A_ij, %x_j : f64
                        %new_tmp_i = arith.addf %tmp_i, %prod : f64
                        affine.store %new_tmp_i, %tmp[%i] : memref<?xf64>
                    }
                }

                // Phase 2 (tiled over j): y[j] += A[i][j] * tmp[i]
                // Uses the fully-accumulated tmp[i]; the same A[i][jj..jj+32]
                // block touched in Phase 1 is reused here, and y-blocks have a
                // bounded working set.
                affine.for %jj = 0 to %n step 32 {
                    affine.for %j = 0 to 32 {
                        %y_j = affine.load %y[%jj + %j] : memref<?xf64>
                        %A_ij = affine.load %A[%i, %jj + %j] : memref<?x?xf64>
                        %tmp_i = affine.load %tmp[%i] : memref<?xf64>
                        %prod = arith.mulf %A_ij, %tmp_i : f64
                        %new_y_j = arith.addf %y_j, %prod : f64
                        affine.store %new_y_j, %y[%jj + %j] : memref<?xf64>
                    }
                }
            }
        } {dmd.extract}
        return
    }
}
