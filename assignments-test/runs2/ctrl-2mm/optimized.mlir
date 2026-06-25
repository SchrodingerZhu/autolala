module {
    func.func @kernel_2mm(%tmp: memref<?x?xf64>, %A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %D: memref<?x?xf64>, %ni: index, %nj: index, %nk: index, %nl: index) {
        affine.for %loop_once = 0 to 1 {

            %alpha = arith.constant 1.0 : f64
            %beta = arith.constant 1.0 : f64
            %c0 = arith.constant 0.0 : f64

            // First matrix multiplication: tmp = alpha * A * B
            // Tiled i, j, k with tile size 32.
            affine.for %ii = 0 to %ni step 32 {
                affine.for %jj = 0 to %nj step 32 {
                    // Initialize tmp[i][j] = 0 for this (ii,jj) tile.
                    affine.for %i = 0 to 32 {
                        affine.for %j = 0 to 32 {
                            affine.store %c0, %tmp[%ii + %i, %jj + %j] : memref<?x?xf64>
                        }
                    }
                    affine.for %kk = 0 to %nk step 32 {
                        affine.for %i = 0 to 32 {
                            affine.for %j = 0 to 32 {
                                affine.for %k = 0 to 32 {
                                    %tmp_ij = affine.load %tmp[%ii + %i, %jj + %j] : memref<?x?xf64>
                                    %A_ik = affine.load %A[%ii + %i, %kk + %k] : memref<?x?xf64>
                                    %B_kj = affine.load %B[%kk + %k, %jj + %j] : memref<?x?xf64>
                                    %prod1 = arith.mulf %alpha, %A_ik : f64
                                    %prod2 = arith.mulf %prod1, %B_kj : f64
                                    %new_tmp_ij = arith.addf %tmp_ij, %prod2 : f64
                                    affine.store %new_tmp_ij, %tmp[%ii + %i, %jj + %j] : memref<?x?xf64>
                                }
                            }
                        }
                    }
                }
            }

            // Second matrix multiplication: D = tmp * C + beta * D
            // Tiled i, j, k with tile size 32.
            affine.for %ii = 0 to %ni step 32 {
                affine.for %jj = 0 to %nl step 32 {
                    // D[i][j] *= beta for this (ii,jj) tile.
                    affine.for %i = 0 to 32 {
                        affine.for %j = 0 to 32 {
                            %D_ij = affine.load %D[%ii + %i, %jj + %j] : memref<?x?xf64>
                            %scaled_D_ij = arith.mulf %D_ij, %beta : f64
                            affine.store %scaled_D_ij, %D[%ii + %i, %jj + %j] : memref<?x?xf64>
                        }
                    }
                    affine.for %kk = 0 to %nj step 32 {
                        affine.for %i = 0 to 32 {
                            affine.for %j = 0 to 32 {
                                affine.for %k = 0 to 32 {
                                    %D_ij_current = affine.load %D[%ii + %i, %jj + %j] : memref<?x?xf64>
                                    %tmp_ik = affine.load %tmp[%ii + %i, %kk + %k] : memref<?x?xf64>
                                    %C_kj = affine.load %C[%kk + %k, %jj + %j] : memref<?x?xf64>
                                    %prod = arith.mulf %tmp_ik, %C_kj : f64
                                    %new_D_ij = arith.addf %D_ij_current, %prod : f64
                                    affine.store %new_D_ij, %D[%ii + %i, %jj + %j] : memref<?x?xf64>
                                }
                            }
                        }
                    }
                }
            }
        } {dmd.extract}
        return
    }
}
