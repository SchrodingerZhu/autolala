module {
    func.func @kernel_3mm(%E: memref<?x?xf64>, %A: memref<?x?xf64>, %B: memref<?x?xf64>, %F: memref<?x?xf64>, %C: memref<?x?xf64>, %D: memref<?x?xf64>, %G: memref<?x?xf64>, %ni: index, %nj: index, %nk: index, %nl: index, %nm: index) {
        affine.for %loop_once = 0 to 1 {

            %c0 = arith.constant 0.0 : f64

            // ===== E := A*B  (tiled i,j,k with 32x32x32 blocking) =====
            // init E[i][j] = 0
            affine.for %i = 0 to %ni {
                affine.for %j = 0 to %nj {
                    affine.store %c0, %E[%i, %j] : memref<?x?xf64>
                }
            }
            // tiled accumulation E[i][j] += A[i][k]*B[k][j]
            affine.for %ii = 0 to %ni step 32 {
                affine.for %jj = 0 to %nj step 32 {
                    affine.for %kk = 0 to %nk step 32 {
                        affine.for %i = 0 to 32 {
                            affine.for %j = 0 to 32 {
                                affine.for %k = 0 to 32 {
                                    %E_ij = affine.load %E[%ii + %i, %jj + %j] : memref<?x?xf64>
                                    %A_ik = affine.load %A[%ii + %i, %kk + %k] : memref<?x?xf64>
                                    %B_kj = affine.load %B[%kk + %k, %jj + %j] : memref<?x?xf64>
                                    %prod = arith.mulf %A_ik, %B_kj : f64
                                    %new_E_ij = arith.addf %E_ij, %prod : f64
                                    affine.store %new_E_ij, %E[%ii + %i, %jj + %j] : memref<?x?xf64>
                                }
                            }
                        }
                    }
                }
            }

            // ===== F := C*D  (tiled i,j,k with 32x32x32 blocking) =====
            // init F[i][j] = 0
            affine.for %i = 0 to %nj {
                affine.for %j = 0 to %nl {
                    affine.store %c0, %F[%i, %j] : memref<?x?xf64>
                }
            }
            // tiled accumulation F[i][j] += C[i][k]*D[k][j]
            affine.for %ii = 0 to %nj step 32 {
                affine.for %jj = 0 to %nl step 32 {
                    affine.for %kk = 0 to %nm step 32 {
                        affine.for %i = 0 to 32 {
                            affine.for %j = 0 to 32 {
                                affine.for %k = 0 to 32 {
                                    %F_ij = affine.load %F[%ii + %i, %jj + %j] : memref<?x?xf64>
                                    %C_ik = affine.load %C[%ii + %i, %kk + %k] : memref<?x?xf64>
                                    %D_kj = affine.load %D[%kk + %k, %jj + %j] : memref<?x?xf64>
                                    %prod = arith.mulf %C_ik, %D_kj : f64
                                    %new_F_ij = arith.addf %F_ij, %prod : f64
                                    affine.store %new_F_ij, %F[%ii + %i, %jj + %j] : memref<?x?xf64>
                                }
                            }
                        }
                    }
                }
            }

            // ===== G := E*F  (tiled i,j,k with 32x32x32 blocking) =====
            // init G[i][j] = 0
            affine.for %i = 0 to %ni {
                affine.for %j = 0 to %nl {
                    affine.store %c0, %G[%i, %j] : memref<?x?xf64>
                }
            }
            // tiled accumulation G[i][j] += E[i][k]*F[k][j]
            affine.for %ii = 0 to %ni step 32 {
                affine.for %jj = 0 to %nl step 32 {
                    affine.for %kk = 0 to %nj step 32 {
                        affine.for %i = 0 to 32 {
                            affine.for %j = 0 to 32 {
                                affine.for %k = 0 to 32 {
                                    %G_ij = affine.load %G[%ii + %i, %jj + %j] : memref<?x?xf64>
                                    %E_ik = affine.load %E[%ii + %i, %kk + %k] : memref<?x?xf64>
                                    %F_kj = affine.load %F[%kk + %k, %jj + %j] : memref<?x?xf64>
                                    %prod = arith.mulf %E_ik, %F_kj : f64
                                    %new_G_ij = arith.addf %G_ij, %prod : f64
                                    affine.store %new_G_ij, %G[%ii + %i, %jj + %j] : memref<?x?xf64>
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
