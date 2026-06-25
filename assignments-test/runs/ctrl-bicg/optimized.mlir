module {
    // Function definition
    func.func @bicg(%argA: memref<?x?xf64>, %argS: memref<?xf64>, %argR: memref<?xf64>, %argQ: memref<?xf64>, %argP: memref<?xf64>, %M : index, %N : index) {
        affine.for %loop_once = 0 to 1 {
            %c0 = arith.constant 0.0 : f64

            // Initialize q[i] = 0 for all i, done exactly once.
            affine.for %i = 0 to %N {
                affine.store %c0, %argQ[%i] : memref<?xf64>
            }

            // Tiled computation: the j loop is tiled by 32 and the j-tile loop
            // is hoisted OUTSIDE the i loop. For each 32-wide block of j, we
            // sweep all i, so the block of s[j] and p[j] stays cache-resident
            // across the entire i sweep instead of being reloaded for every i.
            affine.for %jj = 0 to %M step 32 {
                affine.for %i = 0 to %N {
                    affine.for %j = 0 to 32 {
                        // s[jj+j] = s[jj+j] + r[i] * A[i][jj+j];
                        %s_j = affine.load %argS[%jj + %j] : memref<?xf64>
                        %r_i = affine.load %argR[%i] : memref<?xf64>
                        %a_ij = affine.load %argA[%i, %jj + %j] : memref<?x?xf64>
                        %prod_r_a = arith.mulf %r_i, %a_ij : f64
                        %new_s_j = arith.addf %s_j, %prod_r_a : f64
                        affine.store %new_s_j, %argS[%jj + %j] : memref<?xf64>

                        // q[i] = q[i] + A[i][jj+j] * p[jj+j];
                        %q_i = affine.load %argQ[%i] : memref<?xf64>
                        %p_j = affine.load %argP[%jj + %j] : memref<?xf64>
                        %prod_a_p = arith.mulf %a_ij, %p_j : f64
                        %new_q_i = arith.addf %q_i, %prod_a_p : f64
                        affine.store %new_q_i, %argQ[%i] : memref<?xf64>
                    }
                }
            }
        } {dmd.extract}
    return
    }
}
