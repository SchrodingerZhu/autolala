module {
    // Constants

    // Function definition
    func.func @bicg(%argA: memref<?x?xf64>, %argS: memref<?xf64>, %argR: memref<?xf64>, %argQ: memref<?xf64>, %argP: memref<?xf64>, %M : index, %N : index) {
        affine.for %loop_once = 0 to 1 {
            %c0 = arith.constant 0.0 : f64

            // Outer loop: for (i = 0; i < N; i++)
            affine.for %i = 0 to %N {
                // Initialize q[i] to 0
                affine.store %c0, %argQ[%i] : memref<?xf64>

                // Inner loop: for (j = 0; j < M; j++)
                affine.for %j = 0 to %M {
                    // s[j] = s[j] + r[i] * A[i][j];
                    %s_j = affine.load %argS[%j] : memref<?xf64>
                    %r_i = affine.load %argR[%i] : memref<?xf64>
                    %a_ij = affine.load %argA[%i, %j] : memref<?x?xf64>
                    %prod_r_a = arith.mulf %r_i, %a_ij : f64
                    %new_s_j = arith.addf %s_j, %prod_r_a : f64
                    affine.store %new_s_j, %argS[%j] : memref<?xf64>

                    // q[i] = q[i] + A[i][j] * p[j];
                    %q_i = affine.load %argQ[%i] : memref<?xf64>
                    %p_j = affine.load %argP[%j] : memref<?xf64>
                    %prod_a_p = arith.mulf %a_ij, %p_j : f64
                    %new_q_i = arith.addf %q_i, %prod_a_p : f64
                    affine.store %new_q_i, %argQ[%i] : memref<?xf64>
                }
            }
        }
    return
    }
}
