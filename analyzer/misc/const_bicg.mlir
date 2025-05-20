module {
    // Constants

    // Function definition
    func.func @bicg(%argA: memref<256x256xf32>, %argS: memref<256xf32>, %argR: memref<256xf32>, %argQ: memref<256xf32>, %argP: memref<256xf32>) {
        %c0 = arith.constant 0.0 : f32

        // Outer loop: for (i = 0; i < N; i++)
        affine.for %i = 0 to 256 {
            // Initialize q[i] to 0
            affine.store %c0, %argQ[%i] : memref<256xf32>

            // Inner loop: for (j = 0; j < M; j++)
            affine.for %j = 0 to 256 {
                // s[j] = s[j] + r[i] * A[i][j];
                %s_j = affine.load %argS[%j] : memref<256xf32>
                %r_i = affine.load %argR[%i] : memref<256xf32>
                %a_ij = affine.load %argA[%i, %j] : memref<256x256xf32>
                %prod_r_a = arith.mulf %r_i, %a_ij : f32
                %new_s_j = arith.addf %s_j, %prod_r_a : f32
                affine.store %new_s_j, %argS[%j] : memref<256xf32>

                // q[i] = q[i] + A[i][j] * p[j];
                %q_i = affine.load %argQ[%i] : memref<256xf32>
                %p_j = affine.load %argP[%j] : memref<256xf32>
                %prod_a_p = arith.mulf %a_ij, %p_j : f32
                %new_q_i = arith.addf %q_i, %prod_a_p : f32
                affine.store %new_q_i, %argQ[%i] : memref<256xf32>
            }
        } { slap.extract }

    return
    }
}
