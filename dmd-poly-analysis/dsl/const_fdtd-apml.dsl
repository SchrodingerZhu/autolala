params A_d0, A_d1, A_d2, B_d0, B_d1, B_d2, C_d0, C_d1, D_d0, E_d0, F_d0, F_d1, F_d2, G_d0, G_d1, H_d0, I_d0, J_d0, J_d1, J_d2, K_d0, L_d0, M_d0, M_d1, N_d0, N_d1;
array A[A_d0, A_d1, A_d2];
array B[B_d0, B_d1, B_d2];
array C[C_d0, C_d1];
array D[D_d0];
array E[E_d0];
array F[F_d0, F_d1, F_d2];
array G[G_d0, G_d1];
array H[H_d0];
array I[I_d0];
array J[J_d0, J_d1, J_d2];
array K[K_d0];
array L[L_d0];
array M[M_d0, M_d1];
array N[N_d0, N_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. 256 {
        for i2 in 0 .. 256 {
            for i3 in 0 .. 256 {
                read A[i1, i2, i3];
                read A[i1, i2 + 1, i3];
                read B[i1, i2, i3 + 1];
                read B[i1, i2, i3];
                write C[i1, i2];
                read D[i2];
                read E[i2];
                read F[i1, i2, i3];
                write G[i1, i2];
                read H[i3];
                read I[i3];
                read J[i1, i2, i3];
                read K[i1];
                read L[i1];
                write J[i1, i2, i3];
                write F[i1, i2, i3];
            }
            read A[i1, i2, 256];
            read A[i1, i2 + 1, 256];
            read M[i1, i2];
            read B[i1, i2, 256];
            write C[i1, i2];
            read D[i2];
            read E[i2];
            read F[i1, i2, 256];
            write G[i1, i2];
            read H[256];
            read I[256];
            read J[i1, i2, 256];
            read K[i1];
            read L[i1];
            write J[i1, i2, 256];
            write F[i1, i2, 256];
            for i4 in 0 .. 256 {
                read A[i1, 256, i4];
                read N[i1, i4];
                read B[i1, 256, i4 + 1];
                read B[i1, 256, i4];
                write C[i1, i2];
                read D[256];
                read E[i2];
                read F[i1, i2, i4];
                write G[i1, i2];
                read H[i4];
                read I[i4];
                read J[i1, 256, i4];
                read F[i1, 256, i4];
                read K[i1];
                read L[i1];
                write J[i1, 256, i4];
                write F[i1, 256, i4];
            }
            read A[i1, 256, 256];
            read N[i1, 256];
            read M[i1, 256];
            read B[i1, 256, 256];
            write C[i1, i2];
            read D[256];
            read E[256];
            read F[i1, 256, 256];
            write G[i1, i2];
            read H[256];
            read I[256];
            read J[i1, 256, 256];
            read K[i1];
            read L[i1];
            write J[i1, 256, 256];
            write F[i1, 256, 256];
        }
    }
}
