params p0, A_d0, A_d1, B_d0, C_d0, D_d0, E_d0, F_d0, G_d0, H_d0, I_d0;
array A[A_d0, A_d1];
array B[B_d0];
array C[C_d0];
array D[D_d0];
array E[E_d0];
array F[F_d0];
array G[G_d0];
array H[H_d0];
array I[I_d0];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. p0 {
            read A[i1, i2];
            read B[i1];
            read C[i2];
            read D[i1];
            read E[i2];
            write A[i1, i2];
        }
    }
    for i3 in 0 .. p0 {
        for i4 in 0 .. p0 {
            read F[i3];
            read A[i4, i3];
            read G[i4];
            write F[i3];
        }
    }
    for i5 in 0 .. p0 {
        read F[i5];
        read H[i5];
        write F[i5];
    }
    for i6 in 0 .. p0 {
        for i7 in 0 .. p0 {
            read I[i6];
            read A[i6, i7];
            read F[i7];
            write I[i6];
        }
    }
}
