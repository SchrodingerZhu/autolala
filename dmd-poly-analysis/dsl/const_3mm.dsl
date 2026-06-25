params A_d0, A_d1, B_d0, B_d1, C_d0, C_d1, D_d0, D_d1, E_d0, E_d1, F_d0, F_d1, G_d0, G_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];
array C[C_d0, C_d1];
array D[D_d0, D_d1];
array E[E_d0, E_d1];
array F[F_d0, F_d1];
array G[G_d0, G_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. 180 {
        for i2 in 0 .. 190 {
            write A[i1, i2];
            for i3 in 0 .. 200 {
                read A[i1, i2];
                read B[i1, i3];
                read C[i3, i2];
                write A[i1, i2];
            }
        }
    }
    for i4 in 0 .. 190 {
        for i5 in 0 .. 210 {
            write D[i4, i5];
            for i6 in 0 .. 220 {
                read D[i4, i5];
                read E[i4, i6];
                read F[i6, i5];
                write D[i4, i5];
            }
        }
    }
    for i7 in 0 .. 180 {
        for i8 in 0 .. 210 {
            write G[i7, i8];
            for i9 in 0 .. 190 {
                read G[i7, i8];
                read A[i7, i9];
                read D[i9, i8];
                write G[i7, i8];
            }
        }
    }
}
