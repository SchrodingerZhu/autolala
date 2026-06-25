params p0, p1, A_d0, A_d1, p2, B_d0, B_d1, C_d0, C_d1, p3, D_d0, D_d1, E_d0, E_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];
array C[C_d0, C_d1];
array D[D_d0, D_d1];
array E[E_d0, E_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. p1 {
            write A[i1, i2];
            for i3 in 0 .. p2 {
                read A[i1, i2];
                read B[i1, i3];
                read C[i3, i2];
                write A[i1, i2];
            }
        }
    }
    for i4 in 0 .. p0 {
        for i5 in 0 .. p3 {
            read D[i4, i5];
            write D[i4, i5];
            for i6 in 0 .. p1 {
                read D[i4, i5];
                read A[i4, i6];
                read E[i6, i5];
                write D[i4, i5];
            }
        }
    }
}
