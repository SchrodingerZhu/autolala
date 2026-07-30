params p0, A_d0, p1, B_d0, C_d0, C_d1, D_d0;
array A[A_d0];
array B[B_d0];
array C[C_d0, C_d1];
array D[D_d0];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        write A[i1];
    }
    for i2 in 0 .. p1 {
        write B[i2];
        for i3 in 0 .. p0 {
            read B[i2];
            read C[i2, i3];
            read D[i3];
            write B[i2];
        }
        for i4 in 0 .. p0 {
            read A[i4];
            read C[i2, i4];
            read B[i2];
            write A[i4];
        }
    }
}
