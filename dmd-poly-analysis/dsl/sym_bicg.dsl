params p0, A_d0, p1, B_d0, C_d0, D_d0, D_d1, E_d0;
array A[A_d0];
array B[B_d0];
array C[C_d0];
array D[D_d0, D_d1];
array E[E_d0];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        write A[i1];
        for i2 in 0 .. p1 {
            read B[i2];
            read C[i1];
            read D[i1, i2];
            write B[i2];
            read A[i1];
            read E[i2];
            write A[i1];
        }
    }
}
