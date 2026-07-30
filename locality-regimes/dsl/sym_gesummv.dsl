params p0, A_d0, B_d0, C_d0, C_d1, D_d0, E_d0, E_d1;
array A[A_d0];
array B[B_d0];
array C[C_d0, C_d1];
array D[D_d0];
array E[E_d0, E_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        write A[i1];
        write B[i1];
        for i2 in 0 .. p0 {
            read A[i1];
            read C[i1, i2];
            read D[i2];
            write A[i1];
            read B[i1];
            read E[i1, i2];
            read D[i2];
            write B[i1];
        }
        read A[i1];
        read B[i1];
        write B[i1];
    }
}
