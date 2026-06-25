params A_d0, B_d0, B_d1, C_d0, D_d0;
array A[A_d0];
array B[B_d0, B_d1];
array C[C_d0];
array D[D_d0];

for i0 in 0 .. 1 {
    for i1 in 0 .. 390 {
        write A[i1];
        for i2 in 0 .. 410 {
            read A[i1];
            read B[i1, i2];
            read C[i2];
            write A[i1];
        }
        for i3 in 0 .. 410 {
            read D[i3];
            read B[i1, i3];
            read A[i1];
            write D[i3];
        }
    }
}
