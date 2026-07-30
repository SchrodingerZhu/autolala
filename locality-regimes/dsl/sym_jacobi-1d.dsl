params p0, p1, A_d0, B_d0;
array A[A_d0];
array B[B_d0];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 1 .. p1 {
            read A[i2 - 1];
            read A[i2];
            read A[i2 + 1];
            write B[i2];
        }
        for i3 in 1 .. p1 {
            read B[i3 - 1];
            read B[i3];
            read B[i3 + 1];
            write A[i3];
        }
    }
}
