params p0, p1, A_d0, A_d1, B_d0, B_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 1 .. p1 {
            for i3 in 1 .. p1 {
                read A[i2, i3];
                read A[i2, i3 - 1];
                read A[i2, i3 + 1];
                read A[i2 + 1, i3];
                read A[i2 - 1, i3];
                write B[i2, i3];
            }
        }
        for i4 in 1 .. p1 {
            for i5 in 1 .. p1 {
                read B[i4, i5];
                read B[i4, i5 - 1];
                read B[i4, i5 + 1];
                read B[i4 + 1, i5];
                read B[i4 - 1, i5];
                write A[i4, i5];
            }
        }
    }
}
