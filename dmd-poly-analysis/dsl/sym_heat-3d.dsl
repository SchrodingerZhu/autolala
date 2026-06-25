params p0, p1, A_d0, A_d1, A_d2, B_d0, B_d1, B_d2;
array A[A_d0, A_d1, A_d2];
array B[B_d0, B_d1, B_d2];

for i0 in 0 .. 1 {
    for i1 in 1 .. p0 {
        for i2 in 1 .. p1 {
            for i3 in 1 .. p1 {
                for i4 in 1 .. p1 {
                    read A[i2, i3, i4];
                    read A[i2 + 1, i3, i4];
                    read A[i2 - 1, i3, i4];
                    read A[i2, i3 + 1, i4];
                    read A[i2, i3 - 1, i4];
                    read A[i2, i3, i4 + 1];
                    read A[i2, i3, i4 - 1];
                    write B[i2, i3, i4];
                }
            }
        }
        for i5 in 1 .. p1 {
            for i6 in 1 .. p1 {
                for i7 in 1 .. p1 {
                    read B[i5, i6, i7];
                    read B[i5 + 1, i6, i7];
                    read B[i5 - 1, i6, i7];
                    read B[i5, i6 + 1, i7];
                    read B[i5, i6 - 1, i7];
                    read B[i5, i6, i7 + 1];
                    read B[i5, i6, i7 - 1];
                    write A[i5, i6, i7];
                }
            }
        }
    }
}
