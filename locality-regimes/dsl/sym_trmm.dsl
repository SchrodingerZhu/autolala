params p0, p1, A_d0, A_d1, B_d0, B_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. p1 {
            for i3 in i1 + 1 .. p0 {
                read A[i1, i2];
                read B[i3, i1];
                read A[i3, i2];
                write A[i1, i2];
            }
            read A[i1, i2];
            write A[i1, i2];
        }
    }
}
