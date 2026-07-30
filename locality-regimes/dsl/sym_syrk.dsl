params p0, A_d0, A_d1, p1, B_d0, B_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. i1 + 1 {
            read A[i1, i2];
            write A[i1, i2];
        }
        for i3 in 0 .. p1 {
            for i4 in 0 .. i1 + 1 {
                read A[i1, i4];
                read B[i1, i3];
                read B[i4, i3];
                write A[i1, i4];
            }
        }
    }
}
