params p0, A_d0, A_d1;
array A[A_d0, A_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in i1 .. p0 {
            read A[i1, i2];
            write A[i1, i2];
        }
        for i3 in i1 + 1 .. p0 {
            read A[i3, i1];
            read A[i1, i1];
            write A[i3, i1];
            for i4 in i1 + 1 .. p0 {
                read A[i3, i4];
                read A[i1, i4];
                read A[i3, i1];
                write A[i3, i4];
            }
        }
    }
}
