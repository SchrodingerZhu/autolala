params p0, A_d0, A_d1;
array A[A_d0, A_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. i1 {
            for i3 in 0 .. i2 {
                read A[i1, i3];
                read A[i2, i3];
                read A[i1, i2];
                write A[i1, i2];
            }
            read A[i2, i2];
            read A[i1, i2];
            write A[i1, i2];
        }
        for i4 in 0 .. i1 {
            read A[i1, i4];
            read A[i1, i1];
            write A[i1, i1];
        }
        read A[i1, i1];
        write A[i1, i1];
    }
}
