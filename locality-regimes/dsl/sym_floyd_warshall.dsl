params p0, A_d0, A_d1;
array A[A_d0, A_d1];

for i0 in 0 .. p0 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. p0 {
            read A[i1, i2];
            read A[i1, i0];
            read A[i0, i2];
            write A[i1, i2];
        }
    }
}
