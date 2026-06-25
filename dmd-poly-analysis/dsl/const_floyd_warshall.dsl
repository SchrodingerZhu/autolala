array A[500, 500];

for i0 in 0 .. 500 {
    for i1 in 0 .. 500 {
        for i2 in 0 .. 500 {
            read A[i1, i2];
            read A[i1, i0];
            read A[i0, i2];
            write A[i1, i2];
        }
    }
}
