array A[400, 400];

for i0 in 0 .. 100 {
    for i1 in 1 .. 399 {
        for i2 in 1 .. 399 {
            read A[i1 - 1, i2 - 1];
            read A[i1 - 1, i2];
            read A[i1 - 1, i2 + 1];
            read A[i1, i2 - 1];
            read A[i1, i2];
            read A[i1, i2 + 1];
            read A[i1 + 1, i2 - 1];
            read A[i1 + 1, i2];
            read A[i1 + 1, i2 + 1];
            write A[i1, i2];
        }
    }
}
