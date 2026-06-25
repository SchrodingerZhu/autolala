array A[400, 400];

for i0 in 0 .. 400 {
    for i1 in 0 .. i0 {
        for i2 in 0 .. i1 {
            read A[i0, i2];
            read A[i1, i2];
            read A[i0, i1];
            write A[i0, i1];
        }
        read A[i1, i1];
        read A[i0, i1];
        write A[i0, i1];
    }
    for i3 in 0 .. i0 {
        read A[i0, i3];
        read A[i0, i0];
        write A[i0, i0];
    }
    read A[i0, i0];
    write A[i0, i0];
}
