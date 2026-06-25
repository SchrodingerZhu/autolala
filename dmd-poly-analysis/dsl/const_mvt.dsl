array A[400];
array B[400, 400];
array C[400];
array D[400];
array E[400];

for i0 in 0 .. 1 {
    for i1 in 0 .. 400 {
        for i2 in 0 .. 400 {
            read A[i1];
            read B[i1, i2];
            read C[i2];
            write A[i1];
        }
    }
    for i3 in 0 .. 400 {
        for i4 in 0 .. 400 {
            read D[i3];
            read B[i4, i3];
            read E[i4];
            write D[i3];
        }
    }
}
