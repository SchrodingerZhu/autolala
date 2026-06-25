array A[400, 400];
array B[400];
array C[400];
array D[400];
array E[400];
array F[400];
array G[400];
array H[400];
array I[400];

for i0 in 0 .. 1 {
    for i1 in 0 .. 400 {
        for i2 in 0 .. 400 {
            read A[i1, i2];
            read B[i1];
            read C[i2];
            read D[i1];
            read E[i2];
            write A[i1, i2];
        }
    }
    for i3 in 0 .. 400 {
        for i4 in 0 .. 400 {
            read F[i3];
            read A[i4, i3];
            read G[i4];
            write F[i3];
        }
    }
    for i5 in 0 .. 400 {
        read F[i5];
        read H[i5];
        write F[i5];
    }
    for i6 in 0 .. 400 {
        for i7 in 0 .. 400 {
            read I[i6];
            read A[i6, i7];
            read F[i7];
            write I[i6];
        }
    }
}
