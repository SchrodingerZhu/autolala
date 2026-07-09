array A[200, 240];
array B[200, 240];
array C[200, 200];

for i0 in 0 .. 200 {
    for i1 in 0 .. 240 {
        for i2 in 0 .. i0 {
            read A[i2, i1];
            read B[i0, i1];
            read C[i0, i2];
            write A[i2, i1];
            read B[i2, i1];
        }
        read A[i0, i1];
        read B[i0, i1];
        read C[i0, i0];
        write A[i0, i1];
    }
}
