array A[60];
array B[50, 40, 60];
array C[60, 60];

for i0 in 0 .. 64 {
    for i1 in 0 .. 64 {
        for i2 in 0 .. 64 {
            write A[i2];
            for i3 in 0 .. 64 {
                read B[i0, i1, i3];
                read C[i3, i2];
                read A[i2];
                write A[i2];
            }
        }
        for i4 in 0 .. 64 {
            read A[i4];
            write B[i0, i1, i4];
        }
    }
}
