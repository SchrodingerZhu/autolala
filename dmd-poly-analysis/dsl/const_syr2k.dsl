array A[240, 240];
array B[240, 200];
array C[240, 200];

for i0 in 0 .. 240 {
    for i1 in 0 .. i0 + 1 {
        read A[i0, i1];
        write A[i0, i1];
    }
    for i2 in 0 .. 200 {
        for i3 in 0 .. i0 + 1 {
            read A[i0, i3];
            read B[i3, i2];
            read C[i0, i2];
            read C[i3, i2];
            read B[i0, i2];
            write A[i0, i3];
        }
    }
}
