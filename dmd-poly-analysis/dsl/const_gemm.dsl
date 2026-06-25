array A[200, 220];
array B[200, 240];
array C[240, 220];

for i0 in 0 .. 200 {
    for i1 in 0 .. 220 {
        read A[i0, i1];
        write A[i0, i1];
    }
    for i2 in 0 .. 240 {
        for i3 in 0 .. 220 {
            read A[i0, i3];
            read B[i0, i2];
            read C[i2, i3];
            write A[i0, i3];
        }
    }
}
