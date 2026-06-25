params A_d0, A_d1;
array A[A_d0, A_d1];

for i0 in 0 .. 400 {
    for i1 in i0 .. 400 {
        read A[i0, i1];
        write A[i0, i1];
    }
    for i2 in i0 + 1 .. 400 {
        read A[i2, i0];
        read A[i0, i0];
        write A[i2, i0];
        for i3 in i0 + 1 .. 400 {
            read A[i2, i3];
            read A[i0, i3];
            read A[i2, i0];
            write A[i2, i3];
        }
    }
}
