params A_d0, A_d1, B_d0, B_d1, C_d0, C_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];
array C[C_d0, C_d1];

for i0 in 0 .. 240 {
    for i1 in 0 .. 128 {
        read A[i1, i0];
    }
    write B[i0, i0];
    for i2 in 0 .. 200 {
        read A[i2, i0];
        write C[i2, i0];
    }
    for i3 in i0 + 1 .. 240 {
        for i4 in 0 .. 200 {
            read C[i4, i0];
            read A[i4, i3];
        }
        write B[i0, i3];
        for i5 in 0 .. 200 {
            read A[i5, i3];
            read C[i5, i0];
            read B[i0, i3];
            write A[i5, i3];
        }
    }
}
