array A[256];
array B[62560];
array C[57600];

for i0 in 0 .. 1 {
    for i1 in 0 .. 256 {
        write A[i1];
        for i2 in 0 .. 256 {
            read A[i1];
            read B[i2 * 256 + i1];
            write A[i1];
        }
        read A[i1];
        write A[i1];
    }
    for i3 in 0 .. 256 {
        for i4 in 0 .. 256 {
            read B[i3 * 256 + i4];
            read A[i4];
            write B[i3 * 256 + i4];
        }
    }
    for i5 in 0 .. 256 {
        for i6 in i5 .. 256 {
            write C[i5 * 256 + i6];
            for i7 in 0 .. 256 {
                read B[i7 * 256 + i5];
                read B[i7 * 256 + i6];
                read C[i5 * 256 + i6];
                write C[i5 * 256 + i6];
            }
            read C[i5 * 256 + i6];
            write C[i5 * 256 + i6];
            write C[i6 * 256 + i5];
        }
    }
}
