array A[240];
array B[62400];
array C[240];
array D[57600];

for i0 in 0 .. 1 {
    for i1 in 0 .. 240 {
        write A[i1];
        for i2 in 0 .. 260 {
            read A[i1];
            read B[i2 * 240 + i1];
            write A[i1];
        }
        read A[i1];
        write A[i1];
    }
    for i3 in 0 .. 240 {
        write C[i3];
        for i4 in 0 .. 260 {
            read C[i3];
            read B[i4 * 240 + i3];
            read A[i3];
            write C[i3];
        }
        read C[i3];
        write C[i3];
    }
    for i5 in 0 .. 260 {
        for i6 in 0 .. 240 {
            read B[i5 * 240 + i6];
            read A[i6];
            read C[i6];
            write B[i5 * 240 + i6];
        }
    }
    for i7 in 0 .. 239 {
        write D[i7 * 241];
        for i8 in i7 + 1 .. 240 {
            write D[i7 * 240 + i8];
            for i9 in 0 .. 260 {
                read D[i7 * 240 + i8];
                read B[i9 * 240 + i7];
                read B[i9 * 240 + i8];
                write D[i7 * 240 + i8];
            }
            read D[i7 * 240 + i8];
            write D[i8 * 240 + i7];
        }
    }
    write D[57599];
}
