params p0, A_d0, p1, B_d0, B_d1, C_d0, D_d0, D_d1;
array A[A_d0];
array B[B_d0, B_d1];
array C[C_d0];
array D[D_d0, D_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        write A[i1];
        for i2 in 0 .. p1 {
            read A[i1];
            read B[i2, i1];
            write A[i1];
        }
        read A[i1];
        write A[i1];
    }
    for i3 in 0 .. p0 {
        write C[i3];
        for i4 in 0 .. p1 {
            read C[i3];
            read B[i4, i3];
            read A[i3];
            write C[i3];
        }
        read C[i3];
        write C[i3];
    }
    for i5 in 0 .. p1 {
        for i6 in 0 .. p0 {
            read B[i5, i6];
            read A[i6];
            read C[i6];
            write B[i5, i6];
        }
    }
    for i7 in 0 .. p0 - 1 {
        write D[i7, i7];
        for i8 in i7 + 1 .. p0 {
            write D[i7, i8];
            for i9 in 0 .. p1 {
                read D[i7, i8];
                read B[i9, i7];
                read B[i9, i8];
                write D[i7, i8];
            }
            read D[i7, i8];
            write D[i8, i7];
        }
    }
    write D[p0 - 1, p0 - 1];
}
