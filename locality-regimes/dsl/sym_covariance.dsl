params p0, A_d0, p1, B_d0, B_d1, C_d0, C_d1;
array A[A_d0];
array B[B_d0, B_d1];
array C[C_d0, C_d1];

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
    for i3 in 0 .. p1 {
        for i4 in 0 .. p0 {
            read B[i3, i4];
            read A[i4];
            write B[i3, i4];
        }
    }
    for i5 in 0 .. p0 {
        for i6 in i5 .. p0 {
            write C[i5, i6];
            for i7 in 0 .. p1 {
                read C[i5, i6];
                read B[i7, i5];
                read B[i7, i6];
                write C[i5, i6];
            }
            read C[i5, i6];
            write C[i5, i6];
            read C[i5, i6];
            write C[i6, i5];
        }
    }
}
