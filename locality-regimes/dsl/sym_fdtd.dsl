params p0, p1, A_d0, B_d0, B_d1, p2, C_d0, C_d1, D_d0, D_d1;
array A[A_d0];
array B[B_d0, B_d1];
array C[C_d0, C_d1];
array D[D_d0, D_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. p1 {
            read A[i1];
            write B[0, i2];
        }
        for i3 in 1 .. p2 {
            for i4 in 0 .. p1 {
                read B[i3, i4];
                read C[i3, i4];
                read C[i3 - 1, i4];
                write B[i3, i4];
            }
        }
        for i5 in 0 .. p2 {
            for i6 in 1 .. p1 {
                read D[i5, i6];
                read C[i5, i6];
                read C[i5, i6 - 1];
                write D[i5, i6];
            }
        }
        for i7 in 0 .. p2 - 1 {
            for i8 in 0 .. p1 - 1 {
                read C[i7, i8];
                read D[i7, i8 + 1];
                read D[i7, i8];
                read B[i7 + 1, i8];
                read B[i7, i8];
                write C[i7, i8];
            }
        }
    }
}
