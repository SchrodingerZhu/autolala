params p0, p1, A_d0, A_d1, B_d0, B_d1, C_d0, C_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];
array C[C_d0, C_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. p1 {
            read A[i2, i1];
        }
        write B[i1, i1];
        for i3 in 0 .. p1 {
            read A[i3, i1];
            write C[i3, i1];
        }
        for i4 in i1 + 1 .. p0 {
            for i5 in 0 .. p1 {
                read C[i5, i1];
                read A[i5, i4];
            }
            write B[i1, i4];
            for i6 in 0 .. p1 {
                read A[i6, i4];
                read C[i6, i1];
                read B[i1, i4];
                write A[i6, i4];
            }
        }
    }
}
