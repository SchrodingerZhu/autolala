params p0, p1, p2, A_d0, B_d0, B_d1, B_d2, C_d0, C_d1;
array A[A_d0];
array B[B_d0, B_d1, B_d2];
array C[C_d0, C_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 {
        for i2 in 0 .. p1 {
            for i3 in 0 .. p2 {
                write A[i3];
                for i4 in 0 .. p2 {
                    read B[i1, i2, i4];
                    read C[i4, i3];
                    read A[i3];
                    write A[i3];
                }
            }
            for i5 in 0 .. p2 {
                read A[i5];
                write B[i1, i2, i5];
            }
        }
    }
}
