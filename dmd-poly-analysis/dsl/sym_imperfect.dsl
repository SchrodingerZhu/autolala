params p0, p1, A_d0, p2, B_d0, C_d0;
array A[A_d0];
array B[B_d0];
array C[C_d0];

for i0 in 0 .. p0 {
    for i1 in 0 .. p1 {
        read A[i0 * 16 + i1];
        for i2 in 0 .. p2 {
            for i3 in 0 .. 16 {
                read B[i0 * 16 + i3 + i1 * 16];
                write C[i2 * 16 + i3];
            }
        }
        for i4 in 0 .. p2 {
            write C[i0 * 16 + i4];
        }
    }
}
