params p0, p1, A_d0, A_d1, B_d0, B_d1, C_d0, C_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];
array C[C_d0, C_d1];

for i0 in 0 .. 1 {
    for i1 in 0 .. p0 - p1 {
        for i2 in 0 .. p0 - p1 {
            for i3 in 0 .. p1 {
                for i4 in 0 .. p1 {
                    read A[i3, i4];
                    read B[i1 + i3, i2 + i4];
                }
            }
            write C[i1, i2];
        }
    }
}
