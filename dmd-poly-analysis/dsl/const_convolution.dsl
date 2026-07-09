array A[16, 16];
array B[512, 512];
array C[497, 497];

for i0 in 0 .. 497 {
    for i1 in 0 .. 497 {
        for i2 in 0 .. 16 {
            for i3 in 0 .. 16 {
                read A[i2, i3];
                read B[i0 + i2, i1 + i3];
            }
        }
        write C[i0, i1];
    }
}
