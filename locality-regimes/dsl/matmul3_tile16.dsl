params T, Ad0, Ad1, Bd0, Bd1, Cd0, Cd1;
array A[Ad0, Ad1];
array B[Bd0, Bd1];
array C[Cd0, Cd1];

for i0 in 0 .. T {
    for j0 in 0 .. T {
        for k0 in 0 .. T {
            for i in 0 .. 16 {
                for j in 0 .. 16 {
                    for k in 0 .. 16 {
                        read A[16 * i0 + i, 16 * k0 + k];
                        read B[16 * k0 + k, 16 * j0 + j];
                        write C[16 * i0 + i, 16 * j0 + j];
                    }
                }
            }
        }
    }
}
