params T, Ad0, Ad1, Bd0, Bd1, Cd0, Cd1;
array A[Ad0, Ad1];
array B[Bd0, Bd1];
array C[Cd0, Cd1];

for i0 in 0 .. T {
    for j0 in 0 .. T {
        for k0 in 0 .. T {
            for i in 0 .. 32 {
                for j in 0 .. 32 {
                    for k in 0 .. 32 {
                        read A[32 * i0 + i, 32 * k0 + k];
                        read B[32 * k0 + k, 32 * j0 + j];
                        write C[32 * i0 + i, 32 * j0 + j];
                    }
                }
            }
        }
    }
}
