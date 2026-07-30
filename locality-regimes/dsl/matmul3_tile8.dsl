params T, Ad0, Ad1, Bd0, Bd1, Cd0, Cd1;
array A[Ad0, Ad1];
array B[Bd0, Bd1];
array C[Cd0, Cd1];

for i0 in 0 .. T {
    for j0 in 0 .. T {
        for k0 in 0 .. T {
            for i in 0 .. 8 {
                for j in 0 .. 8 {
                    for k in 0 .. 8 {
                        read A[8 * i0 + i, 8 * k0 + k];
                        read B[8 * k0 + k, 8 * j0 + j];
                        write C[8 * i0 + i, 8 * j0 + j];
                    }
                }
            }
        }
    }
}
