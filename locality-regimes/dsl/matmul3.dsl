params n;
array A[n, n];
array B[n, n];
array C[n, n];

for i in 0 .. n {
    for j in 0 .. n {
        for k in 0 .. n {
            read A[i, k];
            read B[k, j];
            write C[i, j];
        }
    }
}
