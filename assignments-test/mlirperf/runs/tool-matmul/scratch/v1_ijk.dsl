params N;
array A[N, N];
array B[N, N];
array C[N, N];
for i in 0 .. N {
  for j in 0 .. N {
    for k in 0 .. N {
      read A[i, k];
      read B[k, j];
      update C[i, j];
    }
  }
}
