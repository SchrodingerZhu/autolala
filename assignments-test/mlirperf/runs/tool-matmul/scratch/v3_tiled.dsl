params N;
array A[N, N];
array B[N, N];
array C[N, N];
for ii in 0 .. N step 64 {
  for kk in 0 .. N step 256 {
    for jj in 0 .. N step 256 {
      for i in ii .. ii + 64 {
        for k in kk .. kk + 256 {
          for j in jj .. jj + 256 {
            read A[i, k];
            read B[k, j];
            update C[i, j];
          }
        }
      }
    }
  }
}
