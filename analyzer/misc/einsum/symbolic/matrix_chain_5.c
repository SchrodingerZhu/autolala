#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

// Matrix chain multiplication: ik,kl,lm,mn,nj->ij
void kernel_matrix_chain_5(size_t I, size_t K, size_t L, size_t M, size_t N, size_t J,
                          DATA_TYPE A[LIMIT][LIMIT], 
                          DATA_TYPE B[LIMIT][LIMIT], 
                          DATA_TYPE C[LIMIT][LIMIT], 
                          DATA_TYPE D[LIMIT][LIMIT], 
                          DATA_TYPE E[LIMIT][LIMIT], 
                          DATA_TYPE result[LIMIT][LIMIT], 
                          DATA_TYPE tmp1[LIMIT][LIMIT], 
                          DATA_TYPE tmp2[LIMIT][LIMIT], 
                          DATA_TYPE tmp3[LIMIT][LIMIT]) {
  int i, j, k, l, m, n;
  
  // tmp1 := A * B (ik,kl->il)
  for (i = 0; i < I; i++)
    for (l = 0; l < L; l++) {
      tmp1[i][l] = 0.0f;
      for (k = 0; k < K; k++)
        tmp1[i][l] += A[i][k] * B[k][l];
    }
  
  // tmp2 := tmp1 * C (il,lm->im)  
  for (i = 0; i < I; i++)
    for (m = 0; m < M; m++) {
      tmp2[i][m] = 0.0f;
      for (l = 0; l < L; l++)
        tmp2[i][m] += tmp1[i][l] * C[l][m];
    }
  
  // tmp3 := tmp2 * D (im,mn->in)
  for (i = 0; i < I; i++)
    for (n = 0; n < N; n++) {
      tmp3[i][n] = 0.0f;
      for (m = 0; m < M; m++)
        tmp3[i][n] += tmp2[i][m] * D[m][n];
    }
  
  // result := tmp3 * E (in,nj->ij)
  for (i = 0; i < I; i++)
    for (j = 0; j < J; j++) {
      result[i][j] = 0.0f;
      for (n = 0; n < N; n++)
        result[i][j] += tmp3[i][n] * E[n][j];
    }
}