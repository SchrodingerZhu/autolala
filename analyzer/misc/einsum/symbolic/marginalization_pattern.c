#define DATA_TYPE float
#define LIMIT 1024
#define SMALL_LIMIT 16  // Use smaller dimension for 8D tensor
typedef __SIZE_TYPE__ size_t;

// Marginalization: ijklmnop->m
// Memory access pattern: sum over all indices except m
void kernel_marginalization_pattern(size_t I, size_t J, size_t K, size_t L, 
                                   size_t M, size_t N, size_t O, size_t P,
                                   DATA_TYPE A[SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT], 
                                   DATA_TYPE result[LIMIT]) {
  int i, j, k, l, m, n, o, p;
  
  // Initialize output
  for (m = 0; m < M; m++)
    result[m] = 0;
  
  // Actual computation for: sum_i sum_j sum_k sum_l sum_n sum_o sum_p A[i][j][k][l][m][n][o][p] -> result[m]
  for (m = 0; m < M; m++) {
    result[m] = 0.0f; // Initialize output result[m]
    for (i = 0; i < I; i++)
      for (j = 0; j < J; j++)
        for (k = 0; k < K; k++)
          for (l = 0; l < L; l++)
            for (n = 0; n < N; n++)
              for (o = 0; o < O; o++)
                for (p = 0; p < P; p++)
                  result[m] += A[i][j][k][l][m][n][o][p];
  }
}
