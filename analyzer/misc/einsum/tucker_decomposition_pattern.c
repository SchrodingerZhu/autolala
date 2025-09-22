#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define SMALL_LIMIT 32  // Use smaller dimensions for higher-order tensors

// Tucker decomposition: ijkl,ai,bj,ck,dl->abcd
// Memory access pattern: sum_i sum_j sum_k sum_l X[i][j][k][l] * A[a][i] * B[b][j] * C[c][k] * D[d][l] -> Y[a][b][c][d]
void kernel_tucker_decomposition_pattern(size_t I, size_t J, size_t K, size_t L,
                                        size_t A_dim, size_t B_dim, size_t C_dim, size_t D_dim,
                                        DATA_TYPE X[SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT],
                                        DATA_TYPE A[SMALL_LIMIT][SMALL_LIMIT], 
                                        DATA_TYPE B[SMALL_LIMIT][SMALL_LIMIT], 
                                        DATA_TYPE C[SMALL_LIMIT][SMALL_LIMIT], 
                                        DATA_TYPE D[SMALL_LIMIT][SMALL_LIMIT], 
                                        DATA_TYPE Y[SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT]) {
  int i, j, k, l, a, b, c, d;
  
  // Initialize output
  for (a = 0; a < A_dim; a++)
    for (b = 0; b < B_dim; b++)
      for (c = 0; c < C_dim; c++)
        for (d = 0; d < D_dim; d++)
          Y[a][b][c][d] = 0;
  
  // Actual computation for Tucker decomposition
  for (a = 0; a < A_dim; a++) {
    for (b = 0; b < B_dim; b++) {
      for (c = 0; c < C_dim; c++) {
        for (d = 0; d < D_dim; d++) {
          Y[a][b][c][d] = 0.0f; // Initialize output Y[a][b][c][d]
          for (i = 0; i < I; i++) {
            for (j = 0; j < J; j++) {
              for (k = 0; k < K; k++) {
                for (l = 0; l < L; l++) {
                  Y[a][b][c][d] += X[i][j][k][l] * A[a][i] * B[b][j] * C[c][k] * D[d][l];
                }
              }
            }
          }
        }
      }
    }
  }
}
