#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define SMALL_LIMIT 16  // Use very small dimensions for complex tensor networks

// 2×3-tensor network: ij,iml,lo,jk,kmn,no->
// Memory access pattern: complex tensor contraction to scalar
void kernel_tensor_network_2x3_pattern(size_t I, size_t J, size_t M, size_t L, 
                                       size_t K, size_t N, size_t O,
                                       DATA_TYPE A[SMALL_LIMIT][SMALL_LIMIT],           // ij
                                       DATA_TYPE B[SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT], // iml
                                       DATA_TYPE C[SMALL_LIMIT][SMALL_LIMIT],           // lo
                                       DATA_TYPE D[SMALL_LIMIT][SMALL_LIMIT],           // jk
                                       DATA_TYPE E[SMALL_LIMIT][SMALL_LIMIT][SMALL_LIMIT], // kmn
                                       DATA_TYPE F[SMALL_LIMIT][SMALL_LIMIT],           // no
                                       DATA_TYPE *result) {
  int i, j, m, l, k, n, o;
  
  *result = 0;
  
  // Access pattern for: sum over all indices A[i][j] * B[i][m][l] * C[l][o] * D[j][k] * E[k][m][n] * F[n][o] -> scalar
  for (i = 0; i < I; i++) {
    for (j = 0; j < J; j++) {
      A[i][j] = 0; // Access A[i][j]
      D[j][k] = 0; // Access D[j][k] (k will be looped)
      for (m = 0; m < M; m++) {
        B[i][m][l] = 0; // Access B[i][m][l] (l will be looped)
        for (l = 0; l < L; l++) {
          C[l][o] = 0; // Access C[l][o] (o will be looped)
          for (k = 0; k < K; k++) {
            for (n = 0; n < N; n++) {
              E[k][m][n] = 0; // Access E[k][m][n]
              for (o = 0; o < O; o++) {
                F[n][o] = 0; // Access F[n][o]
              }
            }
          }
        }
      }
    }
  }
}