#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

// Bilinear transformation: ik,klj,il->ij
// Memory access pattern: sum_k sum_l A[i][k] * B[k][l][j] * C[i][l] -> D[i][j]
void kernel_bilinear_transformation_pattern(size_t I, size_t K, size_t L, size_t J,
                                           DATA_TYPE A[LIMIT][LIMIT], 
                                           DATA_TYPE B[LIMIT][LIMIT][LIMIT], 
                                           DATA_TYPE C[LIMIT][LIMIT], 
                                           DATA_TYPE D[LIMIT][LIMIT]) {
  int i, j, k, l;
  
  // Initialize output
  for (i = 0; i < I; i++)
    for (j = 0; j < J; j++)
      D[i][j] = 0;
  
  // Actual computation for: sum_k sum_l A[i][k] * B[k][l][j] * C[i][l] -> D[i][j]
  for (i = 0; i < I; i++) {
    for (j = 0; j < J; j++) {
      D[i][j] = 0.0f; // Initialize output D[i][j]
      for (k = 0; k < K; k++) {
        for (l = 0; l < L; l++) {
          D[i][j] += A[i][k] * B[k][l][j] * C[i][l];
        }
      }
    }
  }
}
