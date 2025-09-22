#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

// Triplestore query: ij,i,jk->k
// Memory access pattern: A[i][j] * B[i] * C[j][k] -> D[k]
void kernel_triplestore_query_pattern(size_t I, size_t J, size_t K,
                                     DATA_TYPE A[LIMIT][LIMIT], 
                                     DATA_TYPE B[LIMIT], 
                                     DATA_TYPE C[LIMIT][LIMIT], 
                                     DATA_TYPE D[LIMIT]) {
  int i, j, k;
  
  // Initialize output
  for (k = 0; k < K; k++)
    D[k] = 0;
  
  // Actual computation for: sum_i sum_j A[i][j] * B[i] * C[j][k] -> D[k]
  for (k = 0; k < K; k++) {
    D[k] = 0.0f; // Initialize output D[k]
    for (i = 0; i < I; i++) {
      for (j = 0; j < J; j++) {
        D[k] += A[i][j] * B[i] * C[j][k];
      }
    }
  }
}
