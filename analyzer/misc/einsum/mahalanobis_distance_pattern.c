#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

// Mahalanobis distance: i,ij,j->
// Memory access pattern: x[i] * S_inv[i][j] * y[j] -> scalar
void kernel_mahalanobis_distance_pattern(size_t N, 
                                        DATA_TYPE x[LIMIT], 
                                        DATA_TYPE S_inv[LIMIT][LIMIT], 
                                        DATA_TYPE y[LIMIT], 
                                        DATA_TYPE *result) {
  int i, j;
  
  // Actual computation for: sum_i sum_j x[i] * S_inv[i][j] * y[j]
  *result = 0.0f; // Initialize scalar result
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      *result += x[i] * S_inv[i][j] * y[j];
    }
  }
}
