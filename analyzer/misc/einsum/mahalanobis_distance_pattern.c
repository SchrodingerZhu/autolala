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
  
  // Access pattern for: sum_i sum_j x[i] * S_inv[i][j] * y[j]
  for (i = 0; i < N; i++) {
    x[i] = 0; // Access x[i]
    for (j = 0; j < N; j++) {
      S_inv[i][j] = 0; // Access S_inv[i][j] 
      y[j] = 0; // Access y[j]
    }
  }
  *result = 0; // Final scalar result
}