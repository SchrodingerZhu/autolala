#define DATA_TYPE float
#define N_SIZE 64

// Mahalanobis distance: i,ij,j->
// Memory access pattern: x[i] * S_inv[i][j] * y[j] -> scalar
void kernel_mahalanobis_distance_pattern(DATA_TYPE x[N_SIZE], 
                                        DATA_TYPE S_inv[N_SIZE][N_SIZE], 
                                        DATA_TYPE y[N_SIZE], 
                                        DATA_TYPE *result) {
  int i, j;
  
  // Actual computation for: sum_i sum_j x[i] * S_inv[i][j] * y[j]
  *result = 0.0f; // Initialize scalar result
  for (i = 0; i < N_SIZE; i++) {
    for (j = 0; j < N_SIZE; j++) {
      *result += x[i] * S_inv[i][j] * y[j];
    }
  }
}
