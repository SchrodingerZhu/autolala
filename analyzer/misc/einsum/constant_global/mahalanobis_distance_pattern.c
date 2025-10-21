#define DATA_TYPE double
#define N_SIZE 64



volatile DATA_TYPE x[72];
volatile DATA_TYPE S_inv[N_SIZE][72];
volatile DATA_TYPE y[72];
volatile DATA_TYPE result;
// Mahalanobis distance: i,ij,j->
// Memory access pattern: x[i] * S_inv[i][j] * y[j] -> scalar
void kernel_mahalanobis_distance_pattern() {
  int i, j;
  
  // Actual computation for: sum_i sum_j x[i] * S_inv[i][j] * y[j]
  result = 0.0; // Initialize scalar result
  for (i = 0; i < N_SIZE; i++) {
    for (j = 0; j < N_SIZE; j++) {
      result += x[i] * S_inv[i][j] * y[j];
    }
  }
}
