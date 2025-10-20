#define DATA_TYPE double
#define I_SIZE 64
#define J_SIZE 64
#define K_SIZE 64



volatile DATA_TYPE A[I_SIZE][J_SIZE];
volatile DATA_TYPE B[I_SIZE];
volatile DATA_TYPE C[J_SIZE][K_SIZE];
volatile DATA_TYPE D[K_SIZE];
// Triplestore query: ij,i,jk->k
// Memory access pattern: A[i][j] * B[i] * C[j][k] -> D[k]
void kernel_triplestore_query_pattern() {
  int i, j, k;
  
  // Initialize output
  for (k = 0; k < K_SIZE; k++)
    D[k] = 0;
  
  // Actual computation for: sum_i sum_j A[i][j] * B[i] * C[j][k] -> D[k]
  for (k = 0; k < K_SIZE; k++) {
    D[k] = 0.0; // Initialize output D[k]
    for (i = 0; i < I_SIZE; i++) {
      for (j = 0; j < J_SIZE; j++) {
        D[k] += A[i][j] * B[i] * C[j][k];
      }
    }
  }
}
