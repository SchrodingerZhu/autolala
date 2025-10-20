#define DATA_TYPE float
#define I_SIZE 32
#define J_SIZE 32
#define K_SIZE 32

// Triplestore query: ij,i,jk->k
// Memory access pattern: A[i][j] * B[i] * C[j][k] -> D[k]
void kernel_triplestore_query_pattern(DATA_TYPE A[I_SIZE][36],  // J_SIZE=32 padded to 36
                                     DATA_TYPE B[36],  // I_SIZE=32 padded to 36
                                     DATA_TYPE C[J_SIZE][36],  // K_SIZE=32 padded to 36
                                     DATA_TYPE D[36]) {  // K_SIZE=32 padded to 36
  int i, j, k;
  
  // Initialize output
  for (k = 0; k < K_SIZE; k++)
    D[k] = 0;
  
  // Actual computation for: sum_i sum_j A[i][j] * B[i] * C[j][k] -> D[k]
  for (k = 0; k < K_SIZE; k++) {
    D[k] = 0.0f; // Initialize output D[k]
    for (i = 0; i < I_SIZE; i++) {
      for (j = 0; j < J_SIZE; j++) {
        D[k] += A[i][j] * B[i] * C[j][k];
      }
    }
  }
}
