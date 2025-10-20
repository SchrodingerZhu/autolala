#define DATA_TYPE float
#define I_SIZE 32
#define K_SIZE 24
#define L_SIZE 24
#define J_SIZE 32

// Bilinear transformation: ik,klj,il->ij
// Memory access pattern: sum_k sum_l A[i][k] * B[k][l][j] * C[i][l] -> D[i][j]
void kernel_bilinear_transformation_pattern(DATA_TYPE A[I_SIZE][24],  // K_SIZE=24 already multiple of 12
                                           DATA_TYPE B[K_SIZE][L_SIZE][36],  // J_SIZE=32 padded to 36
                                           DATA_TYPE C[I_SIZE][24],  // L_SIZE=24 already multiple of 12
                                           DATA_TYPE D[I_SIZE][36]) {  // J_SIZE=32 padded to 36
  int i, j, k, l;
  
  // Initialize output
  for (i = 0; i < I_SIZE; i++)
    for (j = 0; j < J_SIZE; j++)
      D[i][j] = 0;
  
  // Actual computation for: sum_k sum_l A[i][k] * B[k][l][j] * C[i][l] -> D[i][j]
  for (i = 0; i < I_SIZE; i++) {
    for (j = 0; j < J_SIZE; j++) {
      D[i][j] = 0.0f; // Initialize output D[i][j]
      for (k = 0; k < K_SIZE; k++) {
        for (l = 0; l < L_SIZE; l++) {
          D[i][j] += A[i][k] * B[k][l][j] * C[i][l];
        }
      }
    }
  }
}
