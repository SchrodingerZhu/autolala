#define DATA_TYPE float
#define I_SIZE 16
#define J_SIZE 16
#define K_SIZE 16
#define L_SIZE 16
#define A_DIM 16
#define B_DIM 16
#define C_DIM 16
#define D_DIM 16

// Tucker decomposition: ijkl,ai,bj,ck,dl->abcd
// Memory access pattern: sum_i sum_j sum_k sum_l X[i][j][k][l] * A[a][i] * B[b][j] * C[c][k] * D[d][l] -> Y[a][b][c][d]
void kernel_tucker_decomposition_pattern(DATA_TYPE X[I_SIZE][J_SIZE][K_SIZE][L_SIZE],
                                        DATA_TYPE A[A_DIM][I_SIZE], 
                                        DATA_TYPE B[B_DIM][J_SIZE], 
                                        DATA_TYPE C[C_DIM][K_SIZE], 
                                        DATA_TYPE D[D_DIM][L_SIZE], 
                                        DATA_TYPE Y[A_DIM][B_DIM][C_DIM][D_DIM]) {
  int i, j, k, l, a, b, c, d;
  
  // Initialize output
  for (a = 0; a < A_DIM; a++)
    for (b = 0; b < B_DIM; b++)
      for (c = 0; c < C_DIM; c++)
        for (d = 0; d < D_DIM; d++)
          Y[a][b][c][d] = 0;
  
  // Actual computation for Tucker decomposition
  for (a = 0; a < A_DIM; a++) {
    for (b = 0; b < B_DIM; b++) {
      for (c = 0; c < C_DIM; c++) {
        for (d = 0; d < D_DIM; d++) {
          Y[a][b][c][d] = 0.0f; // Initialize output Y[a][b][c][d]
          for (i = 0; i < I_SIZE; i++) {
            for (j = 0; j < J_SIZE; j++) {
              for (k = 0; k < K_SIZE; k++) {
                for (l = 0; l < L_SIZE; l++) {
                  Y[a][b][c][d] += X[i][j][k][l] * A[a][i] * B[b][j] * C[c][k] * D[d][l];
                }
              }
            }
          }
        }
      }
    }
  }
}
