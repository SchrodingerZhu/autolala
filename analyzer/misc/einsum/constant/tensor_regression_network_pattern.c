#define DATA_TYPE float
#define TINY_SIZE 4
#define A_SIZE 4
#define B_SIZE 4
#define C_SIZE 4
#define D_SIZE 4
#define E_SIZE 4
#define F_SIZE 4
#define G_SIZE 4
#define H_SIZE 4
#define I_SIZE 4
#define J_SIZE 4
#define K_SIZE 4

// Tensor regression network: abcde,fghij,bf,cg,dh,ei,kj->ak
// Memory access pattern: complex tensor network with multiple contractions
void kernel_tensor_regression_network_pattern(DATA_TYPE X[A_SIZE][B_SIZE][C_SIZE][D_SIZE][E_SIZE], // abcde
                                             DATA_TYPE Y[F_SIZE][G_SIZE][H_SIZE][I_SIZE][J_SIZE], // fghij  
                                             DATA_TYPE M1[B_SIZE][F_SIZE], // bf
                                             DATA_TYPE M2[C_SIZE][G_SIZE], // cg
                                             DATA_TYPE M3[D_SIZE][H_SIZE], // dh
                                             DATA_TYPE M4[E_SIZE][I_SIZE], // ei
                                             DATA_TYPE M5[K_SIZE][J_SIZE], // kj
                                             DATA_TYPE result[A_SIZE][K_SIZE]) { // ak
  int a, b, c, d, e, f, g, h, i, j, k;
  
  // Initialize output
  for (a = 0; a < A_SIZE; a++)
    for (k = 0; k < K_SIZE; k++)
      result[a][k] = 0;
  
  // Actual computation for tensor regression network contraction
  for (a = 0; a < A_SIZE; a++) {
    for (k = 0; k < K_SIZE; k++) {
      result[a][k] = 0.0f; // Initialize output result[a][k]
      for (b = 0; b < B_SIZE; b++) {
        for (c = 0; c < C_SIZE; c++) {
          for (d = 0; d < D_SIZE; d++) {
            for (e = 0; e < E_SIZE; e++) {
              for (f = 0; f < F_SIZE; f++) {
                for (g = 0; g < G_SIZE; g++) {
                  for (h = 0; h < H_SIZE; h++) {
                    for (i = 0; i < I_SIZE; i++) {
                      for (j = 0; j < J_SIZE; j++) {
                        result[a][k] += X[a][b][c][d][e] * Y[f][g][h][i][j] * 
                                        M1[b][f] * M2[c][g] * M3[d][h] * M4[e][i] * M5[k][j];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
