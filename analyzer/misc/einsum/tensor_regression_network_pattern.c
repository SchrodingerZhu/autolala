#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define TINY_LIMIT 8

// Tensor regression network: abcde,fghij,bf,cg,dh,ei,kj->ak
// Memory access pattern: complex tensor network with multiple contractions
void kernel_tensor_regression_network_pattern(size_t A, size_t B, size_t C, size_t D, size_t E,
                                             size_t F, size_t G, size_t H, size_t I, size_t J, size_t K,
                                             DATA_TYPE X[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], // abcde
                                             DATA_TYPE Y[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], // fghij  
                                             DATA_TYPE M1[TINY_LIMIT][TINY_LIMIT], // bf
                                             DATA_TYPE M2[TINY_LIMIT][TINY_LIMIT], // cg
                                             DATA_TYPE M3[TINY_LIMIT][TINY_LIMIT], // dh
                                             DATA_TYPE M4[TINY_LIMIT][TINY_LIMIT], // ei
                                             DATA_TYPE M5[TINY_LIMIT][TINY_LIMIT], // kj
                                             DATA_TYPE result[TINY_LIMIT][TINY_LIMIT]) { // ak
  int a, b, c, d, e, f, g, h, i, j, k;
  
  // Initialize output
  for (a = 0; a < A; a++)
    for (k = 0; k < K; k++)
      result[a][k] = 0;
  
  // Access pattern for tensor regression network contraction
  for (a = 0; a < A; a++) {
    for (k = 0; k < K; k++) {
      result[a][k] = 0; // Access output result[a][k]
      for (b = 0; b < B; b++) {
        M1[b][f] = 0; // Access M1[b][f] (f will be looped)
        for (c = 0; c < C; c++) {
          M2[c][g] = 0; // Access M2[c][g] (g will be looped)  
          for (d = 0; d < D; d++) {
            M3[d][h] = 0; // Access M3[d][h] (h will be looped)
            for (e = 0; e < E; e++) {
              X[a][b][c][d][e] = 0; // Access X[a][b][c][d][e]
              M4[e][i] = 0; // Access M4[e][i] (i will be looped)
              for (f = 0; f < F; f++) {
                for (g = 0; g < G; g++) {
                  for (h = 0; h < H; h++) {
                    for (i = 0; i < I; i++) {
                      for (j = 0; j < J; j++) {
                        Y[f][g][h][i][j] = 0; // Access Y[f][g][h][i][j]
                        M5[k][j] = 0; // Access M5[k][j]
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