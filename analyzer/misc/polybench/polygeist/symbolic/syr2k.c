#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define ALPHA 1.5f
#define BETA 1.2f

void kernel_syr2k(size_t M, size_t N, DATA_TYPE C[LIMIT][LIMIT], DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT]) {
  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j <= i; j++)
      C[i][j] *= BETA;
  for (i = 0; i < N; i++)
    for (j = 0; j <= i; j++)
      for (k = 0; k < M; k++) {
        C[i][j] += ALPHA * A[j][k] * B[i][k];
        C[i][j] += ALPHA * B[j][k] * A[i][k];
      }
}