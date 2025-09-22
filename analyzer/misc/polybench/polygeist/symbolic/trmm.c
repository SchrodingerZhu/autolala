#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define ALPHA 1.5f

void kernel_trmm(size_t M, size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT]) {
  int i, j, k;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      for (k = i+1; k < M; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = ALPHA * B[i][j];
    }
}