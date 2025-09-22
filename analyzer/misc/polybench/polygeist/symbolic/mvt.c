#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

void kernel_mvt(size_t N, DATA_TYPE x1[LIMIT], DATA_TYPE x2[LIMIT], DATA_TYPE y_1[LIMIT], DATA_TYPE y_2[LIMIT], DATA_TYPE A[LIMIT][LIMIT]) {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
}