#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define ALPHA 1.5f
#define BETA 1.2f

void kernel_gemver(size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE u1[LIMIT], DATA_TYPE v1[LIMIT], DATA_TYPE u2[LIMIT], DATA_TYPE v2[LIMIT], DATA_TYPE w[LIMIT], DATA_TYPE x[LIMIT], DATA_TYPE y[LIMIT], DATA_TYPE z[LIMIT]) {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + BETA * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] + ALPHA * A[i][j] * x[j];
}