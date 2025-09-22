#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define ALPHA 1.5f
#define BETA 1.2f

void kernel_gesummv(size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT], DATA_TYPE tmp[LIMIT], DATA_TYPE x[LIMIT], DATA_TYPE y[LIMIT]) {
  int i, j;

  for (i = 0; i < N; i++) {
    tmp[i] = 0.0f;
    y[i] = 0.0f;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}