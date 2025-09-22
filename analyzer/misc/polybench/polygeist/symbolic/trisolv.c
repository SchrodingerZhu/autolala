#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

void kernel_trisolv(size_t N, DATA_TYPE L[LIMIT][LIMIT], DATA_TYPE x[LIMIT], DATA_TYPE b[LIMIT]) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = b[i];
    for (j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}