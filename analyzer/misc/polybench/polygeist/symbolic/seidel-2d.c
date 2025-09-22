#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

void kernel_seidel_2d(size_t TSTEPS, size_t N, DATA_TYPE A[LIMIT][LIMIT]) {
  int t, i, j;

  for (t = 0; t <= TSTEPS - 1; t++)
    for (i = 1; i <= N - 2; i++)
      for (j = 1; j <= N - 2; j++)
        A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                  + A[i][j-1] + A[i][j] + A[i][j+1]
                  + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0f;
}