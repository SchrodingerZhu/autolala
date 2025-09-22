#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

void kernel_jacobi_2d(size_t TSTEPS, size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT]) {
  int t, i, j;
  
  for (t = 0; t < TSTEPS; t++) {
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        B[i][j] = 0.2f * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        A[i][j] = 0.2f * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
  }
}