#include <math.h>

#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

void kernel_gramschmidt(size_t M, size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE R[LIMIT][LIMIT], DATA_TYPE Q[LIMIT][LIMIT]) {
  int i, j, k;
  DATA_TYPE nrm;

  for (k = 0; k < N; k++) {
    nrm = 0.0f;
    for (i = 0; i < M; i++)
      nrm += A[i][k] * A[i][k];
    R[k][k] = sqrtf(nrm);
    for (i = 0; i < M; i++)
      Q[i][k] = A[i][k] / R[k][k];
    for (j = k + 1; j < N; j++) {
      R[k][j] = 0.0f;
      for (i = 0; i < M; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (i = 0; i < M; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
}