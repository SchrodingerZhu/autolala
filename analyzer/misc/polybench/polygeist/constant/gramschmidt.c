#include <math.h>

#define M 500
#define N 600
#define DATA_TYPE float

void kernel_gramschmidt(DATA_TYPE A[M][N], DATA_TYPE R[N][N], DATA_TYPE Q[M][N]) {
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