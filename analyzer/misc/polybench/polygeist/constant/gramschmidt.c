// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/gramschmidt/gramschmidt.h
#include <math.h>

#define M 200
#define N 240
#define DATA_TYPE float


volatile DATA_TYPE A[211][248];  // M=200 padded to 211 (prime), N=240 padded to 248 (8×31)
volatile DATA_TYPE R[241][248];  // N=240 padded to 241 (prime), N=240 padded to 248 (8×31)
volatile DATA_TYPE Q[211][248];  // M=200 padded to 211 (prime), N=240 padded to 248 (8×31)

void kernel_gramschmidt() {
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
