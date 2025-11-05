// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/cholesky/cholesky.h
#include <math.h>

#define N 400
#define DATA_TYPE float


volatile DATA_TYPE A[401][424];  // N=400 padded to 401 (prime) for first dim, N=400 padded to 424 (8×53)

void kernel_cholesky() {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < i; j++) {
      for (k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] /= A[j][j];
    }

    for (k = 0; k < i; k++)
      A[i][i] -= A[i][k] * A[i][k];
    A[i][i] = sqrtf(A[i][i]);
  }
}
