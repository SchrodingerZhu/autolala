// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp.h
#define N 400
#define DATA_TYPE float


volatile DATA_TYPE A[N][N];
volatile DATA_TYPE b[N];
volatile DATA_TYPE x[N];
volatile DATA_TYPE y[N];

void kernel_ludcmp() {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j <i; j++) {
      for (k = 0; k < j; k++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
      A[i][j] /= A[j][j];
    }
    for (j = i; j < N; j++) {
      for (k = 0; k < i; k++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
    }
  }

  for (i = 0; i < N; i++) {
    y[i] = b[i];
    for (j = 0; j <i; j++) {
      y[i] -= A[i][j] * y[j];
    }
  }
  for (i = N - 1; i >= 0; i--) {
    x[i] = y[i];
    for (j = i + 1; j < N; j++) {
      x[i] -= A[i][j] * x[j];
    }
    x[i] = x[i] / A[i][i];
  }
}
