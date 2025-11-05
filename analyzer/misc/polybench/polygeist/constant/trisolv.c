// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/trisolv/trisolv.h
#define N 400
#define DATA_TYPE float


volatile DATA_TYPE L[401][424];  // N=400 padded to 401 (prime) for first dim, N=400 padded to 424 (8×53)
volatile DATA_TYPE x[401];  // N=400 padded to 401 (prime)
volatile DATA_TYPE b[401];  // N=400 padded to 401 (prime)

void kernel_trisolv() {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = b[i];
    for (j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
