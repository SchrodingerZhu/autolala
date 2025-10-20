// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/trisolv/trisolv.h
#define N 400
#define DATA_TYPE float


volatile DATA_TYPE L[N][408];  // N=400 padded to 408
volatile DATA_TYPE x[408];  // N=400 padded to 408
volatile DATA_TYPE b[408];  // N=400 padded to 408

void kernel_trisolv() {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = b[i];
    for (j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
