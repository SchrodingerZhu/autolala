// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/atax/atax.h
#define M 390
#define N 410
#define DATA_TYPE float


volatile DATA_TYPE A[M][420];  // N=410 padded to 420
volatile DATA_TYPE x[420];  // N=410 padded to 420
volatile DATA_TYPE y[420];  // N=410 padded to 420
volatile DATA_TYPE tmp[M];

void kernel_atax() {
  int i, j;

  for (i = 0; i < N; i++)
    y[i] = 0;

  for (i = 0; i < M; i++) {
    tmp[i] = 0.0f;
    for (j = 0; j < N; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (j = 0; j < N; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
}
