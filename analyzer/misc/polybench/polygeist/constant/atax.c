// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/atax/atax.h
#define M 390
#define N 410
#define DATA_TYPE float


volatile DATA_TYPE A[397][424];  // M=390 padded to 397 (prime), N=410 padded to 424 (8×53)
volatile DATA_TYPE x[424];  // N=410 padded to 424 (8×53)
volatile DATA_TYPE y[424];  // N=410 padded to 424 (8×53)
volatile DATA_TYPE tmp[397];  // M=390 padded to 397 (prime)

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
