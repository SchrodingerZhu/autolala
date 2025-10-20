// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemver/gemver.h
#define N 400
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE A[N][408];  // N=400 padded to 408
volatile DATA_TYPE u1[408];  // N=400 padded to 408
volatile DATA_TYPE v1[408];  // N=400 padded to 408
volatile DATA_TYPE u2[408];  // N=400 padded to 408
volatile DATA_TYPE v2[408];  // N=400 padded to 408
volatile DATA_TYPE w[408];  // N=400 padded to 408
volatile DATA_TYPE x[408];  // N=400 padded to 408
volatile DATA_TYPE y[408];  // N=400 padded to 408
volatile DATA_TYPE z[408];  // N=400 padded to 408

void kernel_gemver() {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + BETA * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] + ALPHA * A[i][j] * x[j];
}
