// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/mvt/mvt.h
#define N 400
#define DATA_TYPE float


volatile DATA_TYPE x1[401];  // N=400 padded to 401 (prime)
volatile DATA_TYPE x2[401];  // N=400 padded to 401 (prime)
volatile DATA_TYPE y_1[424];  // N=400 padded to 424 (8×53)
volatile DATA_TYPE y_2[424];  // N=400 padded to 424 (8×53)
volatile DATA_TYPE A[401][424];  // N=400 padded to 401 (prime) for first dim, N=400 padded to 424 (8×53)

void kernel_mvt() {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
}
