// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/trmm/trmm.h
#define M 200
#define N 240
#define DATA_TYPE float
#define ALPHA 1.5f


volatile DATA_TYPE A[M][M];
volatile DATA_TYPE B[M][N];

void kernel_trmm() {
  int i, j, k;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      for (k = i+1; k < M; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = ALPHA * B[i][j];
    }
}
