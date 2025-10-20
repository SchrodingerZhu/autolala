// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/syr2k/syr2k.h
#define M 200
#define N 240
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE C[N][240];  // N=240 already multiple of 12
volatile DATA_TYPE A[N][204];  // M=200 padded to 204
volatile DATA_TYPE B[N][204];  // M=200 padded to 204

void kernel_syr2k() {
  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j <= i; j++)
      C[i][j] *= BETA;
  for (i = 0; i < N; i++)
    for (j = 0; j <= i; j++)
      for (k = 0; k < M; k++) {
        C[i][j] += ALPHA * A[j][k] * B[i][k];
        C[i][j] += ALPHA * B[j][k] * A[i][k];
      }
}
