// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/syr2k/syr2k.h
#define M 200
#define N 240
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE C[241][248];  // N=240 padded to 241 (prime), N=240 padded to 248 (8×31)
volatile DATA_TYPE A[241][232];  // N=240 padded to 241 (prime), M=200 padded to 232 (8×29)
volatile DATA_TYPE B[241][232];  // N=240 padded to 241 (prime), M=200 padded to 232 (8×29)

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
