// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/trmm/trmm.h
#define M 200
#define N 240
#define DATA_TYPE float
#define ALPHA 1.5f


volatile DATA_TYPE A[211][232];  // M=200 padded to 211 (prime), M=200 padded to 232 (8×29)
volatile DATA_TYPE B[211][248];  // M=200 padded to 211 (prime), N=240 padded to 248 (8×31)

void kernel_trmm() {
  int i, j, k;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      for (k = i+1; k < M; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = ALPHA * B[i][j];
    }
}
