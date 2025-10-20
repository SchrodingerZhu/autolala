// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/symm/symm.h
#define M 200
#define N 240
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE C[M][240];  // N=240 already multiple of 12
volatile DATA_TYPE A[M][204];  // M=200 padded to 204
volatile DATA_TYPE B[M][240];  // N=240 already multiple of 12

void kernel_symm() {
  int i, j, k;
  DATA_TYPE temp2;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      temp2 = 0;
      for (k = 0; k < i; k++) {
        C[k][j] += ALPHA * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = BETA * C[i][j] + ALPHA * B[i][j] * A[i][i] + ALPHA * temp2;
    }
}
