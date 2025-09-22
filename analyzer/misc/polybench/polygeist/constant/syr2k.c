#define M 1000
#define N 1200
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f

void kernel_syr2k(DATA_TYPE C[N][N], DATA_TYPE A[N][M], DATA_TYPE B[N][M]) {
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