#define M 1000
#define N 1200
#define DATA_TYPE float
#define ALPHA 1.5f

void kernel_trmm(DATA_TYPE A[M][M], DATA_TYPE B[M][N]) {
  int i, j, k;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      for (k = i+1; k < M; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = ALPHA * B[i][j];
    }
}