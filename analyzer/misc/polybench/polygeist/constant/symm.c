#define M 1000
#define N 1200
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE C[M][N];
volatile DATA_TYPE A[M][M];
volatile DATA_TYPE B[M][N];

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
