#define M 400
#define N 500
#define DATA_TYPE float


volatile DATA_TYPE A[M][N];
volatile DATA_TYPE x[N];
volatile DATA_TYPE y[N];
volatile DATA_TYPE tmp[M];

void kernel_atax() {
  int i, j;

  for (i = 0; i < N; i++)
    y[i] = 0;

  for (i = 0; i < M; i++) {
    tmp[i] = 0.0f;
    for (j = 0; j < N; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (j = 0; j < N; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
}
