#define M 400
#define N 500
#define DATA_TYPE float

void kernel_atax(DATA_TYPE A[M][N], DATA_TYPE x[N], DATA_TYPE y[N], DATA_TYPE tmp[M]) {
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