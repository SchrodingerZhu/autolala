#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

void kernel_atax(size_t M, size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE x[LIMIT], DATA_TYPE y[LIMIT], DATA_TYPE tmp[LIMIT]) {
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