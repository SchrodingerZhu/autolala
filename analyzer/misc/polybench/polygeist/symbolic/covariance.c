#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

void kernel_covariance(size_t M, size_t N, DATA_TYPE data[LIMIT][LIMIT], DATA_TYPE cov[LIMIT][LIMIT], DATA_TYPE mean[LIMIT]) {
  int i, j, k;
  DATA_TYPE float_n = (DATA_TYPE)N;

  for (j = 0; j < M; j++) {
    mean[j] = 0.0f;
    for (i = 0; i < N; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < M; i++)
    for (j = i; j < M; j++) {
      cov[i][j] = 0.0f;
      for (k = 0; k < N; k++)
        cov[i][j] += data[k][i] * data[k][j];
      cov[i][j] /= (float_n - 1.0f);
      cov[j][i] = cov[i][j];
    }
}