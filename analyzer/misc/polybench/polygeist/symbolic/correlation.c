#include <math.h>

#define DATA_TYPE float
#define LIMIT 1024
#define EPS 0.005f
typedef __SIZE_TYPE__ size_t;

void kernel_correlation(size_t M, size_t N, DATA_TYPE data[LIMIT][LIMIT], DATA_TYPE corr[LIMIT][LIMIT], DATA_TYPE mean[LIMIT], DATA_TYPE stddev[LIMIT]) {
  int i, j, k;
  DATA_TYPE float_n = (DATA_TYPE)N;

  for (j = 0; j < M; j++) {
    mean[j] = 0.0f;
    for (i = 0; i < N; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  for (j = 0; j < M; j++) {
    stddev[j] = 0.0f;
    for (i = 0; i < N; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = sqrtf(stddev[j]);
    stddev[j] = stddev[j] <= EPS ? 1.0f : stddev[j];
  }

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrtf(float_n) * stddev[j];
    }

  for (i = 0; i < M-1; i++) {
    corr[i][i] = 1.0f;
    for (j = i+1; j < M; j++) {
      corr[i][j] = 0.0f;
      for (k = 0; k < N; k++)
        corr[i][j] += (data[k][i] * data[k][j]);
      corr[j][i] = corr[i][j];
    }
  }
  corr[M-1][M-1] = 1.0f;
}