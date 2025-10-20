// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/datamining/correlation/correlation.h
#include <math.h>

#define M 240
#define N 260
#define DATA_TYPE float
#define EPS 0.005f


volatile DATA_TYPE data[N][240];  // M=240 already multiple of 12
volatile DATA_TYPE corr[M][240];  // M=240 already multiple of 12
volatile DATA_TYPE mean[M];
volatile DATA_TYPE stddev[M];

void kernel_correlation() {
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
