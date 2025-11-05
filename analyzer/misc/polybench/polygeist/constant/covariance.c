// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/datamining/covariance/covariance.h
#define M 240
#define N 260
#define DATA_TYPE float


volatile DATA_TYPE data[263][248];  // N=260 padded to 263 (prime), M=240 padded to 248 (8×31)
volatile DATA_TYPE cov[241][248];  // M=240 padded to 241 (prime), M=240 padded to 248 (8×31)
volatile DATA_TYPE mean[241];  // M=240 padded to 241 (prime)

void kernel_covariance() {
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
