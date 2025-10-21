// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/bicg/bicg.h
#define M 390
#define N 410
#define DATA_TYPE float


volatile DATA_TYPE A[M][420];  // N=410 padded to 420
volatile DATA_TYPE s[408];
volatile DATA_TYPE q[420];  // N=410 padded to 420
volatile DATA_TYPE p[408];
volatile DATA_TYPE r[420];  // N=410 padded to 420

void kernel_bicg() {
  int i, j;

  for (i = 0; i < M; i++)
    s[i] = 0;

  for (i = 0; i < M; i++) {
    q[i] = 0.0f;
    for (j = 0; j < N; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }
}
