// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/bicg/bicg.h
#define M 390
#define N 410
#define DATA_TYPE float


volatile DATA_TYPE A[397][424];  // M=390 padded to 397 (prime), N=410 padded to 424 (8×53)
volatile DATA_TYPE s[397];  // M=390 padded to 397 (prime)
volatile DATA_TYPE q[424];  // N=410 padded to 424 (8×53)
volatile DATA_TYPE p[397];  // M=390 padded to 397 (prime)
volatile DATA_TYPE r[424];  // N=410 padded to 424 (8×53)

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
