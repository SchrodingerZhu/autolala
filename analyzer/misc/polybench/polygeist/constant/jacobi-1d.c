#define TSTEPS 500
#define N 4000
#define DATA_TYPE float


volatile DATA_TYPE A[N];
volatile DATA_TYPE B[N];

void kernel_jacobi_1d() {
  int t, i;

  for (t = 0; t < TSTEPS; t++) {
    for (i = 1; i < N - 1; i++)
      B[i] = 0.33333f * (A[i-1] + A[i] + A[i + 1]);
    for (i = 1; i < N - 1; i++)
      A[i] = 0.33333f * (B[i-1] + B[i] + B[i + 1]);
  }
}
