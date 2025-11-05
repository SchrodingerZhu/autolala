// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/jacobi-1d/jacobi-1d.h
#define TSTEPS 100
#define N 400
#define DATA_TYPE float


volatile DATA_TYPE A[401];  // N=400 padded to 401 (prime)
volatile DATA_TYPE B[401];  // N=400 padded to 401 (prime)

void kernel_jacobi_1d() {
  int t, i;

  for (t = 0; t < TSTEPS; t++) {
    for (i = 1; i < N - 1; i++)
      B[i] = 0.33333f * (A[i-1] + A[i] + A[i + 1]);
    for (i = 1; i < N - 1; i++)
      A[i] = 0.33333f * (B[i-1] + B[i] + B[i + 1]);
  }
}
