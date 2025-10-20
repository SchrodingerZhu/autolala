// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/jacobi-2d/jacobi-2d.h
#define TSTEPS 100
#define N 250
#define DATA_TYPE float


volatile DATA_TYPE A[N][252];  // N=250 padded to 252
volatile DATA_TYPE B[N][252];  // N=250 padded to 252

void kernel_jacobi_2d() {
  int t, i, j;

  for (t = 0; t < TSTEPS; t++) {
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        B[i][j] = 0.2f * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        A[i][j] = 0.2f * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
  }
}
