// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/seidel-2d/seidel-2d.h
#define TSTEPS 100
#define N 400
#define DATA_TYPE float


volatile DATA_TYPE A[N][408];  // N=400 padded to 408

void kernel_seidel_2d() {
  int t, i, j;

  for (t = 0; t <= TSTEPS - 1; t++)
    for (i = 1; i <= N - 2; i++)
      for (j = 1; j <= N - 2; j++)
        A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                  + A[i][j-1] + A[i][j] + A[i][j+1]
                  + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0f;
}
