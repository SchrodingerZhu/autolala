// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/adi/adi.h
#define TSTEPS 100
#define N 200
#define DATA_TYPE float


volatile DATA_TYPE u[211][232];  // First dim: N=200 padded to 211 (prime), Second dim: N=200 padded to 232 (8×29)
volatile DATA_TYPE v[211][232];  // First dim: N=200 padded to 211 (prime), Second dim: N=200 padded to 232 (8×29)
volatile DATA_TYPE p[211][232];  // First dim: N=200 padded to 211 (prime), Second dim: N=200 padded to 232 (8×29)
volatile DATA_TYPE q[211][232];  // First dim: N=200 padded to 211 (prime), Second dim: N=200 padded to 232 (8×29)

void kernel_adi() {
  int t, i, j;
  DATA_TYPE DX = 1.0f / (DATA_TYPE)N;
  DATA_TYPE DY = 1.0f / (DATA_TYPE)N;
  DATA_TYPE DT = 1.0f / (DATA_TYPE)TSTEPS;
  DATA_TYPE B1 = 2.0f;
  DATA_TYPE B2 = 1.0f;
  DATA_TYPE mul1 = B1 * DT / (DX * DX);
  DATA_TYPE mul2 = B2 * DT / (DY * DY);
  DATA_TYPE a = -mul1 / 2.0f;
  DATA_TYPE b = 1.0f + mul1;
  DATA_TYPE c = a;
  DATA_TYPE d = -mul2 / 2.0f;
  DATA_TYPE e = 1.0f + mul2;
  DATA_TYPE f = d;

  for (t = 1; t <= TSTEPS; t++) {
    /* Column Sweep */
    for (i = 1; i < N-1; i++)
      for (j = 1; j < N-1; j++)
        v[i][j] = -d * u[j][i-1] + e * u[j][i] - f * u[j][i+1];

    /* Row Sweep */
    for (i = 1; i < N-1; i++)
      for (j = 1; j < N-1; j++)
        u[i][j] = -a * v[i-1][j] + b * v[i][j] - c * v[i+1][j];
  }
}
