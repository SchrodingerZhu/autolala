// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/fdtd-2d/fdtd-2d.h
#define TMAX 100
#define NX 200
#define NY 240
#define DATA_TYPE float


volatile DATA_TYPE ex[211][248];  // NX=200 padded to 211 (prime), NY=240 padded to 248 (8×31)
volatile DATA_TYPE ey[211][248];  // NX=200 padded to 211 (prime), NY=240 padded to 248 (8×31)
volatile DATA_TYPE hz[211][248];  // NX=200 padded to 211 (prime), NY=240 padded to 248 (8×31)
volatile DATA_TYPE _fict_[120];

void kernel_fdtd_2d() {
  int t, i, j;

  for (t = 0; t < TMAX; t++) {
    for (j = 0; j < NY; j++)
      ey[0][j] = _fict_[t];
    for (i = 1; i < NX; i++)
      for (j = 0; j < NY; j++)
        ey[i][j] = ey[i][j] - 0.5f * (hz[i][j] - hz[i-1][j]);
    for (i = 0; i < NX; i++)
      for (j = 1; j < NY; j++)
        ex[i][j] = ex[i][j] - 0.5f * (hz[i][j] - hz[i][j-1]);
    for (i = 0; i < NX - 1; i++)
      for (j = 0; j < NY - 1; j++)
        hz[i][j] = hz[i][j] - 0.7f * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
  }
}
