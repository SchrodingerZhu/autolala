#define TMAX 500
#define NX 1000
#define NY 1200
#define DATA_TYPE float

void kernel_fdtd_2d(DATA_TYPE ex[NX][NY], DATA_TYPE ey[NX][NY], DATA_TYPE hz[NX][NY], DATA_TYPE _fict_[TMAX]) {
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