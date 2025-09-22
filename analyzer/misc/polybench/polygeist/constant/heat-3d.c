#define TSTEPS 500
#define N 120
#define DATA_TYPE float

void kernel_heat_3d(DATA_TYPE A[N][N][N], DATA_TYPE B[N][N][N]) {
  int t, i, j, k;
  
  for (t = 1; t <= TSTEPS; t++) {
    for (i = 1; i < N-1; i++) {
      for (j = 1; j < N-1; j++) {
        for (k = 1; k < N-1; k++) {
          B[i][j][k] = 0.125f * (A[i+1][j][k] - 2.0f * A[i][j][k] + A[i-1][j][k])
                      + 0.125f * (A[i][j+1][k] - 2.0f * A[i][j][k] + A[i][j-1][k])
                      + 0.125f * (A[i][j][k+1] - 2.0f * A[i][j][k] + A[i][j][k-1])
                      + A[i][j][k];
        }
      }
    }
    for (i = 1; i < N-1; i++) {
      for (j = 1; j < N-1; j++) {
        for (k = 1; k < N-1; k++) {
          A[i][j][k] = 0.125f * (B[i+1][j][k] - 2.0f * B[i][j][k] + B[i-1][j][k])
                      + 0.125f * (B[i][j+1][k] - 2.0f * B[i][j][k] + B[i][j-1][k])
                      + 0.125f * (B[i][j][k+1] - 2.0f * B[i][j][k] + B[i][j][k-1])
                      + B[i][j][k];
        }
      }
    }
  }
}