#define N 2000
#define DATA_TYPE float


volatile DATA_TYPE A[N][N];

void kernel_lu() {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j <i; j++) {
      for (k = 0; k < j; k++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
      A[i][j] /= A[j][j];
    }
    for (j = i; j < N; j++) {
      for (k = 0; k < i; k++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
    }
  }
}
