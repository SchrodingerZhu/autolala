#define N 2000
#define DATA_TYPE float

void kernel_trisolv(DATA_TYPE L[N][N], DATA_TYPE x[N], DATA_TYPE b[N]) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = b[i];
    for (j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}