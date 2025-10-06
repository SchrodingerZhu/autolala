#define N 2000
#define DATA_TYPE float


volatile DATA_TYPE L[N][N];
volatile DATA_TYPE x[N];
volatile DATA_TYPE b[N];

void kernel_trisolv() {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = b[i];
    for (j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
