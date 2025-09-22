#define N 4000
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f

void kernel_gesummv(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE tmp[N], DATA_TYPE x[N], DATA_TYPE y[N]) {
  int i, j;

  for (i = 0; i < N; i++) {
    tmp[i] = 0.0f;
    y[i] = 0.0f;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}