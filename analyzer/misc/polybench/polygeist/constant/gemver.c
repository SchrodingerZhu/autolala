#define N 4000
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f

void kernel_gemver(DATA_TYPE A[N][N], DATA_TYPE u1[N], DATA_TYPE v1[N], DATA_TYPE u2[N], DATA_TYPE v2[N], DATA_TYPE w[N], DATA_TYPE x[N], DATA_TYPE y[N], DATA_TYPE z[N]) {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + BETA * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] + ALPHA * A[i][j] * x[j];
}