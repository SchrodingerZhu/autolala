#define N 4000
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE A[N][N];
volatile DATA_TYPE u1[N];
volatile DATA_TYPE v1[N];
volatile DATA_TYPE u2[N];
volatile DATA_TYPE v2[N];
volatile DATA_TYPE w[N];
volatile DATA_TYPE x[N];
volatile DATA_TYPE y[N];
volatile DATA_TYPE z[N];

void kernel_gemver() {
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
