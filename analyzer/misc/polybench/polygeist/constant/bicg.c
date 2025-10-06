#define M 390
#define N 410
#define DATA_TYPE float


volatile DATA_TYPE A[M][N];
volatile DATA_TYPE s[M];
volatile DATA_TYPE q[N];
volatile DATA_TYPE p[M];
volatile DATA_TYPE r[N];

void kernel_bicg() {
  int i, j;

  for (i = 0; i < M; i++)
    s[i] = 0;

  for (i = 0; i < M; i++) {
    q[i] = 0.0f;
    for (j = 0; j < N; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }
}
