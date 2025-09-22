#define M 390
#define N 410
#define DATA_TYPE float

void kernel_bicg(DATA_TYPE A[M][N], DATA_TYPE s[M], DATA_TYPE q[N], DATA_TYPE p[M], DATA_TYPE r[N]) {
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