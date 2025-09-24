#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

void kernel_bicg(size_t M, size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE s[LIMIT], DATA_TYPE q[LIMIT], DATA_TYPE p[LIMIT], DATA_TYPE r[LIMIT]) {
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