#define DATA_TYPE float
#define LIMIT 1024
#define ALPHA 1.5f
#define BETA 1.2f
typedef __SIZE_TYPE__ size_t;

void kernel_symm(size_t M, size_t N, DATA_TYPE C[LIMIT][LIMIT], DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT]) {
  int i, j, k;
  DATA_TYPE temp2;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      temp2 = 0;
      for (k = 0; k < i; k++) {
        C[k][j] += ALPHA * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = BETA * C[i][j] + ALPHA * B[i][j] * A[i][i] + ALPHA * temp2;
    }
}