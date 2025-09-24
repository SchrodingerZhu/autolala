#define DATA_TYPE float
#define LIMIT 1024
#define ALPHA 1.5f
#define BETA 1.2f
typedef __SIZE_TYPE__ size_t;

void kernel_gemm(size_t NI, size_t NJ, size_t NK, DATA_TYPE C[LIMIT][LIMIT], DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT]) {
  int i, j, k;
  
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++)
      C[i][j] *= BETA;
    for (k = 0; k < NK; k++) {
      for (j = 0; j < NJ; j++)
        C[i][j] += ALPHA * A[i][k] * B[k][j];
    }
  }
}