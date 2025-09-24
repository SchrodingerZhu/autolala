#define DATA_TYPE float
#define LIMIT 1024
#define ALPHA 1.5f
#define BETA 1.2f
typedef __SIZE_TYPE__ size_t;

void kernel_2mm(size_t NI, size_t NJ, size_t NK, size_t NL, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT], DATA_TYPE C[LIMIT][LIMIT], DATA_TYPE D[LIMIT][LIMIT], DATA_TYPE tmp[LIMIT][LIMIT]) {
  int i, j, k;
  
  /* tmp := alpha * A * B */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      tmp[i][j] = 0.0f;
      for (k = 0; k < NK; ++k)
        tmp[i][j] += ALPHA * A[i][k] * B[k][j];
    }
  
  /* D := beta * tmp * C + D */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      D[i][j] *= BETA;
      for (k = 0; k < NJ; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
}