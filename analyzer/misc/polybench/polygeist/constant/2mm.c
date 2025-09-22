#define NI 200
#define NJ 300
#define NK 400
#define NL 500
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f

void kernel_2mm(DATA_TYPE A[NI][NK], DATA_TYPE B[NK][NJ], DATA_TYPE C[NJ][NL], DATA_TYPE D[NI][NL], DATA_TYPE tmp[NI][NJ]) {
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