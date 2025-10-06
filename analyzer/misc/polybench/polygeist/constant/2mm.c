#define NI 200
#define NJ 300
#define NK 400
#define NL 500
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE A[NI][NK];
volatile DATA_TYPE B[NK][NJ];
volatile DATA_TYPE C[NJ][NL];
volatile DATA_TYPE D[NI][NL];
volatile DATA_TYPE tmp[NI][NJ];

void kernel_2mm() {
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
