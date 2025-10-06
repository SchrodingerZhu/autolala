#define NI 1000
#define NJ 1100
#define NK 900
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE C[NI][NJ];
volatile DATA_TYPE A[NI][NK];
volatile DATA_TYPE B[NK][NJ];

void kernel_gemm() {
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
