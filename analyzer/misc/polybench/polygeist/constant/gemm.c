// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/gemm.h
#define NI 200
#define NJ 220
#define NK 240
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE C[NI][228];  // NJ=220 padded to 228
volatile DATA_TYPE A[NI][240];  // NK=240 already multiple of 12
volatile DATA_TYPE B[NK][228];  // NJ=220 padded to 228

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
