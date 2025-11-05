// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/gemm.h
#define NI 200
#define NJ 220
#define NK 240
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE C[211][232];  // NI=200 padded to 211 (prime), NJ=220 padded to 232 (8×29)
volatile DATA_TYPE A[211][248];  // NI=200 padded to 211 (prime), NK=240 padded to 248 (8×31)
volatile DATA_TYPE B[241][232];  // NK=240 padded to 241 (prime), NJ=220 padded to 232 (8×29)

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
