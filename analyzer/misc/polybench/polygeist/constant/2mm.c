// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/2mm/2mm.h
#define NI 180
#define NJ 190
#define NK 210
#define NL 220
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE A[NI][216];  // NK=210 padded to 216
volatile DATA_TYPE B[NK][192];  // NJ=190 padded to 192
volatile DATA_TYPE C[NJ][228];  // NL=220 padded to 228
volatile DATA_TYPE D[NI][228];  // NL=220 padded to 228
volatile DATA_TYPE tmp[NI][192];  // NJ=190 padded to 192

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
