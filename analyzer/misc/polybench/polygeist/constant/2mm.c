// Configuration from:
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/2mm/2mm.h
#define NI 180
#define NJ 190
#define NK 210
#define NL 220
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f

volatile DATA_TYPE
    A[181][232]; // NI=180 padded to 181 (prime), NK=210 padded to 232 (8×29)
volatile DATA_TYPE
    B[211][232]; // NK=210 padded to 211 (prime), NJ=190 padded to 232 (8×29)
volatile DATA_TYPE
    C[191][232]; // NJ=190 padded to 191 (prime), NL=220 padded to 232 (8×29)
volatile DATA_TYPE
    D[181][232]; // NI=180 padded to 181 (prime), NL=220 padded to 232 (8×29)
volatile DATA_TYPE
    tmp[181][232]; // NI=180 padded to 181 (prime), NJ=190 padded to 232 (8×29)

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
