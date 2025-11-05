// Configuration from:
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/3mm/3mm.h
#define NI 180
#define NJ 190
#define NK 200
#define NL 210
#define NM 220
#define DATA_TYPE float

volatile DATA_TYPE
    A[181][232]; // NI=180 padded to 181 (prime), NK=200 padded to 232 (8×29)
volatile DATA_TYPE
    B[211][232]; // NK=200 padded to 211 (prime), NJ=190 padded to 232 (8×29)
volatile DATA_TYPE
    C[191][232]; // NJ=190 padded to 191 (prime), NM=220 padded to 232 (8×29)
volatile DATA_TYPE
    D[223][232]; // NM=220 padded to 223 (prime), NL=210 padded to 232 (8×29)
volatile DATA_TYPE
    E[181][232]; // NI=180 padded to 181 (prime), NJ=190 padded to 232 (8×29)
volatile DATA_TYPE
    F[191][232]; // NJ=190 padded to 191 (prime), NL=210 padded to 232 (8×29)
volatile DATA_TYPE
    G[181][232]; // NI=180 padded to 181 (prime), NL=210 padded to 232 (8×29)

void kernel_3mm() {
  int i, j, k;

  /* E := A*B */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      E[i][j] = 0.0f;
      for (k = 0; k < NK; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }

  /* F := C*D */
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++) {
      F[i][j] = 0.0f;
      for (k = 0; k < NM; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }

  /* G := E*F */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      G[i][j] = 0.0f;
      for (k = 0; k < NJ; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
}
