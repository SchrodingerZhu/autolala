// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/3mm/3mm.h
#define NI 180
#define NJ 190
#define NK 200
#define NL 210
#define NM 220
#define DATA_TYPE float


volatile DATA_TYPE A[NI][NK];
volatile DATA_TYPE B[NK][NJ];
volatile DATA_TYPE C[NJ][NM];
volatile DATA_TYPE D[NM][NL];
volatile DATA_TYPE E[NI][NJ];
volatile DATA_TYPE F[NJ][NL];
volatile DATA_TYPE G[NI][NL];

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
