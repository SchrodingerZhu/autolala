// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/kernels/doitgen/doitgen.h
#define R 150
#define Q 140
#define P 160
#define DATA_TYPE float


volatile DATA_TYPE A[R][Q][168];  // P=160 padded to 168
volatile DATA_TYPE C4[P][168];  // P=160 padded to 168
volatile DATA_TYPE sum[168];  // P=160 padded to 168

void kernel_doitgen() {
  int r, q, p, s;

  for (r = 0; r < R; r++)
    for (q = 0; q < Q; q++) {
      for (p = 0; p < P; p++) {
        sum[p] = 0.0f;
        for (s = 0; s < P; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < P; p++)
        A[r][q][p] = sum[p];
    }
}
