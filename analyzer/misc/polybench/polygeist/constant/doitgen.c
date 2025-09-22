#define R 150
#define Q 140
#define P 160
#define DATA_TYPE float

void kernel_doitgen(DATA_TYPE A[R][Q][P], DATA_TYPE C4[P][P], DATA_TYPE sum[P]) {
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