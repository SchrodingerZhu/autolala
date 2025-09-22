#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

void kernel_doitgen(size_t R, size_t Q, size_t P, DATA_TYPE A[LIMIT][LIMIT][LIMIT], DATA_TYPE C4[LIMIT][LIMIT], DATA_TYPE sum[LIMIT]) {
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