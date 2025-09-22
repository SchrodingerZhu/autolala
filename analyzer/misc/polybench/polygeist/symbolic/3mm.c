#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

void kernel_3mm(size_t NI, size_t NJ, size_t NK, size_t NL, size_t NM, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT], DATA_TYPE C[LIMIT][LIMIT], DATA_TYPE D[LIMIT][LIMIT], DATA_TYPE E[LIMIT][LIMIT], DATA_TYPE F[LIMIT][LIMIT], DATA_TYPE G[LIMIT][LIMIT]) {
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