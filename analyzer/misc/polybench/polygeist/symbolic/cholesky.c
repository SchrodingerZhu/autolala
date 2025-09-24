#include <math.h>

#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

void kernel_cholesky(size_t N, DATA_TYPE A[LIMIT][LIMIT]) {
  int i, j, k;
  
  for (i = 0; i < N; i++) {
    for (j = 0; j < i; j++) {
      for (k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] /= A[j][j];
    }
    
    for (k = 0; k < i; k++)
      A[i][i] -= A[i][k] * A[i][k];
    A[i][i] = sqrtf(A[i][i]);
  }
}