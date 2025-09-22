#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

// Matrix diagonal: ii->i  
void kernel_matrix_diagonal(size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE diag[LIMIT]) {
  int i;
  
  for (i = 0; i < N; i++)
    diag[i] = A[i][i];
}