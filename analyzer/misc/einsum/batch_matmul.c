#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

// Batch matrix multiplication: bik,bkj->bij
void kernel_batch_matmul(size_t B, size_t I, size_t K, size_t J, 
                        DATA_TYPE A[LIMIT][LIMIT][LIMIT], 
                        DATA_TYPE B_mat[LIMIT][LIMIT][LIMIT], 
                        DATA_TYPE C[LIMIT][LIMIT][LIMIT]) {
  int b, i, j, k;
  
  for (b = 0; b < B; b++)
    for (i = 0; i < I; i++)
      for (j = 0; j < J; j++) {
        C[b][i][j] = 0.0f;
        for (k = 0; k < K; k++)
          C[b][i][j] += A[b][i][k] * B_mat[b][k][j];
      }
}