#define DATA_TYPE float
#define B_SIZE 16
#define I_SIZE 32
#define K_SIZE 32 
#define J_SIZE 32

// Batch matrix multiplication: bik,bkj->bij
void kernel_batch_matmul(DATA_TYPE A[B_SIZE][I_SIZE][K_SIZE], 
                        DATA_TYPE B_mat[B_SIZE][K_SIZE][J_SIZE], 
                        DATA_TYPE C[B_SIZE][I_SIZE][J_SIZE]) {
  int b, i, j, k;
  
  for (b = 0; b < B_SIZE; b++)
    for (i = 0; i < I_SIZE; i++)
      for (j = 0; j < J_SIZE; j++) {
        C[b][i][j] = 0.0f;
        for (k = 0; k < K_SIZE; k++)
          C[b][i][j] += A[b][i][k] * B_mat[b][k][j];
      }
}
