#define DATA_TYPE float
#define M_SIZE 64
#define N_SIZE 64

// Hadamard product (2D): ij,ij->ij
void kernel_hadamard_product_2d(DATA_TYPE A[M_SIZE][N_SIZE], 
                               DATA_TYPE B[M_SIZE][N_SIZE], 
                               DATA_TYPE C[M_SIZE][N_SIZE]) {
  int i, j;
  
  for (i = 0; i < M_SIZE; i++)
    for (j = 0; j < N_SIZE; j++)
      C[i][j] = A[i][j] * B[i][j];
}
