#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

// Hadamard product (2D): ij,ij->ij
void kernel_hadamard_product_2d(size_t M, size_t N, 
                               DATA_TYPE A[LIMIT][LIMIT], 
                               DATA_TYPE B[LIMIT][LIMIT], 
                               DATA_TYPE C[LIMIT][LIMIT]) {
  int i, j;
  
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      C[i][j] = A[i][j] * B[i][j];
}