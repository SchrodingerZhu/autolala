#define DATA_TYPE double
#define M_SIZE 64
#define N_SIZE 64



volatile DATA_TYPE A[M_SIZE][N_SIZE];
volatile DATA_TYPE B[M_SIZE][N_SIZE];
volatile DATA_TYPE C[M_SIZE][N_SIZE];
// Hadamard product (2D): ij,ij->ij
void kernel_hadamard_product_2d() {
  int i, j;
  
  for (i = 0; i < M_SIZE; i++)
    for (j = 0; j < N_SIZE; j++)
      C[i][j] = A[i][j] * B[i][j];
}
