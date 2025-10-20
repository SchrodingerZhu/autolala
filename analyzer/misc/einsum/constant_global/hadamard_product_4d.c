#define DATA_TYPE double
#define N1_SIZE 64
#define N2_SIZE 64
#define N3_SIZE 64
#define N4_SIZE 64



volatile DATA_TYPE A[N1_SIZE][N2_SIZE][N3_SIZE][N4_SIZE];
volatile DATA_TYPE B[N1_SIZE][N2_SIZE][N3_SIZE][N4_SIZE];
volatile DATA_TYPE C[N1_SIZE][N2_SIZE][N3_SIZE][N4_SIZE];
// Hadamard product: ijkl,ijkl->ijkl
void kernel_hadamard_product_4d() {
  int i, j, k, l;
  
  for (i = 0; i < N1_SIZE; i++)
    for (j = 0; j < N2_SIZE; j++)
      for (k = 0; k < N3_SIZE; k++)
        for (l = 0; l < N4_SIZE; l++)
          C[i][j][k][l] = A[i][j][k][l] * B[i][j][k][l];
}
