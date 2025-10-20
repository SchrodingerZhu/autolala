#define DATA_TYPE float
#define N1_SIZE 16
#define N2_SIZE 16
#define N3_SIZE 16
#define N4_SIZE 16

// Hadamard product: ijkl,ijkl->ijkl
void kernel_hadamard_product_4d(DATA_TYPE A[N1_SIZE][N2_SIZE][N3_SIZE][24],  // N4_SIZE=16 padded to 24
                                DATA_TYPE B[N1_SIZE][N2_SIZE][N3_SIZE][24],  // N4_SIZE=16 padded to 24
                                DATA_TYPE C[N1_SIZE][N2_SIZE][N3_SIZE][24]) {  // N4_SIZE=16 padded to 24
  int i, j, k, l;
  
  for (i = 0; i < N1_SIZE; i++)
    for (j = 0; j < N2_SIZE; j++)
      for (k = 0; k < N3_SIZE; k++)
        for (l = 0; l < N4_SIZE; l++)
          C[i][j][k][l] = A[i][j][k][l] * B[i][j][k][l];
}
