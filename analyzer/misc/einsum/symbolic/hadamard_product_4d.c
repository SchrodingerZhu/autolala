#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

// Hadamard product: ijkl,ijkl->ijkl
void kernel_hadamard_product_4d(size_t N1, size_t N2, size_t N3, size_t N4, 
                                DATA_TYPE A[LIMIT][LIMIT][LIMIT][LIMIT], 
                                DATA_TYPE B[LIMIT][LIMIT][LIMIT][LIMIT], 
                                DATA_TYPE C[LIMIT][LIMIT][LIMIT][LIMIT]) {
  int i, j, k, l;
  
  for (i = 0; i < N1; i++)
    for (j = 0; j < N2; j++)
      for (k = 0; k < N3; k++)
        for (l = 0; l < N4; l++)
          C[i][j][k][l] = A[i][j][k][l] * B[i][j][k][l];
}