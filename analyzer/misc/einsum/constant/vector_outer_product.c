#define DATA_TYPE float
#define M_SIZE 32
#define N_SIZE 32

// Vector outer product: i,j->ij
void kernel_vector_outer_product(DATA_TYPE a[36], DATA_TYPE b[36], DATA_TYPE C[M_SIZE][36]) {  // M_SIZE=32 padded to 36, N_SIZE=32 padded to 36
  int i, j;
  
  for (i = 0; i < M_SIZE; i++)
    for (j = 0; j < N_SIZE; j++)
      C[i][j] = a[i] * b[j];
}
