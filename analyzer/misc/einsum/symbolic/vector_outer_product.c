#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

// Vector outer product: i,j->ij
void kernel_vector_outer_product(size_t M, size_t N, DATA_TYPE a[LIMIT], DATA_TYPE b[LIMIT], DATA_TYPE C[LIMIT][LIMIT]) {
  int i, j;
  
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      C[i][j] = a[i] * b[j];
}