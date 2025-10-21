#define DATA_TYPE double
#define M_SIZE 64
#define N_SIZE 64



volatile DATA_TYPE a[72];
volatile DATA_TYPE b[72];
volatile DATA_TYPE C[M_SIZE][72];
// Vector outer product: i,j->ij
void kernel_vector_outer_product() {
  int i, j;
  
  for (i = 0; i < M_SIZE; i++)
    for (j = 0; j < N_SIZE; j++)
      C[i][j] = a[i] * b[j];
}
