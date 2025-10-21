#define DATA_TYPE double
#define N_SIZE 64



volatile DATA_TYPE A[N_SIZE][72];
volatile DATA_TYPE diag[72];
// Matrix diagonal: ii->i  
void kernel_matrix_diagonal() {
  int i;
  
  for (i = 0; i < N_SIZE; i++)
    diag[i] = A[i][i];
}
