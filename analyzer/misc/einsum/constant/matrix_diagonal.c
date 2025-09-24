#define DATA_TYPE float
#define N_SIZE 64

// Matrix diagonal: ii->i  
void kernel_matrix_diagonal(DATA_TYPE A[N_SIZE][N_SIZE], DATA_TYPE diag[N_SIZE]) {
  int i;
  
  for (i = 0; i < N_SIZE; i++)
    diag[i] = A[i][i];
}
