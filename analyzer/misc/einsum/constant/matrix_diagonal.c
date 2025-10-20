#define DATA_TYPE float
#define N_SIZE 64

// Matrix diagonal: ii->i  
void kernel_matrix_diagonal(DATA_TYPE A[N_SIZE][72], DATA_TYPE diag[72]) {  // N_SIZE=64 padded to 72
  int i;
  
  for (i = 0; i < N_SIZE; i++)
    diag[i] = A[i][i];
}
