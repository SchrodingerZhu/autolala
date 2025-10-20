#define DATA_TYPE float
#define N_SIZE 64

// Matrix trace: ii->
void kernel_matrix_trace(DATA_TYPE A[N_SIZE][72], DATA_TYPE *trace) {  // N_SIZE=64 padded to 72
  int i;
  
  *trace = 0.0f;
  for (i = 0; i < N_SIZE; i++)
    *trace += A[i][i];
}
