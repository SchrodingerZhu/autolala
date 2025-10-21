#define DATA_TYPE double
#define N_SIZE 64



volatile DATA_TYPE A[N_SIZE][72];
volatile DATA_TYPE trace;
// Matrix trace: ii->
void kernel_matrix_trace() {
  int i;
  
  trace = 0.0;
  for (i = 0; i < N_SIZE; i++)
    trace += A[i][i];
}
