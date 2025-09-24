#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

// Matrix trace: ii->
void kernel_matrix_trace(size_t N, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE *trace) {
  int i;
  
  *trace = 0.0f;
  for (i = 0; i < N; i++)
    *trace += A[i][i];
}