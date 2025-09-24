#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

void kernel_jacobi_1d(size_t TSTEPS, size_t N, DATA_TYPE A[LIMIT], DATA_TYPE B[LIMIT]) {
  int t, i;
  
  for (t = 0; t < TSTEPS; t++) {
    for (i = 1; i < N - 1; i++)
      B[i] = 0.33333f * (A[i-1] + A[i] + A[i + 1]);
    for (i = 1; i < N - 1; i++)
      A[i] = 0.33333f * (B[i-1] + B[i] + B[i + 1]);
  }
}