#define M 512
#define N 512
#define K 512

volatile float A[M][K];
volatile float B[K][N];
volatile float C[M][N];

void _start() {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k) {
        C[i][j] = A[i][k] * B[k][j];
        asm volatile("" ::: "memory");
      }
  asm volatile("xor %%edi, %%edi\n\t"
               "mov $60, %%eax\n\t"
               "syscall"
               :
               :);
}
