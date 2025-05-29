#define M 256
#define N 256
#define K 256

volatile float A[M][K];
volatile float B[K][N];
volatile float C[M][N];

void _start() {
  for (int i = 0; i < M / 32; ++i)
    for (int j = 0; j < N / 32; ++j)
      for (int k = 0; k < K / 32; ++k)
        for (int ii = 0; ii < 32; ++ii)
          for (int jj = 0; jj < 32; ++jj)
            for (int kk = 0; kk < 32; ++kk) {
              C[i * 32 + ii][j * 32 + jj] =
                  A[i * 32 + ii][k * 32 + kk] * B[k * 32 + kk][j * 32 + jj];
              asm volatile("" ::: "memory");
            }
  asm volatile("xor %%edi, %%edi\n\t"
               "mov $60, %%eax\n\t"
               "syscall"
               :
               :);
}
