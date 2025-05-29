#define M 256
#define N 256
#define K 256

volatile float A[M][K];
volatile float B[K][N];
volatile float C[M][N];

void _start() {
  for (int i = 0; i < M / 128; ++i)
    for (int j = 0; j < N / 128; ++j)
      for (int k = 0; k < K / 128; ++k)
        for (int ii = 0; ii < 128 / 32; ++ii)
          for (int jj = 0; jj < 128 / 32; ++jj)
            for (int kk = 0; kk < 128 / 32; ++kk)
              for (int iii = 0; iii < 32; ++iii)
                for (int jjj = 0; jjj < 32; ++jjj)
                  for (int kkk = 0; kkk < 32; ++kkk) {
                    C[i * 128 + ii * 32 + iii][j * 128 + jj * 32 + jjj] =
                        A[i * 128 + ii * 32 + iii][k * 128 + kk * 32 + kkk] *
                        B[k * 128 + kk * 32 + kkk][j * 128 + jj * 32 + jjj];
                    asm volatile("" ::: "memory");
                  }
  asm volatile("xor %%edi, %%edi\n\t"
               "mov $60, %%eax\n\t"
               "syscall"
               :
               :);
}
