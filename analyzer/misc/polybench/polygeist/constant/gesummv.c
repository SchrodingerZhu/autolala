// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gesummv/gesummv.h
#define N 250
#define DATA_TYPE float
#define ALPHA 1.5f
#define BETA 1.2f


volatile DATA_TYPE A[251][296];  // N=250 padded to 251 (prime) for first dim, N=250 padded to 296 (8×37)
volatile DATA_TYPE B[251][296];  // N=250 padded to 251 (prime) for first dim, N=250 padded to 296 (8×37)
volatile DATA_TYPE tmp[251];  // N=250 padded to 251 (prime)
volatile DATA_TYPE x[296];  // N=250 padded to 296 (8×37)
volatile DATA_TYPE y[251];  // N=250 padded to 251 (prime)

void kernel_gesummv() {
  int i, j;

  for (i = 0; i < N; i++) {
    tmp[i] = 0.0f;
    y[i] = 0.0f;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}
