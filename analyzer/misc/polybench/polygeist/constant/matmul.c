#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

volatile DATA_TYPE A[N][408];  // K=400 padded to 408
volatile DATA_TYPE B[K][300];  // M=300 already multiple of 12
volatile DATA_TYPE C[N][300];  // M=300 already multiple of 12

void matmul() {
  int i, j, k;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
