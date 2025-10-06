#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

volatile DATA_TYPE A[N][K];
volatile DATA_TYPE B[K][M];
volatile DATA_TYPE C[N][M];

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
