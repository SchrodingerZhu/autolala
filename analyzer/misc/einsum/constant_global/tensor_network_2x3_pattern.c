#define DATA_TYPE double
#define I_SIZE 8
#define J_SIZE 8
#define M_SIZE 8
#define L_SIZE 8
#define K_SIZE 8
#define N_SIZE 8
#define O_SIZE 8



volatile DATA_TYPE A[I_SIZE][J_SIZE];
// 2×3-tensor network: ij,iml,lo,jk,kmn,no->
// Memory access pattern: complex tensor contraction to scalar
void kernel_tensor_network_2x3_pattern() {
  int i, j, m, l, k, n, o;
  
  *result = 0.0;
  
  // Actual computation for: sum over all indices A[i][j] * B[i][m][l] * C[l][o] * D[j][k] * E[k][m][n] * F[n][o] -> scalar
  for (i = 0; i < I_SIZE; i++) {
    for (j = 0; j < J_SIZE; j++) {
      for (m = 0; m < M_SIZE; m++) {
        for (l = 0; l < L_SIZE; l++) {
          for (k = 0; k < K_SIZE; k++) {
            for (n = 0; n < N_SIZE; n++) {
              for (o = 0; o < O_SIZE; o++) {
                *result += A[i][j] * B[i][m][l] * C[l][o] * D[j][k] * E[k][m][n] * F[n][o];
              }
            }
          }
        }
      }
    }
  }
}
