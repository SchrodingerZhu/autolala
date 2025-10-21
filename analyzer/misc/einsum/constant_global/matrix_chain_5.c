#define DATA_TYPE double
#define I_SIZE 64
#define K_SIZE 64
#define L_SIZE 64
#define M_SIZE 64
#define N_SIZE 64
#define J_SIZE 64



volatile DATA_TYPE A[I_SIZE][72];
volatile DATA_TYPE B[K_SIZE][72];
volatile DATA_TYPE C[L_SIZE][72];
volatile DATA_TYPE D[M_SIZE][72];
volatile DATA_TYPE E[N_SIZE][72];
volatile DATA_TYPE result[I_SIZE][72];
volatile DATA_TYPE tmp1[I_SIZE][72];
volatile DATA_TYPE tmp2[I_SIZE][72];
volatile DATA_TYPE tmp3[I_SIZE][72];
// Matrix chain multiplication: ik,kl,lm,mn,nj->ij
void kernel_matrix_chain_5() {
  int i, j, k, l, m, n;
  
  // tmp1 := A * B (ik,kl->il)
  for (i = 0; i < I_SIZE; i++)
    for (l = 0; l < L_SIZE; l++) {
      tmp1[i][l] = 0.0;
      for (k = 0; k < K_SIZE; k++)
        tmp1[i][l] += A[i][k] * B[k][l];
    }
  
  // tmp2 := tmp1 * C (il,lm->im)  
  for (i = 0; i < I_SIZE; i++)
    for (m = 0; m < M_SIZE; m++) {
      tmp2[i][m] = 0.0;
      for (l = 0; l < L_SIZE; l++)
        tmp2[i][m] += tmp1[i][l] * C[l][m];
    }
  
  // tmp3 := tmp2 * D (im,mn->in)
  for (i = 0; i < I_SIZE; i++)
    for (n = 0; n < N_SIZE; n++) {
      tmp3[i][n] = 0.0;
      for (m = 0; m < M_SIZE; m++)
        tmp3[i][n] += tmp2[i][m] * D[m][n];
    }
  
  // result := tmp3 * E (in,nj->ij)
  for (i = 0; i < I_SIZE; i++)
    for (j = 0; j < J_SIZE; j++) {
      result[i][j] = 0.0;
      for (n = 0; n < N_SIZE; n++)
        result[i][j] += tmp3[i][n] * E[n][j];
    }
}
