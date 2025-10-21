#define DATA_TYPE double
#define SMALL_SIZE 8
#define M_SIZE 8



volatile DATA_TYPE A[SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][24];
volatile DATA_TYPE result[24];
// Marginalization: ijklmnop->m
// Memory access pattern: sum over all indices except m
void kernel_marginalization_pattern() {
  int i, j, k, l, m, n, o, p;
  
  // Initialize output
  for (m = 0; m < M_SIZE; m++)
    result[m] = 0;
  
  // Actual computation for: sum_i sum_j sum_k sum_l sum_n sum_o sum_p A[i][j][k][l][m][n][o][p] -> result[m]
  for (m = 0; m < M_SIZE; m++) {
    result[m] = 0.0; // Initialize output result[m]
    for (i = 0; i < SMALL_SIZE; i++)
      for (j = 0; j < SMALL_SIZE; j++)
        for (k = 0; k < SMALL_SIZE; k++)
          for (l = 0; l < SMALL_SIZE; l++)
            for (n = 0; n < SMALL_SIZE; n++)
              for (o = 0; o < SMALL_SIZE; o++)
                for (p = 0; p < SMALL_SIZE; p++)
                  result[m] += A[i][j][k][l][m][n][o][p];
  }
}
