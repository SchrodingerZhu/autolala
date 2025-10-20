// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/medley/nussinov/nussinov.h
#define N 500
#define DATA_TYPE int


volatile DATA_TYPE seq[504];  // N=500 padded to 504
volatile DATA_TYPE table[N][504];  // N=500 padded to 504

void kernel_nussinov() {
  int i, j, k;

  for (i = N-1; i >= 0; i--) {
    for (j = i+1; j < N; j++) {
      if (j-1 >= 0)
        table[i][j] = table[i][j] > table[i][j-1] ? table[i][j] : table[i][j-1];
      if (i+1 < N)
        table[i][j] = table[i][j] > table[i+1][j] ? table[i][j] : table[i+1][j];

      if (j-1 >= 0 && i+1 < N) {
        if (i < j-1)
          table[i][j] = table[i][j] > (table[i+1][j-1] + ((seq[i] + seq[j]) == 3 ? 1 : 0)) ? table[i][j] : (table[i+1][j-1] + ((seq[i] + seq[j]) == 3 ? 1 : 0));
        else
          table[i][j] = table[i][j] > table[i+1][j-1] ? table[i][j] : table[i+1][j-1];
      }

      for (k = i+1; k < j; k++) {
        table[i][j] = table[i][j] > (table[i][k] + table[k+1][j]) ? table[i][j] : (table[i][k] + table[k+1][j]);
      }
    }
  }
}
