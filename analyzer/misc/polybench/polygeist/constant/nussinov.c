#define N 2500
#define DATA_TYPE int


volatile DATA_TYPE seq[N];
volatile DATA_TYPE table[N][N];

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
