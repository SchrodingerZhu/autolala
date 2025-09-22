#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024

void kernel_durbin(size_t N, DATA_TYPE r[LIMIT], DATA_TYPE y[LIMIT]) {
  DATA_TYPE z[LIMIT];
  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;
  int i, k;

  y[0] = -r[0];
  beta = 1.0f;
  alpha = -r[0];

  for (k = 1; k < N; k++) {
    beta = (1 - alpha * alpha) * beta;
    sum = 0.0f;
    for (i = 0; i < k; i++) {
      sum += r[k-i-1] * y[i];
    }
    alpha = -(r[k] + sum) / beta;

    for (i = 0; i < k; i++) {
      z[i] = y[i] + alpha * y[k-i-1];
    }
    for (i = 0; i < k; i++) {
      y[i] = z[i];
    }
    y[k] = alpha;
  }
}