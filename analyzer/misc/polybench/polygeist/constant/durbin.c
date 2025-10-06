#define N 2000
#define DATA_TYPE float


volatile DATA_TYPE r[N];
volatile DATA_TYPE y[N];

void kernel_durbin() {
  DATA_TYPE z[N];
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
