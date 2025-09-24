#define DATA_TYPE float
#define LIMIT 1024
typedef __SIZE_TYPE__ size_t;

void matmul(size_t n, size_t k, size_t m, DATA_TYPE A[LIMIT][LIMIT], DATA_TYPE B[LIMIT][LIMIT], DATA_TYPE C[LIMIT][LIMIT]) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            DATA_TYPE sum = 0;
            for (size_t kk = 0; kk < k; kk++) {
                sum += A[i][kk] * B[kk][j];
            }
            C[i][j] = sum;
        }
    }
}
