// atax: tmp = A*x ; y = A^T*tmp   (A is N x N, row-major)
//
// Key optimization: FUSE the two sweeps over A so the N x N matrix (32MB..512MB,
// far larger than any cache) is streamed from DRAM exactly ONCE instead of twice.
//
// The reference does:
//   (1) tmp[i] = sum_j A[i][j]*x[j]      -- full read of A
//   (2) y[j]  += A[i][j]*tmp[i]          -- second full read of A
// Each pass reads the whole matrix; A traffic dominates because A >> LLC.
//
// Fused form: for each row i we first compute t = tmp[i] = A[i]·x, then immediately
// scatter y[j] += A[i][j]*t using the SAME row that is still hot in cache. A is
// therefore brought in from memory only once -> ~2x less DRAM traffic on the
// bottleneck array. x and y are tiny (N doubles each, reused every row) and stay
// resident in L1/L2.
//
// We must finish the full dot product (need t) before the y-update of a row, so the
// two uses of row i are split. A single row is at most 8192 doubles = 64KB, which
// fits in L2; the second use hits L2, while the matrix is still read from DRAM once.
// Multiple scalar accumulators + 4-wide unroll expose ILP for the FMA chains.

void kernel(int N, double* restrict A, double* restrict x,
            double* restrict y, double* restrict tmp){
    for (int j = 0; j < N; j++) y[j] = 0.0;

    for (int i = 0; i < N; i++) {
        const double* restrict Ai = A + (long)i * N;

        // Pass 1: t = sum_j Ai[j] * x[j]
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            s0 += Ai[j]   * x[j];
            s1 += Ai[j+1] * x[j+1];
            s2 += Ai[j+2] * x[j+2];
            s3 += Ai[j+3] * x[j+3];
        }
        for (; j < N; j++) s0 += Ai[j] * x[j];
        double t = (s0 + s1) + (s2 + s3);
        tmp[i] = t;

        // Pass 2 over the same (cached) row: y[j] += Ai[j] * t
        j = 0;
        for (; j + 4 <= N; j += 4) {
            y[j]   += Ai[j]   * t;
            y[j+1] += Ai[j+1] * t;
            y[j+2] += Ai[j+2] * t;
            y[j+3] += Ai[j+3] * t;
        }
        for (; j < N; j++) y[j] += Ai[j] * t;
    }
}
