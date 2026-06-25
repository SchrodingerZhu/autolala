// mvt: x1 += A*y1 ; x2 += A^T*y2
// Fused single row-major sweep over A: each A[i*N+j] loaded once and reused
// for both products, eliminating the transposed (column-major) access pattern.
void kernel(int N, double* restrict A, double* restrict x1, double* restrict x2,
            double* restrict y1, double* restrict y2){
    // Tile over j to keep a block of x2[] hot in cache while streaming rows of A.
    // For each row i, x1[i] is a scalar accumulator (reduction).
    // x2[j] is scattered-accumulated, so we block j to reuse the x2 block across
    // multiple i rows... but x2[j] depends on every i, so we tile i as well.
    const int BI = 256;
    const int BJ = 256;

    for (int ii = 0; ii < N; ii += BI) {
        int imax = ii + BI < N ? ii + BI : N;
        for (int jj = 0; jj < N; jj += BJ) {
            int jmax = jj + BJ < N ? jj + BJ : N;
            for (int i = ii; i < imax; i++) {
                const double* restrict Ai = A + (long)i * N;
                double s1 = 0.0;          // partial x1[i] for this j-block
                double yi2 = y2[i];        // scalar broadcast for x2 update
                int j = jj;
                for (; j < jmax; j++) {
                    double a = Ai[j];
                    s1  += a * y1[j];
                    x2[j] += a * yi2;
                }
                x1[i] += s1;
            }
        }
    }
}
