// Optimized 2mm: tmp = 1.1*A*B ; D = 0.9*D + tmp*C
// Strategy: tiled (blocked) GEMM with ikj inner ordering so the inner loop
// streams contiguously over j (B[k,j]/C[k,j] and the output row). This keeps
// the working set per tile resident in L1/L2 and turns the re-streamed N^2
// footprint into a tile-sized footprint. Remainder loops handle any N.
//
// GEMM1: the 1.1 scale is folded into the A factor (aval = 1.1*A[i,k]).
// GEMM2: D is pre-scaled by 0.9 once, then tmp*C is accumulated in place.

#ifndef TILE
#define TILE 64
#endif

static void gemm_scaledA(int N, const double* restrict A, const double* restrict B,
                         double* restrict Cout, double scaleA) {
    // Cout = scaleA * A * B   (Cout assumed zero-initialized by caller).
    for (int ii = 0; ii < N; ii += TILE) {
        int imax = ii + TILE < N ? ii + TILE : N;
        for (int kk = 0; kk < N; kk += TILE) {
            int kmax = kk + TILE < N ? kk + TILE : N;
            for (int jj = 0; jj < N; jj += TILE) {
                int jmax = jj + TILE < N ? jj + TILE : N;
                for (int i = ii; i < imax; i++) {
                    const double* Arow = A + (long)i * N;
                    double* Crow = Cout + (long)i * N;
                    for (int k = kk; k < kmax; k++) {
                        double aval = scaleA * Arow[k];
                        const double* Brow = B + (long)k * N;
                        for (int j = jj; j < jmax; j++) {
                            Crow[j] += aval * Brow[j];
                        }
                    }
                }
            }
        }
    }
}

void kernel(int N, double* A, double* B, double* C, double* D, double* tmp) {
    double* restrict rA = A;
    double* restrict rB = B;
    double* restrict rC = C;
    double* restrict rD = D;
    double* restrict rtmp = tmp;

    // GEMM1: tmp = 1.1 * A * B. Zero-init tmp first.
    for (long idx = 0, tot = (long)N * N; idx < tot; idx++) rtmp[idx] = 0.0;
    gemm_scaledA(N, rA, rB, rtmp, 1.1);

    // GEMM2: D = 0.9*D + tmp*C. Pre-scale D by 0.9.
    for (long idx = 0, tot = (long)N * N; idx < tot; idx++) rD[idx] *= 0.9;
    gemm_scaledA(N, rtmp, rC, rD, 1.0);
}
