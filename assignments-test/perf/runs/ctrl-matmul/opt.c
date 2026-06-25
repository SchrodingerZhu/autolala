// Optimized GEMM: C += A*B, all N x N row-major double.
//
// Strategy:
//   * Loop order i-k-j so the innermost loop streams contiguously over
//     B[k*N + j] and C[i*N + j] (unit stride), with A[i*N + k] a loop-invariant
//     scalar that the compiler broadcasts.  This is the auto-vectorizer-friendly
//     ("axpy") form: C[i][:] += A[i][k] * B[k][:].
//   * Register blocking over i (MR rows at once) so each loaded B[k][:] vector is
//     reused across MR independent FMA accumulator chains, raising the
//     compute/load ratio and hiding FMA latency.
//   * Cache blocking over k (KC) and j (NC): the KC x NC block of B is reused by
//     all MR-row strips of the current i-panel while resident in L2/L1.
//   * Remainder loops handle any N (and any leftover rows) — no size hardcoding.

#define MR 4      // rows of A / C handled per register-block step
#define KC 256    // k-blocking (B panel rows kept hot)
#define NC 512    // j-blocking (B panel columns kept hot)

void kernel(int N, double* restrict A, double* restrict B, double* restrict C) {
    for (int kk = 0; kk < N; kk += KC) {
        int kmax = kk + KC; if (kmax > N) kmax = N;
        for (int jj = 0; jj < N; jj += NC) {
            int jmax = jj + NC; if (jmax > N) jmax = N;

            int i = 0;
            // ---- Main register-blocked path: MR rows of C at a time ----
            for (; i + MR <= N; i += MR) {
                double* restrict C0 = C + (i + 0) * N;
                double* restrict C1 = C + (i + 1) * N;
                double* restrict C2 = C + (i + 2) * N;
                double* restrict C3 = C + (i + 3) * N;
                const double* restrict A0 = A + (i + 0) * N;
                const double* restrict A1 = A + (i + 1) * N;
                const double* restrict A2 = A + (i + 2) * N;
                const double* restrict A3 = A + (i + 3) * N;

                for (int k = kk; k < kmax; k++) {
                    const double* restrict Bk = B + k * N;
                    double a0 = A0[k], a1 = A1[k], a2 = A2[k], a3 = A3[k];
                    for (int j = jj; j < jmax; j++) {
                        double b = Bk[j];
                        C0[j] += a0 * b;
                        C1[j] += a1 * b;
                        C2[j] += a2 * b;
                        C3[j] += a3 * b;
                    }
                }
            }
            // ---- Remainder rows (N not a multiple of MR) ----
            for (; i < N; i++) {
                double* restrict Ci = C + i * N;
                const double* restrict Ai = A + i * N;
                for (int k = kk; k < kmax; k++) {
                    double a = Ai[k];
                    const double* restrict Bk = B + k * N;
                    for (int j = jj; j < jmax; j++) {
                        Ci[j] += a * Bk[j];
                    }
                }
            }
        }
    }
}
