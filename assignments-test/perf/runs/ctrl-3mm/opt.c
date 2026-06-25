#include <string.h>

// Computes C = A * B for NxN row-major matrices.
// Uses an i-k-j loop order so the innermost j-loop is a contiguous,
// vectorizable AXPY over a row of B and a row of C. Cache-blocked in
// all three dimensions, with register blocking on the i dimension.
static void mm(int N,
               const double* restrict A,
               const double* restrict B,
               double* restrict C) {
    // Tile sizes tuned for L1/L2 with double precision.
    const int IB = 64;   // rows of A / C held in registers-ish block
    const int KB = 256;  // depth block (rows of B reused from cache)
    const int JB = 512;  // column block (B/C row span kept in L1/L2)

    // Zero the output once up front.
    memset(C, 0, (size_t)N * N * sizeof(double));

    for (int ii = 0; ii < N; ii += IB) {
        int imax = ii + IB < N ? ii + IB : N;
        for (int kk = 0; kk < N; kk += KB) {
            int kmax = kk + KB < N ? kk + KB : N;
            for (int jj = 0; jj < N; jj += JB) {
                int jmax = jj + JB < N ? jj + JB : N;

                int i = ii;
                // Process 4 rows of A/C at a time to reuse B rows across them.
                for (; i + 4 <= imax; i += 4) {
                    double* restrict C0 = C + (size_t)(i + 0) * N;
                    double* restrict C1 = C + (size_t)(i + 1) * N;
                    double* restrict C2 = C + (size_t)(i + 2) * N;
                    double* restrict C3 = C + (size_t)(i + 3) * N;
                    const double* restrict A0 = A + (size_t)(i + 0) * N;
                    const double* restrict A1 = A + (size_t)(i + 1) * N;
                    const double* restrict A2 = A + (size_t)(i + 2) * N;
                    const double* restrict A3 = A + (size_t)(i + 3) * N;
                    for (int k = kk; k < kmax; k++) {
                        const double* restrict Bk = B + (size_t)k * N;
                        double a0 = A0[k];
                        double a1 = A1[k];
                        double a2 = A2[k];
                        double a3 = A3[k];
                        for (int j = jj; j < jmax; j++) {
                            double b = Bk[j];
                            C0[j] += a0 * b;
                            C1[j] += a1 * b;
                            C2[j] += a2 * b;
                            C3[j] += a3 * b;
                        }
                    }
                }
                // Remainder rows.
                for (; i < imax; i++) {
                    double* restrict Ci = C + (size_t)i * N;
                    const double* restrict Ai = A + (size_t)i * N;
                    for (int k = kk; k < kmax; k++) {
                        const double* restrict Bk = B + (size_t)k * N;
                        double a = Ai[k];
                        for (int j = jj; j < jmax; j++) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
}

void kernel(int N, double* A, double* B, double* C, double* D,
            double* E, double* F, double* G) {
    mm(N, A, B, E); // E = A * B
    mm(N, C, D, F); // F = C * D
    mm(N, E, F, G); // G = E * F
}
