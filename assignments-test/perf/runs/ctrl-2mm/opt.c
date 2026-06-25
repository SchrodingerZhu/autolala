// 2mm optimized: tmp = 1.1*A*B ;  D = 0.9*D + tmp*C
//
// Both stages are GEMMs of the form  X = alpha*X0 + L*R  with L, R, X all NxN
// row-major. The reference uses an i,j,k order whose inner k-loop walks the
// right operand column-wise (B[k*N+j], C[k*N+j]) -> one cache miss per FMA.
//
// We use an i,k,j order with register blocking on i (4 rows at a time) so:
//   * the inner j-loop streams the R-row (B[k*..], C[k*..]) and the output rows
//     contiguously (unit stride, vectorizable by clang),
//   * each loaded R element is reused across the 4 accumulating output rows,
//     cutting memory traffic ~4x and giving SIMD a clean FMA pattern.
// j is cache-blocked so the accumulator output strips stay in L1/L2.
//
// Math is preserved: GEMM1 folds the 1.1 scale into the A operand value used
// for the FMA; GEMM2 initializes the output row with 0.9*D before accumulating
// tmp*C. Reassociation is only over the k-summation order (kept in order here),
// so results stay all-close.

#include <string.h>

#define JB 256   // j cache block (columns of the output strip)

static void gemm(int N, const double* restrict L, const double* restrict R,
                 double* restrict X, double alpha, double lscale) {
    // X = alpha*X + lscale * (L * R)
    for (int jj = 0; jj < N; jj += JB) {
        int jmax = jj + JB; if (jmax > N) jmax = N;

        int i = 0;
        for (; i + 4 <= N; i += 4) {
            double* restrict x0 = X + (i + 0) * N;
            double* restrict x1 = X + (i + 1) * N;
            double* restrict x2 = X + (i + 2) * N;
            double* restrict x3 = X + (i + 3) * N;

            // init / scale the output strip
            if (alpha == 0.0) {
                for (int j = jj; j < jmax; j++) { x0[j]=0; x1[j]=0; x2[j]=0; x3[j]=0; }
            } else {
                for (int j = jj; j < jmax; j++) {
                    x0[j] *= alpha; x1[j] *= alpha; x2[j] *= alpha; x3[j] *= alpha;
                }
            }

            for (int k = 0; k < N; k++) {
                const double* restrict r = R + k * N;
                double a0 = L[(i + 0) * N + k] * lscale;
                double a1 = L[(i + 1) * N + k] * lscale;
                double a2 = L[(i + 2) * N + k] * lscale;
                double a3 = L[(i + 3) * N + k] * lscale;
                for (int j = jj; j < jmax; j++) {
                    double rj = r[j];
                    x0[j] += a0 * rj;
                    x1[j] += a1 * rj;
                    x2[j] += a2 * rj;
                    x3[j] += a3 * rj;
                }
            }
        }

        // remainder rows (N not a multiple of 4)
        for (; i < N; i++) {
            double* restrict x0 = X + i * N;
            if (alpha == 0.0) {
                for (int j = jj; j < jmax; j++) x0[j] = 0;
            } else {
                for (int j = jj; j < jmax; j++) x0[j] *= alpha;
            }
            for (int k = 0; k < N; k++) {
                double a0 = L[i * N + k] * lscale;
                const double* restrict r = R + k * N;
                for (int j = jj; j < jmax; j++) x0[j] += a0 * r[j];
            }
        }
    }
}

void kernel(int N, double* A, double* B, double* C, double* D, double* tmp) {
    // tmp = 1.1 * (A*B)
    gemm(N, A, B, tmp, 0.0, 1.1);
    // D = 0.9*D + tmp*C
    gemm(N, tmp, C, D, 0.9, 1.0);
}
