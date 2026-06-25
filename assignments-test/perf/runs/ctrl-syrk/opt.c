#include <string.h>

// C = 0.9*C + 1.1*A*A^T, lower triangle only.
// Equivalent: C[i][j] = 0.9*C[i][j] + 1.1 * sum_k A[i][k]*A[j][k]  for j<=i.
//
// Reformulated as row-i . row-j dot products (both row-major friendly),
// with 4x4 register blocking on (i,j) to reuse loaded A values, and k-tiling
// for L1/L2 cache locality.

#define MC 4   // i-block (rows of A held in registers)
#define NC 4   // j-block (rows of A held in registers)
#define KB 256 // k-tile width for cache blocking

void kernel(int N, double* restrict A, double* restrict C) {
    const double alpha = 1.1;
    const double beta  = 0.9;

    // 1) Scale lower triangle of C by beta.
    for (int i = 0; i < N; i++) {
        double* Ci = C + (long)i * N;
        for (int j = 0; j <= i; j++) Ci[j] *= beta;
    }

    // 2) Rank-k update: C[i][j] += alpha * (row_i . row_j), j<=i.
    // Tile over k so the active strips of A stay in cache.
    for (int kk = 0; kk < N; kk += KB) {
        int kend = kk + KB; if (kend > N) kend = N;

        for (int i = 0; i < N; i += MC) {
            int imax = i + MC; if (imax > N) imax = N;
            int ifull = (imax - i == MC);

            // j-blocks span [0, imax): only need columns up to row index.
            for (int j = 0; j < imax; j += NC) {
                int jmax = j + NC; if (jmax > N) jmax = N;

                // Fully-below-diagonal AND full MCxNC tile: fast path.
                int full = ifull && (jmax - j == NC) && (j + NC <= i);

                if (full) {
                    const double* Ai0 = A + (long)(i + 0) * N;
                    const double* Ai1 = A + (long)(i + 1) * N;
                    const double* Ai2 = A + (long)(i + 2) * N;
                    const double* Ai3 = A + (long)(i + 3) * N;
                    const double* Aj0 = A + (long)(j + 0) * N;
                    const double* Aj1 = A + (long)(j + 1) * N;
                    const double* Aj2 = A + (long)(j + 2) * N;
                    const double* Aj3 = A + (long)(j + 3) * N;

                    double c00=0,c01=0,c02=0,c03=0;
                    double c10=0,c11=0,c12=0,c13=0;
                    double c20=0,c21=0,c22=0,c23=0;
                    double c30=0,c31=0,c32=0,c33=0;

                    for (int k = kk; k < kend; k++) {
                        double a0 = Ai0[k], a1 = Ai1[k], a2 = Ai2[k], a3 = Ai3[k];
                        double b0 = Aj0[k], b1 = Aj1[k], b2 = Aj2[k], b3 = Aj3[k];
                        c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                        c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
                        c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
                        c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
                    }

                    double* C0 = C + (long)(i + 0) * N + j;
                    double* C1 = C + (long)(i + 1) * N + j;
                    double* C2 = C + (long)(i + 2) * N + j;
                    double* C3 = C + (long)(i + 3) * N + j;
                    C0[0]+=alpha*c00; C0[1]+=alpha*c01; C0[2]+=alpha*c02; C0[3]+=alpha*c03;
                    C1[0]+=alpha*c10; C1[1]+=alpha*c11; C1[2]+=alpha*c12; C1[3]+=alpha*c13;
                    C2[0]+=alpha*c20; C2[1]+=alpha*c21; C2[2]+=alpha*c22; C2[3]+=alpha*c23;
                    C3[0]+=alpha*c30; C3[1]+=alpha*c31; C3[2]+=alpha*c32; C3[3]+=alpha*c33;
                } else {
                    // Edge / diagonal block: scalar with j<=i guard.
                    for (int ii = i; ii < imax; ii++) {
                        const double* Ai = A + (long)ii * N;
                        double* Cr = C + (long)ii * N;
                        // valid columns: jj in [j, min(jmax, ii+1))
                        int jhi = (jmax < ii + 1) ? jmax : ii + 1;
                        for (int jj = j; jj < jhi; jj++) {
                            const double* Aj = A + (long)jj * N;
                            double acc = 0.0;
                            for (int k = kk; k < kend; k++) acc += Ai[k] * Aj[k];
                            Cr[jj] += alpha * acc;
                        }
                    }
                }
            }
        }
    }
}
