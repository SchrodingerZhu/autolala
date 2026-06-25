// Optimized GEMM: C = 0.9*C + 1.1*A*B  (all N x N, row-major double).
//
// Strategy:
//  - Fuse the C *= 0.9 scaling into the first k-block update so we never make
//    a separate pass over C.
//  - Pre-scale A by 1.1 once into a packed buffer so the inner loop is a plain
//    FMA (c += a*b) with no extra multiply, and so A is accessed contiguously.
//  - Loop order i -> k -> j: the innermost j loop is unit-stride over B[k,*]
//    and C[i,*], which vectorizes cleanly (clang -O3 -march=native).
//  - Register-block over i (MR rows at a time): each B[k,j] value is reused
//    across MR C rows, and the alpha*A[i,k] scalars stay in registers.
//  - Cache-block over k (KC) and j (NC) so the active B panel and C row block
//    stay resident in L1/L2.
//
// All blocking factors degrade gracefully with remainder loops, so any N works.

#include <stdlib.h>
#include <string.h>

#define MR 4          // rows of C handled per micro-kernel pass (register block)
#define KC 256        // k-block: B panel rows kept hot in cache
#define NC 512        // j-block: width of C/B panel kept in cache

void kernel(int N, double* restrict A, double* restrict B, double* restrict C) {
    const long n = N;
    const double beta  = 0.9;
    const double alpha = 1.1;

    // Packed buffer of alpha*A for an MR x KC block, row-major (MR rows).
    double* Apack = (double*) malloc((size_t)MR * KC * sizeof(double));
    if (!Apack) {
        // Fallback: simple correct path if allocation fails.
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < n; j++) {
                double c = beta * C[i*n+j];
                for (long k = 0; k < n; k++) c += alpha * A[i*n+k] * B[k*n+j];
                C[i*n+j] = c;
            }
        }
        return;
    }

    for (long jc = 0; jc < n; jc += NC) {
        const long jb = (jc + NC <= n) ? NC : (n - jc);

        for (long kc = 0; kc < n; kc += KC) {
            const long kb = (kc + KC <= n) ? KC : (n - kc);
            const int first_k = (kc == 0);   // apply beta scaling on first k-block

            for (long ic = 0; ic < n; ic += MR) {
                const long ib = (ic + MR <= n) ? MR : (n - ic);

                // Pack alpha*A[ic..ic+ib, kc..kc+kb] into Apack (ib rows of kb).
                for (long ii = 0; ii < ib; ii++) {
                    const double* Arow = A + (ic + ii) * n + kc;
                    double* Aprow = Apack + ii * KC;
                    for (long kk = 0; kk < kb; kk++)
                        Aprow[kk] = alpha * Arow[kk];
                }

                if (ib == MR) {
                    // Full MR-row micro-kernel.
                    double* C0 = C + (ic + 0) * n + jc;
                    double* C1 = C + (ic + 1) * n + jc;
                    double* C2 = C + (ic + 2) * n + jc;
                    double* C3 = C + (ic + 3) * n + jc;
                    const double* Ap0 = Apack + 0 * KC;
                    const double* Ap1 = Apack + 1 * KC;
                    const double* Ap2 = Apack + 2 * KC;
                    const double* Ap3 = Apack + 3 * KC;

                    if (first_k) {
                        for (long j = 0; j < jb; j++) {
                            C0[j] *= beta; C1[j] *= beta;
                            C2[j] *= beta; C3[j] *= beta;
                        }
                    }
                    for (long kk = 0; kk < kb; kk++) {
                        const double a0 = Ap0[kk];
                        const double a1 = Ap1[kk];
                        const double a2 = Ap2[kk];
                        const double a3 = Ap3[kk];
                        const double* Brow = B + (kc + kk) * n + jc;
                        for (long j = 0; j < jb; j++) {
                            const double b = Brow[j];
                            C0[j] += a0 * b;
                            C1[j] += a1 * b;
                            C2[j] += a2 * b;
                            C3[j] += a3 * b;
                        }
                    }
                } else {
                    // Remainder rows (ib < MR): scalar-per-row fallback.
                    for (long ii = 0; ii < ib; ii++) {
                        double* Crow = C + (ic + ii) * n + jc;
                        const double* Aprow = Apack + ii * KC;
                        if (first_k)
                            for (long j = 0; j < jb; j++) Crow[j] *= beta;
                        for (long kk = 0; kk < kb; kk++) {
                            const double a = Aprow[kk];
                            const double* Brow = B + (kc + kk) * n + jc;
                            for (long j = 0; j < jb; j++)
                                Crow[j] += a * Brow[j];
                        }
                    }
                }
            }
        }
    }

    free(Apack);
}
