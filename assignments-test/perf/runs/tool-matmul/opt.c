// Optimized GEMM: C += A*B, all N x N row-major double.
// Strategy:
//  - Loop order ikj so every array access streams with unit stride (row-major
//    friendly); B[k,*] and C[i,*] rows are reused across the inner j loop.
//  - Cache blocking on (i, k, j) with L2/L1-friendly macro tiles so the working
//    set of B and C tiles stays resident.
//  - Register blocking: accumulate a small MR x NR micro-tile of C in registers
//    across the k dimension, removing C load/store traffic from the inner loop.
//  - Full remainder handling for any N (not just multiples of tile sizes).
//
// The locality analyzer (dmd) showed ikj keeps reuse-distance linear in N
// (vs quadratic N^2/8 for ijk's column-strided B), and tiling makes
// data-movement-per-access flat in N (~2.1 vs growing 20->45). Register
// blocking is the orthogonal constant-factor win removing inner C traffic.

#include <stddef.h>

// Macro (cache) tile sizes. MC x KC tile of A and KC x NC panel of B are
// reused; tuned to be L2-friendly. These are blocking hints only; correctness
// holds for any N via remainder loops.
#define MC 256
#define KC 256
#define NC 1024

// Register micro-kernel dimensions (C micro-tile kept in registers).
#define MR 4
#define NR 4

// Micro-kernel: compute C[ic:ic+mr, jc:jc+nr] += A[ic:ic+mr, kc:kc+kk] *
//                                                B[kc:kc+kk, jc:jc+nr]
// mr <= MR, nr <= NR. Accumulates over kk steps of the k dimension.
static inline void micro_kernel(int N, int mr, int nr, int kk,
                                const double* restrict A,
                                const double* restrict B,
                                double* restrict C,
                                int ic, int jc, int kc) {
    if (mr == MR && nr == NR) {
        // Fully-unrolled 4x4 register tile.
        double c00=0,c01=0,c02=0,c03=0;
        double c10=0,c11=0,c12=0,c13=0;
        double c20=0,c21=0,c22=0,c23=0;
        double c30=0,c31=0,c32=0,c33=0;
        const double* Ap = A + (size_t)ic*N + kc;
        const double* Bp = B + (size_t)kc*N + jc;
        for (int k = 0; k < kk; ++k) {
            double a0 = Ap[0*N + k];
            double a1 = Ap[1*N + k];
            double a2 = Ap[2*N + k];
            double a3 = Ap[3*N + k];
            const double* Br = Bp + (size_t)k*N;
            double b0 = Br[0];
            double b1 = Br[1];
            double b2 = Br[2];
            double b3 = Br[3];
            c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
            c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
            c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
            c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
        }
        double* Cp = C + (size_t)ic*N + jc;
        Cp[0*N+0]+=c00; Cp[0*N+1]+=c01; Cp[0*N+2]+=c02; Cp[0*N+3]+=c03;
        Cp[1*N+0]+=c10; Cp[1*N+1]+=c11; Cp[1*N+2]+=c12; Cp[1*N+3]+=c13;
        Cp[2*N+0]+=c20; Cp[2*N+1]+=c21; Cp[2*N+2]+=c22; Cp[2*N+3]+=c23;
        Cp[3*N+0]+=c30; Cp[3*N+1]+=c31; Cp[3*N+2]+=c32; Cp[3*N+3]+=c33;
    } else {
        // Edge micro-tile (mr<MR or nr<NR). General accumulation.
        double acc[MR][NR];
        for (int i = 0; i < mr; ++i)
            for (int j = 0; j < nr; ++j)
                acc[i][j] = 0.0;
        const double* Ap = A + (size_t)ic*N + kc;
        const double* Bp = B + (size_t)kc*N + jc;
        for (int k = 0; k < kk; ++k) {
            const double* Br = Bp + (size_t)k*N;
            for (int i = 0; i < mr; ++i) {
                double a = Ap[(size_t)i*N + k];
                for (int j = 0; j < nr; ++j)
                    acc[i][j] += a * Br[j];
            }
        }
        double* Cp = C + (size_t)ic*N + jc;
        for (int i = 0; i < mr; ++i)
            for (int j = 0; j < nr; ++j)
                Cp[(size_t)i*N + j] += acc[i][j];
    }
}

void kernel(int N, double* A, double* B, double* C) {
    const double* restrict Ar = A;
    const double* restrict Br = B;
    double* restrict Cr = C;

    // Cache blocking over k (outer, to keep A/B tiles resident), then j, then i.
    for (int kc = 0; kc < N; kc += KC) {
        int kk = (kc + KC <= N) ? KC : (N - kc);
        for (int jc = 0; jc < N; jc += NC) {
            int nc = (jc + NC <= N) ? NC : (N - jc);
            for (int ic = 0; ic < N; ic += MC) {
                int mc = (ic + MC <= N) ? MC : (N - ic);
                // Iterate register micro-tiles within the macro block.
                for (int i = ic; i < ic + mc; i += MR) {
                    int mr = (i + MR <= ic + mc) ? MR : (ic + mc - i);
                    for (int j = jc; j < jc + nc; j += NR) {
                        int nr = (j + NR <= jc + nc) ? NR : (jc + nc - j);
                        micro_kernel(N, mr, nr, kk, Ar, Br, Cr, i, j, kc);
                    }
                }
            }
        }
    }
}
