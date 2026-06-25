#include <stddef.h>
// gesummv: y = 1.1*(A*x) + 0.9*(B*x), A,B are N x N row-major.
//
// Strategy:
//   * Single fused pass: each element of A and B is read exactly once
//     (the 2*N^2 compulsory stream is irreducible).
//   * Unroll the outer i loop by 4. Every loaded x[j] is reused across 4
//     rows of A and 4 rows of B, cutting x's re-streaming traffic 4x and
//     giving the compiler 8 independent FMA accumulators (4 for A, 4 for B)
//     to hide FMA latency and saturate the SIMD units.
//   * Inner j loop is a plain unit-stride reduction over each row, which the
//     compiler vectorizes cleanly and the HW prefetcher streams perfectly.
//   * Remainder rows (N % 4) handled by a scalar tail loop.
//
// x (N doubles, 16KB..64KB for N in [2048,8192]) stays resident in L2 across
// the whole sweep, so no explicit j-tiling is needed; i-unrolling is the
// lever that reduces x traffic and exposes ILP.
void kernel(int N, double* restrict A, double* restrict B,
            double* restrict x, double* restrict y, double* restrict tmp){
    int i = 0;

    // Process 4 rows at a time.
    for(; i + 4 <= N; i += 4){
        const double* a0 = A + (size_t)(i+0)*N;
        const double* a1 = A + (size_t)(i+1)*N;
        const double* a2 = A + (size_t)(i+2)*N;
        const double* a3 = A + (size_t)(i+3)*N;
        const double* b0 = B + (size_t)(i+0)*N;
        const double* b1 = B + (size_t)(i+1)*N;
        const double* b2 = B + (size_t)(i+2)*N;
        const double* b3 = B + (size_t)(i+3)*N;

        double ta0=0, ta1=0, ta2=0, ta3=0;
        double tb0=0, tb1=0, tb2=0, tb3=0;

        for(int j=0;j<N;j++){
            double xj = x[j];
            ta0 += a0[j]*xj;  tb0 += b0[j]*xj;
            ta1 += a1[j]*xj;  tb1 += b1[j]*xj;
            ta2 += a2[j]*xj;  tb2 += b2[j]*xj;
            ta3 += a3[j]*xj;  tb3 += b3[j]*xj;
        }

        tmp[i+0]=ta0; y[i+0]=1.1*ta0+0.9*tb0;
        tmp[i+1]=ta1; y[i+1]=1.1*ta1+0.9*tb1;
        tmp[i+2]=ta2; y[i+2]=1.1*ta2+0.9*tb2;
        tmp[i+3]=ta3; y[i+3]=1.1*ta3+0.9*tb3;
    }

    // Remainder rows.
    for(; i < N; i++){
        const double* a0 = A + (size_t)i*N;
        const double* b0 = B + (size_t)i*N;
        double ta=0, tb=0;
        for(int j=0;j<N;j++){
            double xj = x[j];
            ta += a0[j]*xj;
            tb += b0[j]*xj;
        }
        tmp[i]=ta;
        y[i]=1.1*ta+0.9*tb;
    }
}
