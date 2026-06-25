// syrk: C = 0.9*C + 1.1*A*A^T, lower triangle only (j<=i).
// Optimized: GEMM-style tiling with register blocking.
//
// Math reorganization (exact same result, only FP reassociation in the k-sum):
//   C[i,j] = 0.9*C[i,j] + sum_k (1.1*A[i,k]) * A[j,k]   for j<=i
//
// We compute B[i,k] = 1.1*A[i,k] is NOT materialized; instead we pull the 1.1
// out: C[i,j] = 0.9*C[i,j] + 1.1 * sum_k A[i,k]*A[j,k]. Computing the raw dot
// product sum_k A[i,k]*A[j,k] and then scaling by 1.1 is algebraically equal to
// the reference (sum of (1.1*A[i,k])*A[j,k]) up to FP reassociation, well within
// rtol 1e-6.
//
// Strategy:
//   - First scale lower triangle of C by 0.9 (cheap O(N^2/2) prologue).
//   - Then accumulate the triangular A*A^T into C.
//   - Tile (i,j) into MR x NR register blocks; stream k.
//   - Off-diagonal blocks (block_j_end <= block_i_start) are full rectangles.
//   - Diagonal blocks apply the j<=i mask scalar-wise.

#include <stddef.h>

#define MR 4
#define NR 4
#define KC 256
#define MC 192
#define NC 256

// Process a full MR x NR rectangular tile of C, rows [i0,i0+MR), cols [j0,j0+NR),
// accumulating sum over k in [k0,k1) of A[i,k]*A[j,k], scaled by 1.1, into C.
// All MR,NR in-bounds (no triangle mask). C already pre-scaled by 0.9.
static inline void microkernel_full(int N, const double* restrict A, double* restrict C,
                                    int i0, int j0, int k0, int k1) {
    double c00=0,c01=0,c02=0,c03=0;
    double c10=0,c11=0,c12=0,c13=0;
    double c20=0,c21=0,c22=0,c23=0;
    double c30=0,c31=0,c32=0,c33=0;
    const double* Ai0 = A + (size_t)(i0+0)*N;
    const double* Ai1 = A + (size_t)(i0+1)*N;
    const double* Ai2 = A + (size_t)(i0+2)*N;
    const double* Ai3 = A + (size_t)(i0+3)*N;
    const double* Aj0 = A + (size_t)(j0+0)*N;
    const double* Aj1 = A + (size_t)(j0+1)*N;
    const double* Aj2 = A + (size_t)(j0+2)*N;
    const double* Aj3 = A + (size_t)(j0+3)*N;
    for (int k=k0; k<k1; k++) {
        double a0=Ai0[k], a1=Ai1[k], a2=Ai2[k], a3=Ai3[k];
        double b0=Aj0[k], b1=Aj1[k], b2=Aj2[k], b3=Aj3[k];
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3;
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3;
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3;
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3;
    }
    double* C0 = C + (size_t)(i0+0)*N + j0;
    double* C1 = C + (size_t)(i0+1)*N + j0;
    double* C2 = C + (size_t)(i0+2)*N + j0;
    double* C3 = C + (size_t)(i0+3)*N + j0;
    C0[0]+=1.1*c00; C0[1]+=1.1*c01; C0[2]+=1.1*c02; C0[3]+=1.1*c03;
    C1[0]+=1.1*c10; C1[1]+=1.1*c11; C1[2]+=1.1*c12; C1[3]+=1.1*c13;
    C2[0]+=1.1*c20; C2[1]+=1.1*c21; C2[2]+=1.1*c22; C2[3]+=1.1*c23;
    C3[0]+=1.1*c30; C3[1]+=1.1*c31; C3[2]+=1.1*c32; C3[3]+=1.1*c33;
}

// Generic block, handles arbitrary i-range x j-range with optional triangle mask.
// Accumulates over k in [k0,k1).
static inline void block_generic(int N, const double* restrict A, double* restrict C,
                                 int i0, int i1, int j0, int j1, int k0, int k1,
                                 int diag) {
    for (int i=i0; i<i1; i++) {
        const double* Ai = A + (size_t)i*N;
        int jend = j1;
        if (diag && jend > i+1) jend = i+1;   // only j<=i
        double* Ci = C + (size_t)i*N;
        for (int j=j0; j<jend; j++) {
            const double* Aj = A + (size_t)j*N;
            double acc=0;
            for (int k=k0; k<k1; k++) acc += Ai[k]*Aj[k];
            Ci[j] += 1.1*acc;
        }
    }
}

void kernel(int N, double* A, double* C){
    const double* restrict Ar = A;
    double* restrict Cr = C;

    // Prologue: scale lower triangle of C by 0.9.
    for (int i=0;i<N;i++){
        double* Ci = Cr + (size_t)i*N;
        for (int j=0;j<=i;j++) Ci[j]*=0.9;
    }

    // Tiled triangular A*A^T accumulation.
    // Outer cache blocking over j (NC), i (MC), k (KC).
    for (int jc=0; jc<N; jc+=NC) {
        int jcend = jc+NC; if (jcend>N) jcend=N;
        for (int ic=0; ic<N; ic+=MC) {
            int icend = ic+MC; if (icend>N) icend=N;
            // Lower triangle: need j<=i, so only process tiles where jc <= icend-1.
            if (jc > icend-1) continue;
            for (int kc=0; kc<N; kc+=KC) {
                int kcend = kc+KC; if (kcend>N) kcend=N;

                // Within this cache block, register-block over i (MR) and j (NR).
                for (int i=ic; i<icend; i+=MR) {
                    int iend = i+MR; if (iend>icend) iend=icend;
                    int jlimit = jcend; if (jlimit > i+MR) {} // cap below per-block
                    for (int j=jc; j<jcend; j+=NR) {
                        int jend = j+NR; if (jend>jcend) jend=jcend;

                        // Determine relationship of this MRxNR block to the diagonal.
                        // Block rows [i, iend), cols [j, jend).
                        // If jend-1 <= i  -> fully below diagonal (all j<=i for every row) -> full kernel
                        //    (smallest i in block is i; need j <= every row's i. The row with
                        //     smallest index is i, so require jend-1 <= i.)
                        // If j > iend-1   -> fully above diagonal (all entries have j>i) -> skip.
                        // Else            -> straddles diagonal -> generic masked.
                        if (j > iend-1) {
                            // entirely strictly upper (j>i for all) -> skip; but careful:
                            // j>i means above diagonal, not in lower triangle.
                            continue;
                        }
                        if (jend-1 <= i && iend-i==MR && jend-j==NR) {
                            microkernel_full(N, Ar, Cr, i, j, kc, kcend);
                        } else {
                            // straddles diagonal or remainder edge -> generic masked
                            block_generic(N, Ar, Cr, i, iend, j, jend, kc, kcend, 1);
                        }
                    }
                }
            }
        }
    }
}
