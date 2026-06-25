#include <stdlib.h>

// covariance: mean-center columns of data (N x N), then cov = data^T*data (symmetric).
//
// cov[i][j] = sum_k (data[k][i]-mean[i]) * (data[k][j]-mean[j])
//           = sum_k dataT[i][k] * dataT[j][k]
// where dataT[i][k] is the mean-centered transpose. With dataT row-major both
// operands of the inner reduction stream contiguously -> dataT @ dataT^T.
//
// Strategy (guided by the dmd affine locality analyzer):
//  1. Compute column means.
//  2. Build a mean-centered transposed copy dataT (also update `data` in place to
//     keep the same observable side effect as the reference: data[i][j]-=mean[j]).
//  3. Tiled, symmetric Gram product with a register-blocked micro-kernel; inner
//     reduction streams unit-stride over k.

#ifndef TI
#define TI 128   // i tile (rows of cov / dataT)
#endif
#ifndef TJ
#define TJ 128   // j tile
#endif
#ifndef TK
#define TK 256   // k tile (reduction depth blocking)
#endif

#define MR 4     // register micro-tile rows
#define NR 4     // register micro-tile cols

void kernel(int N, double* restrict data, double* restrict cov, double* restrict mean){
    // ---- 1. column means ----
    for (int j = 0; j < N; j++) mean[j] = 0.0;
    for (int i = 0; i < N; i++) {
        const double* restrict row = data + (size_t)i * N;
        for (int j = 0; j < N; j++) mean[j] += row[j];
    }
    const double invN = 1.0 / (double)N;
    for (int j = 0; j < N; j++) mean[j] *= invN;

    // ---- 2. mean-center data in place AND build transposed centered copy ----
    double* restrict dataT = (double*)malloc((size_t)N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double* restrict row = data + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            double v = row[j] - mean[j];
            row[j] = v;
            dataT[(size_t)j * N + i] = v;   // dataT[j][i] = centered data[i][j]
        }
    }
    // Now dataT[r][k] = centered data[k][r], so
    // cov[i][j] = sum_k dataT[i][k]*dataT[j][k].

    // ---- 3. tiled symmetric Gram product ----
    // Zero cov once (we accumulate over k-tiles).
    for (size_t t = 0; t < (size_t)N * N; t++) cov[t] = 0.0;

    for (int ii = 0; ii < N; ii += TI) {
        int iend = ii + TI; if (iend > N) iend = N;
        for (int jj = 0; jj < N; jj += TJ) {
            // Symmetry at tile granularity: only compute tiles with jj >= ii.
            if (jj + TJ <= ii) continue;        // wholly below diagonal -> skip
            int jend = jj + TJ; if (jend > N) jend = N;

            for (int kk = 0; kk < N; kk += TK) {
                int kend = kk + TK; if (kend > N) kend = N;

                // micro-kernel over the (i,j) tile, MRxNR register block
                int i = ii;
                for (; i + MR <= iend; i += MR) {
                    const double* restrict a0 = dataT + (size_t)(i + 0) * N;
                    const double* restrict a1 = dataT + (size_t)(i + 1) * N;
                    const double* restrict a2 = dataT + (size_t)(i + 2) * N;
                    const double* restrict a3 = dataT + (size_t)(i + 3) * N;

                    // for diagonal-spanning tiles, start j at max(jj, i) to stay on/above diagonal
                    int j = jj;
                    if (j < i) j = i;            // never compute strictly-lower (i>j) entries

                    for (; j + NR <= jend; j += NR) {
                        const double* restrict b0 = dataT + (size_t)(j + 0) * N;
                        const double* restrict b1 = dataT + (size_t)(j + 1) * N;
                        const double* restrict b2 = dataT + (size_t)(j + 2) * N;
                        const double* restrict b3 = dataT + (size_t)(j + 3) * N;

                        double c00=0,c01=0,c02=0,c03=0;
                        double c10=0,c11=0,c12=0,c13=0;
                        double c20=0,c21=0,c22=0,c23=0;
                        double c30=0,c31=0,c32=0,c33=0;

                        for (int k = kk; k < kend; k++) {
                            double v0 = a0[k], v1 = a1[k], v2 = a2[k], v3 = a3[k];
                            double w0 = b0[k], w1 = b1[k], w2 = b2[k], w3 = b3[k];
                            c00+=v0*w0; c01+=v0*w1; c02+=v0*w2; c03+=v0*w3;
                            c10+=v1*w0; c11+=v1*w1; c12+=v1*w2; c13+=v1*w3;
                            c20+=v2*w0; c21+=v2*w1; c22+=v2*w2; c23+=v2*w3;
                            c30+=v3*w0; c31+=v3*w1; c32+=v3*w2; c33+=v3*w3;
                        }
                        cov[(size_t)(i+0)*N+(j+0)] += c00;
                        cov[(size_t)(i+0)*N+(j+1)] += c01;
                        cov[(size_t)(i+0)*N+(j+2)] += c02;
                        cov[(size_t)(i+0)*N+(j+3)] += c03;
                        cov[(size_t)(i+1)*N+(j+0)] += c10;
                        cov[(size_t)(i+1)*N+(j+1)] += c11;
                        cov[(size_t)(i+1)*N+(j+2)] += c12;
                        cov[(size_t)(i+1)*N+(j+3)] += c13;
                        cov[(size_t)(i+2)*N+(j+0)] += c20;
                        cov[(size_t)(i+2)*N+(j+1)] += c21;
                        cov[(size_t)(i+2)*N+(j+2)] += c22;
                        cov[(size_t)(i+2)*N+(j+3)] += c23;
                        cov[(size_t)(i+3)*N+(j+0)] += c30;
                        cov[(size_t)(i+3)*N+(j+1)] += c31;
                        cov[(size_t)(i+3)*N+(j+2)] += c32;
                        cov[(size_t)(i+3)*N+(j+3)] += c33;
                    }
                    // remainder columns (j tail)
                    for (; j < jend; j++) {
                        const double* restrict b0 = dataT + (size_t)j * N;
                        double c0=0,c1=0,c2=0,c3=0;
                        for (int k = kk; k < kend; k++) {
                            double w = b0[k];
                            c0+=a0[k]*w; c1+=a1[k]*w; c2+=a2[k]*w; c3+=a3[k]*w;
                        }
                        cov[(size_t)(i+0)*N+j] += c0;
                        cov[(size_t)(i+1)*N+j] += c1;
                        cov[(size_t)(i+2)*N+j] += c2;
                        cov[(size_t)(i+3)*N+j] += c3;
                    }
                }
                // remainder rows (i tail)
                for (; i < iend; i++) {
                    const double* restrict a0 = dataT + (size_t)i * N;
                    int j = jj; if (j < i) j = i;
                    for (; j < jend; j++) {
                        const double* restrict b0 = dataT + (size_t)j * N;
                        double c = 0.0;
                        for (int k = kk; k < kend; k++) c += a0[k]*b0[k];
                        cov[(size_t)i*N+j] += c;
                    }
                }
            }
        }
    }

    // ---- 4. mirror upper triangle into lower triangle ----
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            cov[(size_t)j*N+i] = cov[(size_t)i*N+j];
        }
    }

    free(dataT);
}
