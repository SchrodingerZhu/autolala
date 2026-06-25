#include <string.h>

// covariance: mean-center columns of data (N x N), then cov = data^T * data (symmetric).
// Optimized for single-core native performance.
//
// Key transformations vs ref:
//  1. Mean accumulation done row-wise (contiguous) instead of column-wise.
//  2. Centering fused with row-wise traversal.
//  3. Covariance: cov[i][j] = sum_k data[k][i]*data[k][j] = (data^T data).
//     Reorder so k is the OUTER loop -> each access data[k][*] is contiguous.
//     This turns the cache-hostile column-strided inner loop into rank-1 updates
//     of a cov tile that stays resident in cache. Tile over (i,j) so the active
//     cov block fits in L2/L1; only the upper triangle is computed (symmetry).

#ifndef TI
#define TI 64
#endif
#ifndef TJ
#define TJ 256
#endif

void kernel(int N, double* restrict data, double* restrict cov, double* restrict mean){
    // ---- Phase 1: column means, computed row-wise (contiguous reads) ----
    for(int j=0;j<N;j++) mean[j]=0.0;
    for(int k=0;k<N;k++){
        const double* restrict row = data + (long)k*N;
        for(int j=0;j<N;j++) mean[j]+=row[j];
    }
    const double invN = 1.0/(double)N;
    for(int j=0;j<N;j++) mean[j]*=invN;

    // ---- Phase 2: center columns (row-wise, contiguous) ----
    for(int k=0;k<N;k++){
        double* restrict row = data + (long)k*N;
        for(int j=0;j<N;j++) row[j]-=mean[j];
    }

    // ---- Phase 3: cov = data^T * data, upper triangle, k outermost ----
    // Zero the full cov matrix first (we accumulate into it).
    memset(cov, 0, (size_t)((long)N*N)*sizeof(double));

    for(int ii=0; ii<N; ii+=TI){
        int iimax = ii+TI<N ? ii+TI : N;
        // j tiles: start at ii because only upper triangle (j>=i) is needed.
        for(int jj=ii; jj<N; jj+=TJ){
            int jjmax = jj+TJ<N ? jj+TJ : N;
            // Accumulate rank-1 updates over all k for this (i-block, j-block).
            for(int k=0; k<N; k++){
                const double* restrict row = data + (long)k*N;
                for(int i=ii; i<iimax; i++){
                    double di = row[i];
                    double* restrict crow = cov + (long)i*N;
                    // For the diagonal i-block tile, j must be >= i.
                    int j0 = jj>i ? jj : i;
                    for(int j=j0; j<jjmax; j++){
                        crow[j] += di*row[j];
                    }
                }
            }
        }
    }

    // ---- Mirror upper triangle to lower triangle ----
    for(int i=0;i<N;i++){
        double* restrict crow = cov + (long)i*N;
        for(int j=i+1;j<N;j++){
            cov[(long)j*N+i] = crow[j];
        }
    }
}
