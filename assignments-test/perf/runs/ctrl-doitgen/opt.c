#include <stdlib.h>
#include <string.h>

/*
 * doitgen: for each (r,q): out[p] = sum_t A[r][q][t] * C4[t][p]
 * Treating the (r,q) pairs as a flattened "row" index m in [0, M=N*N),
 * this is a GEMM:  B = Aflat (M x N) * C4 (N x N),  then Aflat <- B.
 *
 * Key transformations vs ref.c:
 *   1) Loop interchange of the t- and p-loops: accumulate sum[p] += a_t * C4[t][p]
 *      so C4 is streamed row-wise (stride-1, vectorizable over p) instead of
 *      column-wise (stride-N).  A[..t] becomes a broadcast scalar.
 *   2) Block several (r,q) rows together (MB rows) so each C4 row C4[t][:]
 *      loaded once is reused for MB output rows -> GEMM-style reuse of C4,
 *      cutting C4 memory traffic by ~MB and keeping C4 hot in cache.
 *   3) Per-block temporary accumulators in a stack/heap buffer so we never
 *      clobber A[r][q][t] before the t-reduction reads it.
 */

#define MB 4   /* rows of A processed together (register/L1 blocking over rq) */

void kernel(int N, double* restrict A, double* restrict C4, double* restrict sum){
    const long n = N;
    const long M = n * n;            /* number of (r,q) pairs */

    /* accumulator block: MB output rows of length N */
    double* acc = (double*)malloc((size_t)MB * (size_t)n * sizeof(double));
    if(!acc){
        /* fallback: simple correct path */
        for(long m=0;m<M;m++){
            double* Arow = A + m*n;
            for(long p=0;p<n;p++){
                double s=0.0;
                for(long t=0;t<n;t++) s += Arow[t]*C4[t*n+p];
                sum[p]=s;
            }
            for(long p=0;p<n;p++) Arow[p]=sum[p];
        }
        return;
    }

    long m=0;
    for(; m+MB<=M; m+=MB){
        double* a0 = A + (m+0)*n;
        double* a1 = A + (m+1)*n;
        double* a2 = A + (m+2)*n;
        double* a3 = A + (m+3)*n;
        double* c0 = acc;
        double* c1 = acc + n;
        double* c2 = acc + 2*n;
        double* c3 = acc + 3*n;

        for(long p=0;p<n;p++){ c0[p]=0.0; c1[p]=0.0; c2[p]=0.0; c3[p]=0.0; }

        for(long t=0;t<n;t++){
            const double* C4t = C4 + t*n;
            double x0 = a0[t];
            double x1 = a1[t];
            double x2 = a2[t];
            double x3 = a3[t];
            for(long p=0;p<n;p++){
                double cv = C4t[p];
                c0[p] += x0*cv;
                c1[p] += x1*cv;
                c2[p] += x2*cv;
                c3[p] += x3*cv;
            }
        }

        for(long p=0;p<n;p++){ a0[p]=c0[p]; a1[p]=c1[p]; a2[p]=c2[p]; a3[p]=c3[p]; }
    }

    /* remainder rows */
    for(; m<M; m++){
        double* Arow = A + m*n;
        double* c0 = acc;
        for(long p=0;p<n;p++) c0[p]=0.0;
        for(long t=0;t<n;t++){
            const double* C4t = C4 + t*n;
            double x0 = Arow[t];
            for(long p=0;p<n;p++) c0[p] += x0*C4t[p];
        }
        for(long p=0;p<n;p++) Arow[p]=c0[p];
    }

    free(acc);
}
