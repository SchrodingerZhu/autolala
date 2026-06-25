#include <string.h>

// atax: tmp = A*x ; y = A^T*tmp   (A is N x N, row-major)
//
// Key transformation: FUSE the two passes over A so each row A[i, :] is
// streamed from DRAM exactly once instead of twice. A is N*N doubles
// (32MB..512MB for N in [2048,8192]) and far exceeds cache, so the
// reference's second full sweep of A is pure DRAM traffic. After computing
// tmp[i] from row i (full reduction over j), we immediately reuse that same
// row -- still hot in L1/L2 -- for the y[j] += A[i*N+j]*tmp[i] scatter.
//
// The analyzer (dmd) showed this drops A's data-movement term from ~N^3
// (reuse distance ~N^2 across the second sweep) to ~N^2.25.
void kernel(int N, double* restrict A, double* restrict x,
            double* restrict y, double* restrict tmp){
    // Match the reference, which accumulates into y[] (and reads tmp[i]
    // via +=). Reference assumes these start zeroed; zero them to be safe
    // and match semantics exactly.
    memset(y, 0, (size_t)N * sizeof(double));

    for(int i = 0; i < N; i++){
        const double* restrict Ai = A + (size_t)i * N;

        // Pass 1 over row i: tmp[i] = dot(A[i,:], x).
        // Scalar accumulators help the compiler vectorize/unroll the reduction.
        double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
        int j = 0;
        for(; j + 4 <= N; j += 4){
            t0 += Ai[j+0] * x[j+0];
            t1 += Ai[j+1] * x[j+1];
            t2 += Ai[j+2] * x[j+2];
            t3 += Ai[j+3] * x[j+3];
        }
        double t = (t0 + t1) + (t2 + t3);
        for(; j < N; j++) t += Ai[j] * x[j];
        tmp[i] = t;

        // Pass 2 over the SAME (now cache-resident) row i:
        // y[j] += A[i,j] * t. axpy into y, which stays live across all i.
        for(j = 0; j + 4 <= N; j += 4){
            y[j+0] += Ai[j+0] * t;
            y[j+1] += Ai[j+1] * t;
            y[j+2] += Ai[j+2] * t;
            y[j+3] += Ai[j+3] * t;
        }
        for(; j < N; j++) y[j] += Ai[j] * t;
    }
}
