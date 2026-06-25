#include <string.h>

// gemver:
//   Phase 1: A += u1*v1^T + u2*v2^T
//   Phase 2: x = 1.1 * A^T * y          (column sweep of A in ref -> transposed GEMV)
//   Phase 3: x += z
//   Phase 4: w = 1.2 * A * x            (row sweep of A)
//
// Strategy:
//   * Fuse Phase 1 into Phase 2. We walk A row-by-row (cache friendly,
//     contiguous, vectorizable). For each row i we:
//       - update the row in place: A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]
//       - accumulate its contribution to x: x[j] += 1.1*y[i]*A[i][j]
//     This turns the transposed (column) access of Phase 2 into a sequence
//     of row sweeps with an x[] accumulator, eliminating the strided
//     N*N cache-miss pattern, and lets A be updated and consumed in a
//     single streaming pass over memory.
//   * Phase 3 (x += z) is folded into the x initialization.
//   * Phase 4 is a standard row-wise GEMV with scalar accumulation and
//     register blocking over multiple rows to reuse x[] from registers.

void kernel(int N, double* restrict A, double* restrict u1, double* restrict v1,
            double* restrict u2, double* restrict v2, double* restrict w,
            double* restrict x, double* restrict y, double* restrict z) {

    // ---- Phases 1 + 2 fused, plus Phase 3 ----
    // ref keeps the incoming x (xi = x[i]) and adds 1.1*A^T*y, then adds z.
    // So initialize x[j] = x_orig[j] + z[j] (Phase 3 folded in).
    for (int j = 0; j < N; j++) x[j] += z[j];

    for (int i = 0; i < N; i++) {
        const double a = u1[i];
        const double b = u2[i];
        const double yi = 1.1 * y[i];
        double* restrict Ai = A + (long)i * N;
        for (int j = 0; j < N; j++) {
            double aij = Ai[j] + a * v1[j] + b * v2[j];
            Ai[j] = aij;
            x[j] += yi * aij;
        }
    }

    // ---- Phase 4: w = 1.2 * A * x  (row GEMV) ----
    // 4-row register blocking: load x[j] once, reuse across 4 rows.
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        const double* restrict A0 = A + (long)(i + 0) * N;
        const double* restrict A1 = A + (long)(i + 1) * N;
        const double* restrict A2 = A + (long)(i + 2) * N;
        const double* restrict A3 = A + (long)(i + 3) * N;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (int j = 0; j < N; j++) {
            double xj = x[j];
            s0 += A0[j] * xj;
            s1 += A1[j] * xj;
            s2 += A2[j] * xj;
            s3 += A3[j] * xj;
        }
        w[i + 0] = 1.2 * s0;
        w[i + 1] = 1.2 * s1;
        w[i + 2] = 1.2 * s2;
        w[i + 3] = 1.2 * s3;
    }
    for (; i < N; i++) {
        const double* restrict Ai = A + (long)i * N;
        double s = 0.0;
        for (int j = 0; j < N; j++) s += Ai[j] * x[j];
        w[i] = 1.2 * s;
    }
}
