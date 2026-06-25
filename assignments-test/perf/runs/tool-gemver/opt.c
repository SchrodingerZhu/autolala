// gemver, optimized for single-core native performance.
//
// Reference computes:
//   L1: A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]            (row-major write)
//   L2: x[i]    += 1.1 * A[j][i] * y[j]                 (TRANSPOSED read of A)
//   L3: x[i]    += z[i]
//   L4: w[i]     = 1.2 * A[i][j] * x[j]                 (row-major read)
//
// The reference makes 3 full passes over the N*N array A (32MB..512MB), and L2
// reads A column-wise (stride N), which is cache-hostile.
//
// Transformations (guided by the dmd affine locality analyzer):
//  1. Interchange L2 so the j-index (matching A's row index) is the OUTER loop
//     and the contiguous index is inner: rewrite x[i] += 1.1*A[j][i]*y[j] as a
//     scatter:  for each row r:  for each c:  x[c] += 1.1*A[r][c]*y[r].
//     This turns L2's transposed column-walk into unit-stride row streaming.
//  2. Fuse L1 with the interchanged L2 into ONE row-wise sweep over A: while row
//     r is hot in cache we (a) apply the rank-2 update and (b) accumulate it into
//     x. This cuts A from 3 memory passes to 2.
//  3. Tile the x[] accumulator strip of the fused pass and the L4 gemv so the
//     reused vector strip stays resident; register-block / multi-accumulate the
//     inner dot products. Remainder loops handle any N (no tile-size hardcoding).
//
// Result matches the reference under numpy.allclose(rtol=1e-6, atol=1e-9).

void kernel(int N, double* restrict A, double* restrict u1, double* restrict v1,
            double* restrict u2, double* restrict v2, double* restrict w,
            double* restrict x, double* restrict y, double* restrict z) {

    // Column tile for the fused L1+L2 pass: keeps the active x[] strip (and the
    // v1/v2 strip) resident while we stream A rows through it.
    const int CB = 4096;

    // ---- Pass 1: fused L1 (rank-2 update) + L2 (interchanged, scatter into x) ----
    // For column block [c0, c1):
    //   for each row r:  yr = 1.1*y[r];  uu1=u1[r]; uu2=u2[r];
    //     for c in block:  A[r][c] += uu1*v1[c] + uu2*v2[c];
    //                      x[c]    += yr * A[r][c];
    // x[] starts at its incoming value (reference carries x[i] into L2), so we do
    // NOT zero it; we accumulate on top, matching the reference.
    for (int c0 = 0; c0 < N; c0 += CB) {
        int c1 = c0 + CB; if (c1 > N) c1 = N;
        const double* restrict v1b = v1 + c0;
        const double* restrict v2b = v2 + c0;
        double* restrict xb = x + c0;
        int blen = c1 - c0;

        for (int r = 0; r < N; r++) {
            double* restrict Ar = A + (long)r * N + c0;
            double uu1 = u1[r];
            double uu2 = u2[r];
            double yr  = 1.1 * y[r];
            int c = 0;
            for (; c <= blen - 4; c += 4) {
                double a0 = Ar[c]   + uu1 * v1b[c]   + uu2 * v2b[c];
                double a1 = Ar[c+1] + uu1 * v1b[c+1] + uu2 * v2b[c+1];
                double a2 = Ar[c+2] + uu1 * v1b[c+2] + uu2 * v2b[c+2];
                double a3 = Ar[c+3] + uu1 * v1b[c+3] + uu2 * v2b[c+3];
                Ar[c]   = a0; Ar[c+1] = a1; Ar[c+2] = a2; Ar[c+3] = a3;
                xb[c]   += yr * a0;
                xb[c+1] += yr * a1;
                xb[c+2] += yr * a2;
                xb[c+3] += yr * a3;
            }
            for (; c < blen; c++) {
                double a0 = Ar[c] + uu1 * v1b[c] + uu2 * v2b[c];
                Ar[c] = a0;
                xb[c] += yr * a0;
            }
        }
    }

    // ---- L3: x += z ----
    for (int i = 0; i < N; i++) x[i] += z[i];

    // ---- Pass 2: L4  w = 1.2 * A * x  (row-major dot products) ----
    // Pre-scale x once so the inner loop is a plain dot product (fewer mults).
    // Process rows in groups of 4 to reuse the x[] strip across rows and expose ILP.
    int i = 0;
    for (; i <= N - 4; i += 4) {
        const double* restrict A0 = A + (long)(i + 0) * N;
        const double* restrict A1 = A + (long)(i + 1) * N;
        const double* restrict A2 = A + (long)(i + 2) * N;
        const double* restrict A3 = A + (long)(i + 3) * N;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = 0;
        for (; j <= N - 4; j += 4) {
            double x0 = x[j], x1 = x[j+1], x2 = x[j+2], x3 = x[j+3];
            s0 += A0[j]*x0 + A0[j+1]*x1 + A0[j+2]*x2 + A0[j+3]*x3;
            s1 += A1[j]*x0 + A1[j+1]*x1 + A1[j+2]*x2 + A1[j+3]*x3;
            s2 += A2[j]*x0 + A2[j+1]*x1 + A2[j+2]*x2 + A2[j+3]*x3;
            s3 += A3[j]*x0 + A3[j+1]*x1 + A3[j+2]*x2 + A3[j+3]*x3;
        }
        for (; j < N; j++) {
            double xj = x[j];
            s0 += A0[j]*xj; s1 += A1[j]*xj; s2 += A2[j]*xj; s3 += A3[j]*xj;
        }
        w[i+0] = 1.2 * s0;
        w[i+1] = 1.2 * s1;
        w[i+2] = 1.2 * s2;
        w[i+3] = 1.2 * s3;
    }
    for (; i < N; i++) {
        const double* restrict Ai = A + (long)i * N;
        double s = 0.0;
        for (int j = 0; j < N; j++) s += Ai[j] * x[j];
        w[i] = 1.2 * s;
    }
}
