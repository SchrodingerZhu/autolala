// mvt: x1 += A*y1 ; x2 += A^T*y2
//
// Key transformation: the reference's second pass reads A column-major
// (A[j*N+i]), streaming the whole NxN matrix a second time with a
// cache-hostile stride. We rewrite that pass by swapping the roles of i/j so
// that A is *always* accessed row-major: for element A[i*N+j],
//     x1[i] += A[i*N+j]*y1[j]   (pass 1, unchanged)
//     x2[j] += A[i*N+j]*y2[i]   (pass 2, rewritten -- no transposed access)
// Both updates use the SAME A[i*N+j] load, so the two passes fuse into one
// sweep of A (each element read exactly once instead of twice).
//
// We then tile the (i,j) iteration space so the working set per tile fits in
// cache. y1/x2 are indexed by j (reused across the i dimension of a tile);
// y2/x1 are indexed by i. Tiling along j bounds those vector footprints.
// Remainder iterations are handled by min-clamped tile bounds, so any N works.

#define TI 64
#define TJ 256

void kernel(int N, double* restrict A, double* restrict x1, double* restrict x2,
            double* restrict y1, double* restrict y2) {
    for (int ii = 0; ii < N; ii += TI) {
        int iimax = ii + TI < N ? ii + TI : N;
        for (int jj = 0; jj < N; jj += TJ) {
            int jjmax = jj + TJ < N ? jj + TJ : N;
            for (int i = ii; i < iimax; i++) {
                const double* restrict Ai = A + (long)i * N;
                double s1 = x1[i];          // pass-1 accumulator (scalar in register)
                double y2i = y2[i];         // pass-2 broadcast operand
                for (int j = jj; j < jjmax; j++) {
                    double a = Ai[j];       // single row-major load serves both passes
                    s1   += a * y1[j];      // x1[i] += A[i,j]*y1[j]
                    x2[j] += a * y2i;       // x2[j] += A[i,j]*y2[i]  (was A[j,i]*y2[j])
                }
                x1[i] = s1;
            }
        }
    }
}
