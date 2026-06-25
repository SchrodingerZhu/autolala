// bicg: q = A*p ; s = A^T*r  fused over one row-major sweep of A.
//
// Optimization: tile the inner j-loop (loop order jt -> i -> j).
// In the reference, the vector s[j] is swept in full for every i, so its
// working set (and that of p[j]) does not fit in cache for large N and is
// re-streamed N times. By blocking j into tiles of width TJ, the slice
// s[jt..jt+TJ) and p[jt..jt+TJ) stays L1-resident across the entire i-sweep,
// while A is still read in unit-stride order (one full pass, optimal).
//
// Because jt is outermost, q[i] is no longer completed in a single inner pass,
// so it must accumulate across j-tiles (q[i] += ...) rather than into a scalar.
// We zero q in the first j-tile. s keeps the reference semantics (s[j] += ...),
// i.e. the caller pre-zeros s exactly as for ref.c.

void kernel(int N, double* restrict A, double* restrict p, double* restrict r,
            double* restrict q, double* restrict s){
    const int TJ = 64;

    for (int jt = 0; jt < N; jt += TJ){
        int jend = jt + TJ;
        if (jend > N) jend = N;
        const int first = (jt == 0);

        for (int i = 0; i < N; i++){
            const double ri = r[i];
            const double* restrict Ai = A + (long)i * N;
            double qi = first ? 0.0 : q[i];

            // inner j tile: s[j] and p[j] slices are cache-resident here
            for (int j = jt; j < jend; j++){
                double aij = Ai[j];
                s[j] += ri * aij;
                qi   += aij * p[j];
            }
            q[i] = qi;
        }
    }
}
