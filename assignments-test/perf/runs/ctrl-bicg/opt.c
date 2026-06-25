// bicg: q = A*p and s = A^T*r, fused over one sweep of A (row-major, N x N).
//
// Optimization: register-block the outer i-loop by 4 rows. For each column j
// inside the blocked inner loop we load s[j] and p[j] ONCE and use them across
// all 4 rows. This amortizes the read-modify-write traffic on s[] and the load
// of p[] by 4x, while the four A rows stream contiguously. q is accumulated in
// scalar registers (one per row). The inner loop is vectorizer-friendly: a
// single sweep over j with independent FMAs.
void kernel(int N, double* restrict A, double* restrict p, double* restrict r,
            double* restrict q, double* restrict s) {
    // s must be zero-initialized (ref relies on caller-zeroed s; matches ref
    // which does s[j] += ... without init, so do NOT zero here to match ref).

    int i = 0;
    for (; i + 4 <= N; i += 4) {
        const double* restrict A0 = A + (long)(i + 0) * N;
        const double* restrict A1 = A + (long)(i + 1) * N;
        const double* restrict A2 = A + (long)(i + 2) * N;
        const double* restrict A3 = A + (long)(i + 3) * N;
        const double r0 = r[i + 0];
        const double r1 = r[i + 1];
        const double r2 = r[i + 2];
        const double r3 = r[i + 3];
        double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;

        for (int j = 0; j < N; j++) {
            const double pj = p[j];
            const double a0 = A0[j];
            const double a1 = A1[j];
            const double a2 = A2[j];
            const double a3 = A3[j];

            // s = A^T * r : accumulate the 4 row contributions into s[j]
            s[j] += r0 * a0 + r1 * a1 + r2 * a2 + r3 * a3;

            // q = A * p : per-row dot products
            q0 += a0 * pj;
            q1 += a1 * pj;
            q2 += a2 * pj;
            q3 += a3 * pj;
        }
        q[i + 0] = q0;
        q[i + 1] = q1;
        q[i + 2] = q2;
        q[i + 3] = q3;
    }

    // Remainder rows (N not a multiple of 4).
    for (; i < N; i++) {
        const double* restrict Ai = A + (long)i * N;
        const double ri = r[i];
        double qi = 0.0;
        for (int j = 0; j < N; j++) {
            const double a = Ai[j];
            s[j] += ri * a;
            qi += a * p[j];
        }
        q[i] = qi;
    }
}
