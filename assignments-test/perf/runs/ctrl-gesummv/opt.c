void kernel(int N, double* restrict A, double* restrict B,
            double* restrict x, double* restrict y, double* restrict tmp){
    const double alpha = 1.1, beta = 0.9;
    int i = 0;

    /* Process 4 rows of A and B at a time.
       x[j] is loaded once and reused across the 4 rows, while A and B
       are streamed sequentially (cache-optimal). Two accumulators per
       row break the FMA dependency chain so the vector units stay busy. */
    for (; i + 4 <= N; i += 4) {
        const double* a0 = A + (long)(i+0)*N;
        const double* a1 = A + (long)(i+1)*N;
        const double* a2 = A + (long)(i+2)*N;
        const double* a3 = A + (long)(i+3)*N;
        const double* b0 = B + (long)(i+0)*N;
        const double* b1 = B + (long)(i+1)*N;
        const double* b2 = B + (long)(i+2)*N;
        const double* b3 = B + (long)(i+3)*N;

        double ta0 = 0, ta1 = 0, ta2 = 0, ta3 = 0;
        double tb0 = 0, tb1 = 0, tb2 = 0, tb3 = 0;

        for (int j = 0; j < N; j++) {
            double xj = x[j];
            ta0 += a0[j] * xj;  tb0 += b0[j] * xj;
            ta1 += a1[j] * xj;  tb1 += b1[j] * xj;
            ta2 += a2[j] * xj;  tb2 += b2[j] * xj;
            ta3 += a3[j] * xj;  tb3 += b3[j] * xj;
        }

        tmp[i+0] = ta0;  y[i+0] = alpha*ta0 + beta*tb0;
        tmp[i+1] = ta1;  y[i+1] = alpha*ta1 + beta*tb1;
        tmp[i+2] = ta2;  y[i+2] = alpha*ta2 + beta*tb2;
        tmp[i+3] = ta3;  y[i+3] = alpha*ta3 + beta*tb3;
    }

    /* Remainder rows. */
    for (; i < N; i++) {
        const double* a = A + (long)i*N;
        const double* b = B + (long)i*N;
        double ta = 0, tb = 0;
        for (int j = 0; j < N; j++) {
            double xj = x[j];
            ta += a[j] * xj;
            tb += b[j] * xj;
        }
        tmp[i] = ta;
        y[i] = alpha*ta + beta*tb;
    }
}
