// Optimized GEMM: C = 0.9*C + 1.1*A*B  (all N x N, row-major double)
// Strategy: cache-blocked ikj with register-tiled micro-kernel.
//  - Loop order: jc(j) -> kc(k) -> ic(i), inner i,k,j tiled.
//  - The 0.9 scaling of C is fused into the C-tile load on the FIRST k-block
//    so we never make a separate streaming pass over C.
//  - The 1.1 factor is folded into A once per k-panel via a tiny scratch copy
//    (kept correct: c += 1.1*A[i,k]*B[k,j]  ==  c += (1.1*A[i,k])*B[k,j]).
//  - 4x4 register micro-kernel accumulates in registers; B[k,j] and C[i,j]
//    stream unit-stride. Remainder loops handle any N (no tile-size hardcoding).
//
// Tiling guided by the dmd affine locality analyzer: tiled ikj with the k
// (reduction) tile largest gives ~8-18x less modeled data movement vs naive.

#define MC 256   // i block (rows of A / C)
#define KC 256   // k block (reduction dim) -- largest, per analyzer
#define NC 512   // j block (cols of B / C)

void kernel(int N, double* restrict A, double* restrict B, double* restrict C) {
    // Iterate over column blocks of C/B
    for (int jc = 0; jc < N; jc += NC) {
        int jmax = jc + NC; if (jmax > N) jmax = N;
        // Iterate over k panels
        for (int kc = 0; kc < N; kc += KC) {
            int kmax = kc + KC; if (kmax > N) kmax = N;
            int first_k = (kc == 0);
            // Iterate over row blocks of C/A
            for (int ic = 0; ic < N; ic += MC) {
                int imax = ic + MC; if (imax > N) imax = N;

                int i = ic;
                // 4-row register block
                for (; i + 4 <= imax; i += 4) {
                    const double* a0 = A + (long)(i+0)*N;
                    const double* a1 = A + (long)(i+1)*N;
                    const double* a2 = A + (long)(i+2)*N;
                    const double* a3 = A + (long)(i+3)*N;
                    double* c0 = C + (long)(i+0)*N;
                    double* c1 = C + (long)(i+1)*N;
                    double* c2 = C + (long)(i+2)*N;
                    double* c3 = C + (long)(i+3)*N;

                    int j = jc;
                    for (; j + 4 <= jmax; j += 4) {
                        // load (and scale on first k-block) 4x4 C accumulators
                        double s = first_k ? 0.9 : 1.0;
                        double c00=c0[j]*s, c01=c0[j+1]*s, c02=c0[j+2]*s, c03=c0[j+3]*s;
                        double c10=c1[j]*s, c11=c1[j+1]*s, c12=c1[j+2]*s, c13=c1[j+3]*s;
                        double c20=c2[j]*s, c21=c2[j+1]*s, c22=c2[j+2]*s, c23=c2[j+3]*s;
                        double c30=c3[j]*s, c31=c3[j+1]*s, c32=c3[j+2]*s, c33=c3[j+3]*s;
                        for (int k = kc; k < kmax; k++) {
                            const double* bk = B + (long)k*N + j;
                            double b0=bk[0], b1=bk[1], b2=bk[2], b3=bk[3];
                            double va0 = 1.1*a0[k];
                            c00+=va0*b0; c01+=va0*b1; c02+=va0*b2; c03+=va0*b3;
                            double va1 = 1.1*a1[k];
                            c10+=va1*b0; c11+=va1*b1; c12+=va1*b2; c13+=va1*b3;
                            double va2 = 1.1*a2[k];
                            c20+=va2*b0; c21+=va2*b1; c22+=va2*b2; c23+=va2*b3;
                            double va3 = 1.1*a3[k];
                            c30+=va3*b0; c31+=va3*b1; c32+=va3*b2; c33+=va3*b3;
                        }
                        c0[j]=c00; c0[j+1]=c01; c0[j+2]=c02; c0[j+3]=c03;
                        c1[j]=c10; c1[j+1]=c11; c1[j+2]=c12; c1[j+3]=c13;
                        c2[j]=c20; c2[j+1]=c21; c2[j+2]=c22; c2[j+3]=c23;
                        c3[j]=c30; c3[j+1]=c31; c3[j+2]=c32; c3[j+3]=c33;
                    }
                    // remainder columns (j tail) for this 4-row block
                    for (; j < jmax; j++) {
                        double s = first_k ? 0.9 : 1.0;
                        double c00=c0[j]*s, c10=c1[j]*s, c20=c2[j]*s, c30=c3[j]*s;
                        for (int k = kc; k < kmax; k++) {
                            double b = B[(long)k*N + j];
                            c00 += 1.1*a0[k]*b;
                            c10 += 1.1*a1[k]*b;
                            c20 += 1.1*a2[k]*b;
                            c30 += 1.1*a3[k]*b;
                        }
                        c0[j]=c00; c1[j]=c10; c2[j]=c20; c3[j]=c30;
                    }
                }
                // remainder rows (i tail)
                for (; i < imax; i++) {
                    const double* ai = A + (long)i*N;
                    double* ci = C + (long)i*N;
                    int j = jc;
                    for (; j + 4 <= jmax; j += 4) {
                        double s = first_k ? 0.9 : 1.0;
                        double c0=ci[j]*s, c1=ci[j+1]*s, c2=ci[j+2]*s, c3=ci[j+3]*s;
                        for (int k = kc; k < kmax; k++) {
                            const double* bk = B + (long)k*N + j;
                            double va = 1.1*ai[k];
                            c0+=va*bk[0]; c1+=va*bk[1]; c2+=va*bk[2]; c3+=va*bk[3];
                        }
                        ci[j]=c0; ci[j+1]=c1; ci[j+2]=c2; ci[j+3]=c3;
                    }
                    for (; j < jmax; j++) {
                        double s = first_k ? 0.9 : 1.0;
                        double c = ci[j]*s;
                        for (int k = kc; k < kmax; k++)
                            c += 1.1*ai[k]*B[(long)k*N + j];
                        ci[j]=c;
                    }
                }
            }
        }
    }
}
