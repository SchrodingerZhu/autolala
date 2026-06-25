// 3mm: E=A*B ; F=C*D ; G=E*F  (all N x N, row-major double)
// Optimized via tiled ikj matmul with i-register-blocking (4 rows at a time).
// Inner j loop is unit-stride over B and the output, enabling vectorization;
// A[i,k] are loop-invariant scalars broadcast across the j stream.

#include <string.h>

#define BK 256   // K-block: streams BK rows of B into cache per pass
#define BJ 512   // J-block: keeps a panel of the output / B rows hot

// One tiled matmul: Out = L * R   (all N x N, row-major).
// Out must be zeroed by the caller (we accumulate into it).
static void mm(int N,
               const double* restrict L,
               const double* restrict R,
               double* restrict Out)
{
    for (int kk = 0; kk < N; kk += BK) {
        int kmax = kk + BK; if (kmax > N) kmax = N;
        for (int jj = 0; jj < N; jj += BJ) {
            int jmax = jj + BJ; if (jmax > N) jmax = N;

            int i = 0;
            // Process 4 rows of the output at once for register reuse of R rows.
            for (; i + 4 <= N; i += 4) {
                const double* restrict L0 = L + (size_t)(i + 0) * N;
                const double* restrict L1 = L + (size_t)(i + 1) * N;
                const double* restrict L2 = L + (size_t)(i + 2) * N;
                const double* restrict L3 = L + (size_t)(i + 3) * N;
                double* restrict O0 = Out + (size_t)(i + 0) * N;
                double* restrict O1 = Out + (size_t)(i + 1) * N;
                double* restrict O2 = Out + (size_t)(i + 2) * N;
                double* restrict O3 = Out + (size_t)(i + 3) * N;

                for (int k = kk; k < kmax; k++) {
                    const double* restrict Rk = R + (size_t)k * N;
                    double a0 = L0[k], a1 = L1[k], a2 = L2[k], a3 = L3[k];
                    for (int j = jj; j < jmax; j++) {
                        double b = Rk[j];
                        O0[j] += a0 * b;
                        O1[j] += a1 * b;
                        O2[j] += a2 * b;
                        O3[j] += a3 * b;
                    }
                }
            }
            // Remainder rows.
            for (; i < N; i++) {
                const double* restrict Li = L + (size_t)i * N;
                double* restrict Oi = Out + (size_t)i * N;
                for (int k = kk; k < kmax; k++) {
                    double a = Li[k];
                    const double* restrict Rk = R + (size_t)k * N;
                    for (int j = jj; j < jmax; j++) {
                        Oi[j] += a * Rk[j];
                    }
                }
            }
        }
    }
}

void kernel(int N, double* A, double* B, double* C, double* D,
            double* E, double* F, double* G)
{
    size_t n2 = (size_t)N * (size_t)N * sizeof(double);
    memset(E, 0, n2);
    memset(F, 0, n2);
    memset(G, 0, n2);

    mm(N, A, B, E);   // E = A * B
    mm(N, C, D, F);   // F = C * D
    mm(N, E, F, G);   // G = E * F
}
