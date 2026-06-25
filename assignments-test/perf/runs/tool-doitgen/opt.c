#include <stdlib.h>
#include <string.h>

/*
 * doitgen, optimized for single-core locality.
 *
 * Reference (per (r,q) row, flattened rq):
 *   for p: s=0; for t: s += A[rq][t]*C4[t][p]; out[p]=s;
 *   A[rq][:] = out[:]
 *
 * This is a vector-matrix product  out[p] = sum_t A[rq][t] * C4[t][p],
 * with C4 (N x N) SHARED across all N*N rows.
 *
 * Transformations (analyzer-guided, AutoLALA/dmd):
 *  - Interchange to i-t-p order: innermost p makes C4 rows and the output
 *    row unit-stride; A[rq][t] is invariant in p. (~2x from spatial locality.)
 *  - Tile the row loop (TI) and the p loop (TP) so a (N x TP) panel of C4
 *    stays resident in cache while TI rows stream through it, collapsing
 *    C4's reuse distance from the whole matrix to the panel.
 *  - Accumulate a TI x N output block in a temp buffer so the per-row
 *    overwrite A[rq][:] = out[:] is safe (A[rq][t] is still read during
 *    the t-reduction across all p-panels).
 */

#define TI 32
#define TP 64

void kernel(int N, double* restrict A, double* restrict C4, double* restrict sum){
  (void)sum; /* reference scratch; we use our own block buffer */

  const long n = N;
  const long M = n * n;            /* number of (r,q) rows */

  /* temp output block: TI rows x N columns */
  double* restrict obuf = (double*)malloc((size_t)TI * (size_t)n * sizeof(double));
  if (!obuf) return;

  for (long ii = 0; ii < M; ii += TI){
    long ib = ii + TI < M ? TI : (M - ii);   /* rows in this block */

    /* zero the block accumulator */
    memset(obuf, 0, (size_t)ib * (size_t)n * sizeof(double));

    /* tile p into panels of width TP; a N x TP panel of C4 stays resident */
    for (long pp = 0; pp < n; pp += TP){
      long pe = pp + TP < n ? pp + TP : n;

      for (long i = 0; i < ib; i++){
        const double* restrict Arow = A + (ii + i) * n;
        double* restrict Orow = obuf + i * n;

        for (long t = 0; t < n; t++){
          double a = Arow[t];
          const double* restrict C4row = C4 + t * n;
          /* innermost p: unit stride over C4 row and output row */
          for (long p = pp; p < pe; p++){
            Orow[p] += a * C4row[p];
          }
        }
      }
    }

    /* write the completed block back into A */
    for (long i = 0; i < ib; i++){
      double* restrict Adst = A + (ii + i) * n;
      const double* restrict Osrc = obuf + i * n;
      memcpy(Adst, Osrc, (size_t)n * sizeof(double));
    }
  }

  free(obuf);
}
