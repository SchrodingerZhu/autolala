# doitgen optimization rationale

## What the kernel is
For each `(r,q)`: `out[p] = sum_t A[r][q][t] * C4[t][p]`. Flattening the `M = N*N`
pairs `(r,q)` into a row index, this is exactly a GEMM `B = Aflat(M×N) · C4(N×N)`
followed by `Aflat <- B`, with `C4` shared across all rows.

## Transformations
1. **t/p loop interchange (GEMV -> AXPY accumulation).** Reference accesses
   `C4[t*N+p]` with the `t`-loop innermost, i.e. column-major, stride-N reads of
   C4 — cache-hostile and poorly vectorizable. I swap to make `p` innermost and
   accumulate `acc[p] += A_t * C4[t][p]`. Now C4 is streamed row-wise (stride-1,
   fully SIMD-vectorizable over p), and `A[..t]` is a broadcast scalar.
2. **rq-blocking (GEMV -> GEMM reuse, MB=4 rows).** Each loaded C4 row `C4[t][:]`
   is reused for MB=4 output rows in registers, cutting C4 memory traffic ~4× and
   keeping C4 hot in L1/L2. The 4 independent accumulator streams also hide FMA
   latency (ILP).
3. **Temp accumulators.** Per-block `acc` buffer holds results so `A[r][q][t]` is
   never clobbered before the t-reduction reads it; written back after.
4. Remainder loop over leftover rows; `restrict` on all pointers.

## Why faster / predicted speedup
Eliminates strided C4 access (now stride-1 + vectorized), reduces C4 traffic ~4×,
and exposes 4-way ILP. Vs the scalar strided reference I expect roughly **4–8×**
on a wide-SIMD AVX2/AVX-512 core for N in [128,256]. Verified bit-exact (maxdiff=0)
at N = 128,129,150,200,255,256.
