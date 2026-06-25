# Optimization rationale — `matmul` (C += A*B)

## Reference baseline
The reference uses the textbook `i,j,k` order with the inner loop accumulating
`c += A[i*N+k] * B[k*N+j]`. The inner loop walks **B column-wise** (`B[k*N+j]`
with stride `N`), so every inner iteration touches a different cache line of B.
This is catastrophic for locality at N in [768, 2048] (B is 4.7–32 MB) and the
strided access defeats auto-vectorization. The reference is bound by cache misses
on B.

## Transformations applied

1. **Loop interchange to `i, k, j`.** The innermost loop becomes
   `C[i][j] += A[i][k] * B[k][j]` for all `j`. Now both `B[k][:]` and `C[i][:]`
   are accessed with **unit stride**, and `A[i][k]` is loop-invariant. This is the
   "axpy / rank-1 update" form the clang auto-vectorizer turns into packed FMA
   over contiguous data — the single biggest win.

2. **Register blocking over `i` (MR = 4 rows).** Each loaded vector `B[k][:]` is
   reused by 4 independent accumulator streams (`C0..C3`), giving 4 FMAs per B
   load (compute/load ratio 4x higher) and 4 independent dependency chains to
   hide FMA latency and saturate the FMA units.

3. **Cache blocking over `k` (KC = 256) and `j` (NC = 512).** A `KC x NC` panel of
   B (~1 MB) is reused across every 4-row strip of the current `i`-sweep while it
   stays hot in L2. `C` strips and the panel of B fit working set into cache,
   converting B's traffic from O(N^3) DRAM reads to mostly cache hits.

4. **`restrict` qualifiers** on A, B, C and the row pointers tell the compiler the
   strips don't alias, enabling vectorization and keeping accumulators in
   registers across the inner loop.

5. **Remainder loops** handle any `N`: a scalar-row tail covers `N % MR != 0`, and
   the `kmax/jmax = min(.., N)` clamps handle partial K/J blocks. No size is
   hardcoded; verified correct for N = 1,2,3,5,7,16,17,33,100,257,769.

## Why faster
- Strided B access (1 useful double per cache line) becomes unit-stride,
  vectorized loads (8 doubles per 2 cache lines).
- Cache blocking keeps the reused B panel resident, eliminating the dominant
  DRAM traffic of the reference.
- 4-way register blocking maximizes FMA throughput and hides latency.

## Predicted speedup
At N in [768, 2048], reference is memory/latency bound while this version is
compute-bound on vectorized FMAs with a hot cache. Expected **~8–20x** faster
(typically ~10x), increasing with N as the reference's cache thrashing worsens.
