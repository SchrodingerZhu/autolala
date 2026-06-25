# covariance optimization rationale

**Transformations**
1. **Mean & centering made row-wise.** Ref reads `data[i*N+j]` column-strided (stride N, a cache miss per element). I accumulate means by streaming each contiguous row into a `mean[]` accumulator, and fuse centering as a second contiguous row pass. Sequential access + full cache-line use + vectorizable.
2. **Covariance loop reorder (k outermost) + tiling.** `cov[i][j]=Σ_k data[k][i]·data[k][j]` is `dataᵀ·data`. Ref's innermost `k` loop strides both operands by N (two column walks → catastrophic for caches). Hoisting `k` to the outer loop turns it into rank-1 updates: `data[k][*]` is read once contiguously and broadcast/multiplied across a resident `cov` tile. I tile `(i,j)` (TI=64, TJ=256) so the active cov block fits in L1/L2 and is reused over all `k`.
3. **Symmetry exploited.** Only the upper triangle `j>=i` is computed; the lower triangle is mirrored in one cheap pass. Halves the FLOPs of the O(N³) part.
4. **Boundary-safe:** all tile loops use min-clamped bounds, so any N (including non-multiples of TI/TJ) is handled by natural remainder iterations. `restrict` enables aggressive vectorization.

**Why faster:** eliminates column-strided misses everywhere, keeps the cov tile cache-resident across the k-reduction (O(N) reuse), and the contiguous inner j-loop AVX/FMA-vectorizes cleanly.

**Predicted speedup vs ref:** roughly 8–20× at N∈[768,2048] — dominated by removing transposed cache misses in the O(N³) kernel, plus the ~2× from triangle symmetry and improved vectorization.
