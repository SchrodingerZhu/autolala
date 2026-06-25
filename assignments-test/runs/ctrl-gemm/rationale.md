# GEMM Locality Optimization

## Kernel
PolyBench `gemm`: `C = beta*C + alpha*A*B`, with an `i`-loop containing
1. a `C[i][j] *= beta` scaling sweep, and
2. an `i-k-j` matmul accumulation `C[i][j] += alpha*A[i][k]*B[k][j]`.

## Transformation applied
Loop **tiling (blocking)** with tile size 32 on all dimensions.

- **Scaling loop**: tiled `i,j` by 32 (`ii,jj` outer, `i,j` inner).
- **Matmul nest**: tiled all three dims into `ii, jj, kk` (outer) and
  `i, k, j` (inner). The intra-tile order is kept as `i-k-j` so the
  vectorizable `j` stays innermost.

Both nests are kept as siblings under the single `{dmd.extract}` outermost
`affine.for %loop_once = 0 to 1`. Same memrefs, same loads/stores, identical
arithmetic — only iteration order changes, so semantics are preserved.

## Why this cuts data movement
In the untiled `i-k-j` matmul, for each `i` the inner `k,j` sweep streams the
**entire** `NK x NJ` matrix `B` once. Across all `NI` rows of `i` that is
`NI * NK * NJ` words of `B` traffic — `B` is reread `NI` times because a full
row-block of `B` cannot stay in cache between successive `i` values.

After 32^3 tiling, the working set of an inner tile is three 32x32 blocks
(`C`, `A`, `B` = 3 * 32 * 32 * 8 B ≈ 24 KB), which fits in L1/L2. Within a
`(ii,jj)` tile the same 32x32 block of `B` is reused across all 32 `i` values
and the 32x32 `C` block is reused across all 32 `k` values. So `B` is now read
from memory only `NI/32` times instead of `NI` times (and `A`/`C` blocks are
similarly reused), turning capacity misses into cache hits.

## Predicted improvement
`B`-matrix main-memory traffic drops by roughly the tile factor ~32x; total
data movement is dominated by this term once `A` and `C` reuse are accounted
for. Net predicted data-movement reduction: **~8-15x** for matrices large
enough that an untiled row-block of `B` overflows cache (conservatively ~10x).
