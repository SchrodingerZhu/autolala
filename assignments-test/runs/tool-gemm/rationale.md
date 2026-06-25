# GEMM Tiling — Rationale

## Kernel
PolyBench-style GEMM inside a `dmd.extract`-tagged outer loop:
- Phase 1: `C[i][j] *= beta`
- Phase 2: `C[i][j] += alpha * A[i][k] * B[k][j]`

The data-movement cost is dominated by Phase 2 (the triple loop, `4·NI·NJ·NK` accesses).

## Transformation applied
**Loop tiling/blocking (32×32×32) on both phases**, plus a loop ordering in
Phase 2 of `(ii, kk, jj, i, k, j)`:

- Phase 1 (`C *= beta`): tiled `(i, j)` into 32×32 blocks.
- Phase 2 (matmul): tiled `(i, k, j)` into 32×32×32 blocks. The tile loops are
  ordered `ii → kk → jj`, and inside a tile the order is `i → k → j`.

All loop bounds remain affine: outer tile loops use `0 to %N step 32`, inner
loops use `0 to 32`, and array subscripts use `%ii + %i` index expressions, as
required. The set of loads/stores, the memrefs, and the math are unchanged —
only the iteration order is permuted, so semantics are preserved (each
`C[i][j]` is still scaled once and accumulated over the full `k` range; the
accumulation order over `k` is unaffected because all `k` for a given `(i,j)`
are still summed, just grouped into tiles).

## Why it cuts data movement (reuse / locality)
In the original Phase 2 the inner `j` loop streams a full row of `B` and a full
column-strip of `C` for every `(i,k)`, so the *reuse distance* for a reused
element of `B[k][j]` and `C[i][j]` grows as O(NJ·NK) — both grow with the
problem size, so reused data is evicted from cache long before it is touched
again.

Tiling restricts the working set of each innermost 32×32×32 block to
~3·32² doubles (one tile each of A, B, C), which fits in cache. As a result:
- **A[i][k]** tile is reused across the `j` sweep,
- **B[k][j]** tile is reused across the `i` sweep,
- **C[i][j]** tile is reused (read-modify-write accumulated) across the `k` sweep.

This collapses almost all reuse distances from O(NJ·NK) down to bounded
constants on the order of tile² (≈ 32²), so reuses are served from cache
instead of refetched from memory.

## Analyzer verification (mcp__dmd / dmd-cli, block_size=64)
- Both variants **extract** under `dmd.extract` with no error.
- **Identical** total access count: `4·NI·NJ·NK + 2·NI·NJ` (nothing added/removed).
- Leading-order DMD term stays N⁴ (same asymptotic class, as expected for a
  constant-factor locality win), but the **leading coefficient drops**:
  - Original:  1/512  ≈ 1.95e-3
  - Optimized: 31/524288 ≈ 5.91e-5
  - Ratio ≈ **33×** lower leading-order data movement.
- Mechanism confirmed in the reuse-distance breakdown: the original's dominant
  reuse distances scale as O(NJ·NK); the tiled version caps them at bounded
  ~tile² constants.

## Predicted improvement factor
**≈ 33× reduction** in leading-order data movement (constant-factor, not an
asymptotic-order change). Actual speedup depends on whether the 32×32×32 tiles
fit the target cache; the tile size can be tuned to cache capacity.
