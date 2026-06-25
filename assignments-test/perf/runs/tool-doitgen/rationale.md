# doitgen optimization rationale

## What the kernel is
For each `(r,q)` row (flattened `rq`, with `M = N*N` such rows), doitgen computes a
vector-matrix product `out[p] = sum_t A[rq][t] * C4[t][p]`, then overwrites
`A[rq][:] = out[:]`. The `C4` matrix (`N x N`, up to 512 KB at N=256) is **shared**
across all `M = N*N` rows — that shared reuse is the locality bottleneck.

## Transformations applied
1. **Loop interchange to i-t-p order** (p innermost). In the reference, the inner `t`
   loop walks a *column* of `C4` (stride N — one useful element per cache line) and
   re-streams a C4 column for every `p`. With `p` innermost, both the `C4` row and the
   output row are unit-stride, and `A[rq][t]` is loop-invariant across `p`. The
   analyzer measured this as a ~2x data-movement win from spatial locality alone
   (0.505x of baseline at 64-byte lines).
2. **Tile the row loop (TI=32) and the p loop (TP=64).** With p-panels outermost inside
   a row block, a `N x TP` panel of `C4` stays resident in cache while all `TI` rows of
   the block stream through it. This collapses C4's reuse distance from the whole matrix
   to the panel size. Analyzer: total DMD ~0.38x of baseline (spatial model).
3. **Block output accumulation buffer.** Because the result overwrites `A[rq][:]`, and
   `A[rq][t]` is still read during the t-reduction across *all* p-panels, the per-block
   output is accumulated in a `TI x N` temp buffer and copied back only after the block
   is complete — preserving correctness. `restrict` is used so the compiler can
   vectorize the unit-stride inner loop.

## Why it's faster
The baseline pays full-C4 reuse cost (long reuse distance + strided column access) on
every one of the `N^2 * M` multiply-adds. After interchange + panel tiling, the inner
loop is unit-stride/vectorizable and the C4 traffic that dominated is served from a
cache-resident panel reused across TI rows.

## Predicted speedup
~2x from interchange/vectorization, plus the tiling drops modeled data movement to
~0.38x of baseline — overall roughly **2–3x** wall-clock on a single core for N in
[128,256] (exact factor is hardware-dependent).

## Analyzer involvement
Yes. The AutoLALA/dmd affine locality analyzer (`compare_variants`) ranked five
variants; it confirmed the interchange (`itp`) and the i+p tiling (variant 4) as the
data-movement winners (0.380x spatial DMD vs baseline) and guided the tile-direction
choice. Exact tile sizes (TI=32, TP=64) are picked for cache residency; the analyzer's
tile sizes are symbolic, so the *direction* is analyzer-backed, the exact values are
hardware-tuned.
