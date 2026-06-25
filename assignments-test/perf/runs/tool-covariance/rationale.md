# Covariance kernel optimization rationale

## What the reference does
1. Column means of `data` (N×N row-major).
2. Mean-center columns in place: `data[i][j] -= mean[j]`.
3. Symmetric Gram matrix `cov[i][j] = sum_k data[k][i]*data[k][j]` — the O(N³) cost.

The reference's inner reduction reads `data[k*N+i]` and `data[k*N+j]`: **column-wise,
stride-N accesses**. At a 64-byte cache line (8 doubles) only 1 of 8 loaded doubles is
used, and the retained working set across the reduction is O(N²), so it thrashes cache.

## Transformations applied
- **Transpose + fuse.** Build a mean-centered transposed copy `dataT[r][k] = centered
  data[k][r]` while centering `data` in place. Then
  `cov[i][j] = sum_k dataT[i][k]*dataT[j][k]` — i.e. `dataT @ dataT^T` with **both
  operands streamed unit-stride over k**. This is a one-time O(N²) cost vs O(N³) work.
- **3-D cache tiling** over (i, j, k) with tiles TI=TJ=128, TK=256; cov accumulated
  across k-tiles. Pulls the reuse distance down to the tile working set.
- **Symmetry at tile granularity.** Skip tile blocks wholly below the diagonal and start
  the inner j at `max(jj, i)`; a final pass mirrors the upper triangle into the lower.
  Halves the access stream (3·N³ → 3/2·N³).
- **4×4 register micro-kernel.** A 4×4 accumulator block keeps `cov` sub-tile in
  registers, reuses each loaded `dataT` value 4× (raises arithmetic intensity, lets the
  vectorizer/FMA units saturate). Full remainder loops handle any N (tested 1..1300).

## Why it is faster
Removes the stride-N column penalty (full cache-line utilization), shrinks the reuse
working set to cache-resident tiles, halves the FLOPs/traffic via symmetry, and raises
compute intensity with register blocking so the FMA pipeline is the bottleneck, not memory.

## Predicted speedup
~3.5–6× over the reference at N in [768, 2048] (analyzer showed ~3.6× lower data movement
for tiled+symmetric vs reference at N=2048; register blocking and line-utilization add more
on real hardware).

## Analyzer
Yes. The dmd affine locality analyzer ranked variants by symbolic data movement: it showed
the untiled transpose alone leaves an O(N²) retained footprint (~N³·√N), that 3-D tiling
with T≈128 minimizes traffic across N, and that tile-level symmetry cleanly halves the
access stream. It also flagged that the i·k·j order on this kernel is *worse* (both
operands indexed by k), so I kept the inner reduction over k with j in registers.
