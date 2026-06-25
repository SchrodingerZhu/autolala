# gemver optimization rationale

## Reference behavior
gemver makes **3 full passes over A** (N×N doubles = 32 MB at N=2048 to 512 MB at
N=8192, far larger than any cache, so each pass is a memory pass):
- L1 writes A row-wise (rank-2 update).
- L2 computes `x += 1.1·Aᵀ·y` reading A **column-wise** (`A[j*N+i]`, stride N) — the
  cache-hostile bottleneck.
- L4 computes `w = 1.2·A·x` reading A row-wise.

## Transformations (guided by the dmd affine locality analyzer)
The analyzer modeled L2 original vs interchanged vs tiled and the fused L1+L2 form:

| variant | DMD @N=8192 | exponent |
|---|---:|---:|
| L2 original (transposed) | 6.92e9 | 2.49 |
| L2 interchanged (i inner) | 5.07e8 | 2.34 |
| L2 tiled | 2.77e8 | 2.02 |

1. **Interchange L2** so the contiguous index is innermost: rewrite the transposed
   dot-product `x[i] += 1.1·A[j][i]·y[j]` as a row-wise **scatter** `x[c] += 1.1·y[r]·A[r][c]`.
   This converts column-walking (stride N, a fresh cache line per access) into
   unit-stride row streaming — the analyzer predicts ~13.6× less data movement.
2. **Fuse L1 + interchanged L2** into one row sweep over A: while row r is hot, apply
   the rank-2 update and immediately accumulate it into x. This cuts A from **3 passes
   to 2** (the floor — L4 mandatorily needs the final x).
3. **Tile the x[] strip** of the fused pass (CB=4096 cols) so the reused accumulator
   stays resident; **register-block L4** by 4 rows × 4 cols with 4 independent
   accumulators for ILP. Pre-scaling `x` avoided; w scaled once at the end.
4. `restrict` on all pointers; remainder loops handle any N (no tile-size hardcoding).

## Why faster
Halves A's memory traffic for the L1/L2 portion (2 passes vs 3) and removes the
stride-N transposed access entirely, so all A traffic is now sequential and
prefetcher-friendly. L4's row-blocking amortizes the x[] strip across 4 rows.

## Predicted speedup
Memory-bound, A-traffic dominated: ~3→2 passes plus eliminating the ~13× worse
transposed L2 stream predicts roughly **1.8–2.5× wall-clock** at large N.

The analyzer directly informed the interchange, the fusion, and the 2-pass floor.
Verified all-close (A, x, w) at N = 2048, 2049, 3000, 4099, 127, 1.
