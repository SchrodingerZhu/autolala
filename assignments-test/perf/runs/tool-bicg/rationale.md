# bicg optimization rationale

## Computation
`q = A*p` and `s = A^T*r`, fused over a single row-major sweep of A (N x N).

## Bottleneck
In the reference (`for i { for j }`), the vector `s[j]` (and `p[j]`) is swept in
full for every `i`. For N in [2048, 8192], A is far larger than cache and is read
exactly once regardless of schedule, but `s`/`p` get re-streamed N times. The DMD
affine analyzer confirmed the dominant cost is the `s[j]`/`p[j]` reuse, whose reuse
distance scales as ~3N (region `value=3N+4` over ~2N^2 accesses). A is read once
(compulsory floor N^2), so tiling A buys nothing.

## Transformation
Tile only the inner `j` loop, loop order `jt -> i -> j` (tile width TJ = 64).
- A slice `s[jt..jt+TJ)` + `p[jt..jt+TJ)` (~1 KB) stays L1-resident across the
  whole `i`-sweep, capping the reuse distance at the tile width.
- A is still traversed in unit stride (`A[i*N+j]`), preserving spatial locality.
- Because `jt` is outermost, `q[i]` no longer completes in one pass, so it
  accumulates into the `q` array across j-tiles (zeroed on the first tile)
  instead of a scalar register. `s` keeps reference semantics (`s[j] +=`,
  caller pre-zeros), and the j ordering is identical, so results are bit-exact.

## Analyzer involvement
Yes. The DMD analyst compared baseline, j-tiled, i-tiled, and 2D-tiled variants.
Under a realistic 64-byte cache line (block=8) j-tiling cut modeled data movement
~2x (e.g. N=8192: 1.30e9 -> 6.58e8), while i-tiling regressed (column-strided A).
j-tile T in 32-128 was flat; T=64 chosen as L1-resident sweet spot.

## Predicted speedup
~1.5-2x wall-clock for large N (memory-traffic bound), bit-exact output
(maxdiff = 0 across N = 1,2,7,64,65,127,200,513,1000 vs reference).
