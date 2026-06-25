# matmul optimization rationale

Core transform (all regimes): **i-k-j interchange + unroll-and-jam i x4 + cache tiling of k and j.**
The naive i-j-k order loads `B[k,j]` with column stride (cache-miss per FMA). Hoisting k above j
makes the innermost j-loop stream `B[k,j]` and `C[i,j]` stride-1 (clean `llc -O3` vectorization),
with `A[i,k]` loop-invariant in j (kept in a register). Unrolling i by 4 reuses each `B[k,j]` load
across 4 C rows (4 FMAs/load), cutting B/C traffic ~4x in the hot loop. A scalar tail handles N%4.

- **small** (192-384): k-tile=512 (=> single k-block, so C is read/written exactly once), j-tile=512.
  Matrices fit in L2/L3; tiling overhead avoided, just interchange + unroll-and-jam.
- **medium** (512-1024): k-tile=256, j-tile=512. Keeps the B-panel + A-strip resident in L2 while
  reused across all i-rows; C re-streamed only N/256 times.
- **large** (1152-1536): k-tile=128, j-tile=512 (also `opt.mlir` default). Tighter k blocking keeps
  the working B-panel in L2 for the much larger N; trades a few extra C sweeps for far fewer B misses.

All bounds use `affine.min` so any N (incl. non-multiples of tile/unroll) is correct (verified
allclose at N=193,257,384,513,767,1023,1153,1281,1535). Predicted avg speedup vs ref: ~4-8x
(largest gains at medium/large where the naive column-strided B dominates; ~3-4x at small).
