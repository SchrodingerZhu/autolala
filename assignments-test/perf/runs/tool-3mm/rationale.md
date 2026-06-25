# 3mm optimization rationale

## What it computes
`E=A*B`, `F=C*D`, `G=E*F` — three sequential N×N matmuls (G depends on E and F),
so the dominant cost is 3·N³ FMAs over three matmuls.

## Transformations
1. **Loop reorder ijk → ikj.** The reference uses `ijk` with a scalar dot-product,
   which strides B column-wise (`B[k*N+j]`, stride N) — cache-hostile and
   non-vectorizable. Reordering to `ikj` makes the inner `j` loop unit-stride over
   both B's row and the output row, with `A[i,k]` a loop-invariant broadcast scalar.
   This vectorizes cleanly and streams cache lines.
2. **Cache tiling (blocking) on K and J** (BK=256, BJ=512) so the reused B-panel and
   output panel stay resident, removing the N²-scaling reuse-distance term.
3. **i-register blocking (4 rows).** Four output rows are accumulated per pass, so
   each loaded `B[k,j]` is reused 4× from registers, raising arithmetic intensity and
   FMA-unit utilization. Remainder loops handle rows/cols not divisible by tile sizes.
4. **Zero + accumulate** via `memset`, letting all three matmuls share one tiled path.

## Why faster
Eliminates the strided column access of B, enables SIMD on the unit-stride inner loop,
and caps the working set so data movement drops from an ~N⁴-class reuse term to the
~N³ streaming floor. Register blocking cuts B reloads 4×.

## Analyzer
The `dmd` affine analyzer informed the choice: it showed ijk/ikj carry an N⁴-class DMD
(reuse distance ∝ full matrix), while tiled-ikj collapses to a pure ~N³ floor
(~7–10× lower DMD at N=1024), and recommended tiled-ikj with unit-stride inner `j`.

## Predicted speedup
Roughly 4–8× over the naive reference (memory-bound strided ref → vectorized,
cache-blocked, register-reusing kernel), larger at the bigger end of N∈[512,1536].

## Correctness
Verified all-close vs reference for N = 1,2,7,33,100,257,513,700 (max excess over
rtol=1e-6/atol=1e-9 tolerance = 0 across E, F, G).
