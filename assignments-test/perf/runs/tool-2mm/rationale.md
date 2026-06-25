# 2mm optimization rationale

## What it computes
`tmp = 1.1*A*B` then `D = 0.9*D + tmp*C` — two chained NxN GEMMs.

## Transformations
1. **Loop interchange to ikj** inside each GEMM. The reference uses `ijk` with a
   scalar `t` accumulator, which strides B/C down columns (`B[k*N+j]` for fixed j) —
   cache-hostile. With `ikj`, the innermost loop streams contiguously over `j`, so
   `B[k,j]`/`C[k,j]` and the output row are read/written along cache lines (good
   spatial locality + autovectorizable).
2. **3D tiling (i,j,k) with TILE=64**. Blocks the re-streamed N×N footprint down to a
   tile-sized working set that stays resident in L1/L2, eliminating the dominant
   capacity misses. Remainder loops via `min(ii+TILE,N)` handle any N (incl.
   non-multiples) — verified correct at N = 1,2,63,64,65,127,200,513,777.
3. **Scalar folding**: the 1.1 is folded into `aval = 1.1*A[i,k]`; D is pre-scaled by
   0.9 in one linear pass before accumulating `tmp*C` in place. No extra math, no
   change to the computed result.

## Why faster
The inner kernel becomes a contiguous AXPY over a register-broadcast scalar
(`Crow[j] += aval*Brow[j]`), which clang -O3 -march=native vectorizes (FMA, AVX).
Tiling keeps the active sub-matrices in cache so memory traffic scales with the tile
working set instead of re-streaming whole matrices per output.

## Analyzer-informed
Yes. The dmd affine analyzer (DSL variants, `scale` method) showed: interchange alone
is a no-op on the leading data-movement term (coeff 2.0 → 2.0), but tiling collapses
the leading N⁴ coefficient ~14× (2.0 → 0.06 at T=32), with the total-traffic optimum
around T≈32–64. I chose T=64 (close to optimum, larger contiguous vector runs, fits
L2 with the L1-resident inner panel). Fusion of the two GEMMs was modeled as a
secondary (~1.4×) win and skipped to keep the code simple and the materialized `tmp`
identical to the reference.

## Predicted speedup
Roughly 4–10× over the naive ijk reference at N in [512,1536], dominated by the tiling
(cache) win plus inner-loop vectorization; exact factor is hardware-dependent.
