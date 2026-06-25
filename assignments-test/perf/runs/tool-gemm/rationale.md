# GEMM optimization rationale

Kernel: `C = 0.9*C + 1.1*A*B`, N x N row-major double.

## Transformations
1. **Loop interchange to ikj** so the innermost `j` loop streams `B[k,j]` and
   `C[i,j]` unit-stride (full cache-line reuse) instead of the reference's `ijk`
   which walks `B` column-wise (stride-N, a fresh line per access).
2. **Cache blocking** with tiles `MC=256` (i), `KC=256` (k, the reduction dim),
   `NC=512` (j). The k tile is the largest, per the analyzer, because it is the
   contraction dimension accumulated into one C tile.
3. **4x4 register micro-kernel**: a 4-row x 4-col block of C is held in 16
   registers across the whole k panel; each `B[k,j..j+3]` and `A[i..i+3,k]` is
   loaded once and reused 4x, cutting load traffic and exposing ILP/FMA.
4. **Scaling fusion**: the reference's separate `C *= 0.9` streaming pass is
   eliminated by folding `0.9` into the C-tile load on the first k-block only.
5. **Remainder loops** on i (rows) and j (cols) handle any N, including
   non-multiples of the tile / 4-wide block sizes. No N is hardcoded.

## Why faster
Reference is memory-bound: O(N^4)-scaling reuse distance from column-strided B
plus a full extra pass over C. Tiled ikj keeps the working set L1/L2-resident and
the register micro-kernel turns the inner loop into back-to-back FMAs on data
already in registers.

## Analyzer
The dmd affine analyzer informed the choice: ikj interchange alone cut modeled
data movement to ~0.60-0.70 of naive; tiled ikj (k tile largest) reached
~0.06-0.12 of naive (~8-18x less, growing with N). It set the loop order and the
"k-tile largest, i/j tiles smaller and roughly equal" sizing direction.

## Predicted speedup
Roughly **5-12x** wall-clock over the naive reference at N in [768,2048] on a
typical machine (memory-traffic reduction plus register reuse and FMA throughput),
larger at bigger N.

## Correctness
Verified against ref.c with numpy-style all-close (rtol 1e-6, atol 1e-9) at
N = 1,2,3,7,13,64,255,257,300,768 — all exact (maxdiff 0), since per-element
accumulation order is preserved.
