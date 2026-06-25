# 2mm optimization rationale

`2mm` is two chained GEMMs: `tmp = 1.1*A*B`, then `D = 0.9*D + tmp*C`.

## Problem with ref.c
The reference uses `i,j,k` order. The inner k-loop reads the right operand
column-wise (`B[k*N+j]`, `C[k*N+j]`) — stride `N` — so each FMA touches a new
cache line. For N in [512,1536] the right operand (2–18 MB) does not fit in
cache, giving ~1 miss per multiply and no vectorization of the reduction.

## Transformations
1. **Loop interchange to `i,k,j`.** The inner j-loop now streams the R-row and
   the output row contiguously (unit stride), so clang auto-vectorizes it into
   packed FMAs and prefetching works.
2. **Register blocking over i (4 rows).** Each R element `r[j]` is loaded once
   and reused across 4 accumulator rows, cutting right-operand memory traffic
   ~4x and raising the FMA-to-load ratio. Remainder rows (N%4) handled by a
   scalar tail loop, so any N is correct.
3. **j cache-blocking (JB=256).** Keeps the 4 active output strips resident in
   L1/L2 while the full k-dimension is swept.
4. **Constant folding.** `1.1` is folded into the L-operand value; the `0.9*D`
   pre-scale is applied to the output strip before accumulation, removing a
   separate pass.
5. **`restrict`** on all pointers lets the compiler keep accumulators in
   registers across the inner loop.

The k-summation order is preserved, so output is bit-identical to ref
(measured maxrel = 0 at N = 1,2,7,63,64,65,512,513,777,1024).

## Predicted speedup
Removing strided right-operand misses plus 4x register reuse and SIMD FMAs
should yield roughly **4–8x** over ref at the tested sizes (larger N benefits
more, being more memory-bound in the reference).
