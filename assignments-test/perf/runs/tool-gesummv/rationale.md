# gesummv optimization rationale

## Kernel
`y = 1.1*(A*x) + 0.9*(B*x)`, A and B are N×N row-major, N in [2048, 8192].
Each element of A and B is read exactly once; x[j] is the only reusable datum.

## Transformations
1. **Keep the A/B passes fused (single sweep).** Each of A and B is streamed
   once — the 2·N² compulsory traffic is irreducible. Fusing keeps x hot for
   both matrix reads in the same iteration.
2. **Unroll the outer i loop by 4 with register accumulation.** Every loaded
   `x[j]` feeds 4 rows of A and 4 rows of B, cutting x's re-streaming traffic
   ~4×. This is the analyzer's key finding: the bottleneck is x being
   re-fetched every row (dominant reuse-distance region `RD = 3N+3`,
   count ≈ N²); the `i` loop carries the reuse, so reducing how often x is
   pulled is the lever.
3. **8 independent FMA accumulators** (4 for A, 4 for B) hide FMA latency and
   keep the AVX FMA units busy; the unit-stride inner `j` loop vectorizes
   cleanly and is perfectly prefetcher-friendly.
4. **Remainder loop** for `N % 4` rows, and `restrict` on all pointers so the
   compiler knows the streams don't alias.

## Why faster
The reference reloads x once per row and exposes only 2 accumulators, so it is
limited by FMA latency and by x cache traffic. i-unrolling amortizes x loads
across 4 rows and gives wide ILP, turning the inner loop into a
bandwidth/throughput-bound stream over A and B.

## Analyzer's role
The `dmd` affine analyst confirmed: i-tiling alone is a no-op and 2D tiling is
worse — the reuse to capture lives on the `i` dimension as x[j] reuse, not on a
j-block that already fits in L2 (x is 16–64 KB, L2-resident for all N in range).
That steered me to i-unrolling (which the model effectively rewards) rather than
the j-tiling its block-size=1 model nominally favored, and to keep A/B fused so
x stays hot for both. Predicted speedup ≈ 1.5–2.5× over the reference,
dominated by ILP and reduced x traffic.

## Correctness
Verified `numpy.allclose`-style (rtol 1e-6, atol 1e-9) at
N = 13, 257, 2048, 3000, 4096, 5001, 8192 — all pass for both `y` and `tmp`.
Only safe reassociation (the j-reduction split across accumulators) is used.
