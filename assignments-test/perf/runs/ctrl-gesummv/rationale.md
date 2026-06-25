# gesummv optimization rationale

## What it computes
`tmp[i] = A[i,:]·x`, `y[i] = 1.1*(A[i,:]·x) + 0.9*(B[i,:]·x)`. A and B are N×N
and each is read exactly once with no element reuse, so the kernel is fundamentally
**memory-bandwidth bound** at the N in range (each matrix is 32MB–512MB, far beyond cache).

## Transformations applied
1. **Row register-blocking (4 rows at a time).** The inner loop loads `x[j]` once and
   reuses it across 4 rows of A and 4 rows of B. This raises arithmetic-per-x-load and,
   more importantly, gives the compiler many independent multiply-add streams to vectorize
   and software-pipeline.
2. **Per-row independent accumulators** (`ta*`, `tb*`). Breaks the serial FMA dependency
   chain of the reference (one accumulator), so multiple FMAs are in flight and the
   pipeline never stalls on accumulator latency.
3. **`restrict` on all pointers.** Lets clang assume no aliasing between A/B/x/y/tmp and
   keep accumulators in vector registers across the whole inner loop.
4. **Preserved sequential streaming of A and B** (row-major, contiguous). This is the
   cache-/prefetch-optimal access pattern; no tiling of the matrices is useful since there
   is no temporal reuse of matrix data.
5. **Remainder loop** handles any N not divisible by 4 — no hardcoded N.

## Why faster
The reference has a single accumulator pair with a tight dependency chain that under-uses
the FMA units, and reloads `x[j]` per row. Blocking 4 rows + split accumulators saturates
the vector pipeline and amortizes x loads. Math is unchanged (verified bit-exact, maxdiff=0
for N=2048..2051).

## Predicted speedup
Compute is fully overlapped with the memory stream after this change, so runtime approaches
the streaming-bandwidth floor. Expect ~1.3–2x over the reference (larger when the reference's
dependency chain, not bandwidth, was the limiter; closer to 1.3x once bandwidth-bound).
