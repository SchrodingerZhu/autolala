# gemver optimization rationale

## Reference structure
gemver runs four sweeps over the N×N matrix A (N in [2048,8192], so A is
32 MB–512 MB — far larger than LLC; the kernel is memory-bandwidth bound):

1. `A += u1*v1^T + u2*v2^T`        (rank-2 update, row access)
2. `x = x + 1.1*A^T*y`             (**transposed** GEMV — strided column access)
3. `x += z`
4. `w = 1.2*A*x`                   (GEMV, row access)

The reference touches all of A in **three separate passes** (phase 1 writes,
phase 2 reads transposed, phase 4 reads). Phase 2's `A[j*N+i]` access is the
worst part: it walks A column-by-column with stride N, so every element load
is a fresh cache line (and TLB miss) — effectively reading the whole 32–512 MB
matrix with ~1 useful double per 64-byte line brought in.

## Transformations applied
1. **Fuse phase 1 into phase 2, and rewrite phase 2 as row sweeps.**
   Walk A one row i at a time (contiguous, vectorizable). For each row:
   - update it in place: `A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]`
   - scatter-accumulate its contribution into x: `x[j] += 1.1*y[i]*A[i][j]`,
     since `(A^T y)[j] = sum_i A[i][j]*y[i]`.
   This converts the strided column GEMV into a sequence of streaming row
   updates with an x[] accumulator vector, and **merges two full passes over
   A into one** (write+read fused). A is now touched contiguously everywhere.
2. **Fold phase 3** (`x += z`) into the x initialization (`x[j] += z[j]`),
   preserving the reference's keep-original-x semantics.
3. **Phase 4 register blocking:** process 4 rows of A per inner loop so each
   loaded `x[j]` feeds 4 multiply-adds (4 independent accumulators hide FMA
   latency and improve the flops-per-x-load ratio). Scalar tail loop handles
   the remainder when N % 4 != 0.

`restrict` on all pointers lets the vectorizer assume no aliasing.

## Why faster
- Total streaming passes over A drop from 3 (write, strided-read, read) to 2
  (fused write+read, then read). On a bandwidth-bound kernel that is the
  dominant cost.
- The eliminated strided pass alone was reading ~1 line per element with N×
  cache/TLB miss amplification; replacing it with contiguous access removes
  that entirely. x[]/y[] fit in cache, so the accumulator traffic is free.
- Both surviving sweeps are fully contiguous, SIMD-vectorizable, and
  prefetcher-friendly.

## Predicted speedup
For N in [2048,8192] (memory-bound), expect roughly **2–3.5×** vs ref: the
strided phase-2 pass (which dominated runtime via cache/TLB miss
amplification) is eliminated and phases 1+2 collapse into a single pass.
