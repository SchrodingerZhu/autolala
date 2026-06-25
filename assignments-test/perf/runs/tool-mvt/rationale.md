# MVT optimization rationale

## What ref.c does
- Pass 1: `x1[i] += A[i,j]*y1[j]`  — A read row-major (good).
- Pass 2: `x2[i] += A[j,i]*y2[j]` — A read **column-major** (`A[j*N+i]`), and A
  (32 MB–512 MB for N in [2048,8192]) is streamed a **second** time. The
  transposed stride is the dominant cost: every inner step jumps N*8 bytes, so
  each cache line delivers one useful double, and the matrix exceeds all caches.

## Transformations applied
1. **Role-swap pass 2 to make A row-major.** The element `A[i,j]` contributes
   `A[i,j]*y2[i]` to `x2[j]`. So instead of reading `A[j,i]` into `x2[i]`, we
   accumulate `x2[j] += A[i,j]*y2[i]`. A is now always accessed row-major.
2. **Fuse the two passes.** Both updates use the *same* `A[i,j]` load, so a
   single sweep over A serves x1 and x2. A is read **once** instead of twice
   (total array traffic 6N^2 -> 5N^2). Fusion is safe: y1/y2 are read-only
   inputs and x1/x2 are write-only outputs, so there is no cross-pass
   dependence.
3. **Tile the (i,j) space (TI=64, TJ=256)** so the per-tile working set (a strip
   of x2[j]/y1[j] of length TJ plus scalar i-state) stays cache-resident,
   bounding reuse distance. `x1[i]` and `y2[i]` are held in registers across the
   j-loop; `x1` is written once per row. Remainder iterations use min-clamped
   tile bounds, so any N (incl. non-multiples of 64/256) is handled.

## Why faster
- Eliminates the strided column traversal -> full cache-line utilization and
  hardware-prefetcher-friendly sequential access.
- Halves A's memory traffic via fusion (A is the bandwidth bottleneck).
- Tiling keeps the reused vectors in L1/L2 despite A being far larger than cache.

## Analyzer (dmd) guidance
The AutoLALA/dmd locality analyzer informed the choice. Reference DMD scales
~N^2.5 (the transposed `A[j,i]` produces an N^2-scale reuse distance). The fused
variant drops to ~0.19x of reference (worst RD collapses from ~N^2/8 to linear
~3N/8); tiling the fused nest reaches ~0.116x at N=8192. Among the tile sizes
compared (32/64/128/256/512), B=64 was the model's best on the i-tile dimension;
I kept j-tiling wider (256) to amortize the x2 read/modify/write while staying
cache-resident — the model's tile size is only directional (no capacity cutoff),
so this is a small hardware-aware adjustment around its recommendation.

## Correctness
Verified bit-exact (max diff 0) vs ref.c across N = 1,2,7,63,64,65,127,200,257,
300,513,1025 — covering tile-boundary and remainder cases. Builds clean with
`clang -O3 -march=native -funroll-loops`.
