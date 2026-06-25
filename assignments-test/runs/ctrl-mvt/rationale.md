# mvt locality optimization

## Kernel
PolyBench `mvt`: two matrix-vector products over the same matrix A.
- Nest 1: `x1[i] += A[i][j] * y_1[j]`  — A read **row-major** `A[i][j]`.
- Nest 2: `x2[i] += A[j][i] * y_2[j]`  — A read **column-major** `A[j][i]`.

A is N×N and dominates traffic (O(N²)) versus the O(N) vectors. Each nest
sweeps all of A exactly once at the element level; the locality question is
how many cache lines / how much capacity is touched per useful access.

## Transformation applied
**2-D tiling (blocking) of both loop nests** with tile size 32 on the `i`
and `j` index ranges. Each nest becomes a 4-deep nest
`(ii step 32)(jj step 32)(i 0..32)(j 0..32)` using `%ii+%i`, `%jj+%j`
index expressions. No fusion, interchange of semantics, or domain change —
the iteration set and every load/store is identical to the original, only
the visit order is reordered (legal because each nest's body has no
loop-carried dependence except the associative reduction into x1[i]/x2[i],
which tiling preserves since the i index is partitioned, not reordered
across its own reduction).

## Why it cuts data movement
- **Nest 2 (the bottleneck):** the original `A[j][i]` access is column-major
  with stride N. With row-major storage every inner `j` step jumps a full
  row, so each cache line of A is evicted long before its other elements are
  reused — effectively touching ~N lines repeatedly. Tiling restricts the
  working set to a 32×32 block of A (8 KB for f64) plus 32-element slices of
  x2 and y_2, all of which fit in L1. Each A line is now fully consumed
  before eviction, converting capacity/conflict misses into a single cold
  miss per line.
- **Nest 1:** already row-major, but tiling additionally keeps the
  `y_1[jj..jj+32]` slice and `x1[ii..ii+32]` slice resident across the block,
  improving vector reuse without harming A's streaming.
- **Reuse distances:** the longest reuse distance for A drops from O(N²)
  (full-matrix sweep before any line revisit in the strided nest) and for the
  vectors from O(N) to O(tile²)=O(1024), i.e. bounded and cache-resident
  independent of N.

## Predicted improvement
Dominated by nest 2. For a typical L1 (32 KB) with N large enough that a
matrix column/working set blows the cache, the column-major nest moves on the
order of 8 (lines fetched per useful f64 with 64-byte lines, all-miss) ×
versus ~1 after tiling. Net data-movement reduction across both nests is
conservatively **~3–6×**, growing with N (the untiled column-major nest
degrades with N while the tiled version stays flat). Predicted factor: **~4×**.
