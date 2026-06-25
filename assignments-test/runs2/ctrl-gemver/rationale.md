# GEMVER optimization rationale

## Kernel
PolyBench `gemver`, four loop nests sharing the N×N matrix `A`:
1. `A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]`  (rank-2 update, A read+write, row-major)
2. `x[i] += beta * A[j][i] * y[j]`         (A read TRANSPOSED — `A[j][i]`)
3. `x[i] += z[i]`                          (vector, 1-D)
4. `w[i] += alpha * A[i][j] * x[j]`        (A read row-major)

A is the dominant footprint (N² vs. O(N) for every vector). It is streamed
three times across the program, so its traffic governs data movement.

## Transformations applied

### 1. Loop interchange on nest 2 (the key fix)
In the original, nest 2 has `i` outer / `j` inner but indexes `A[j][i]`.
With row-major storage, advancing the inner index `j` strides by N elements —
a full column walk per inner step. Every inner iteration touches a different
cache line, so A's spatial locality is destroyed (≈1 useful element per line
of 8 f64 = ~8× wasted traffic on A in this nest).

Interchanging to `j` outer / `i` inner makes the inner index `i` advance
`A[j][i]` by unit stride → contiguous, fully-used cache lines. The reduction
target `x[i]` is a running sum over `j`; summation is associative/commutative
in ordering, so reordering the `(i,j)` loops preserves the computed result.

### 2. 32×32 tiling on all three N×N nests
Each N×N nest is blocked into 32×32 tiles. A 32×32 f64 tile is 8 KiB, plus the
relevant 32-length vector slices — comfortably L1/L2 resident. Tiling bounds
the reuse distance of A within a nest to one tile, ensuring each A element/line
is brought in once per nest and reused for all its block work before eviction,
instead of relying on a full N²-element working set fitting in cache.

### 3. Access patterns preserved
Nests 1 and 4 already access `A` row-major; tiling keeps that and additionally
improves temporal locality of the streamed vectors (`v1,v2` reused across `ii`
tiles; `x[j]` reused across `ii` tiles in nest 4).

## Semantics preservation
- No loads/stores added or removed; same memrefs, same read/write sets.
- Nest-2 interchange is legal: the only loop-carried value is the `x[i]`
  reduction, which is order-independent.
- Tiling is a pure iteration-space reblocking of independent point updates.

## Predicted improvement
The headline win is nest 2: turning column-major into row-major A access
recovers ~8× of A's cache-line utilization in that nest. Tiling caps the
reuse-distance of A in the other two nests, eliminating capacity misses when
N² exceeds cache. Combined predicted data-movement reduction: roughly
**2.5×–4×** overall (dominated by the nest-2 stride fix plus reduced capacity
traffic on A across all three matrix nests), larger as N grows beyond the
cache size.
