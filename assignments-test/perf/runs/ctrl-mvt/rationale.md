# mvt optimization rationale

## Reference behavior
`ref.c` does two separate full sweeps of the N×N matrix A:
1. `x1[i] += A[i][j]*y1[j]`  — row-major, cache-friendly.
2. `x2[i] += A[j][i]*y2[j]`  — **column-major (A^T) access**, one cache line touched
   per inner iteration with stride `N*8` bytes. For N in [2048,8192] this thrashes the
   cache: ~N² cache-line fetches, and A (32MB–512MB) is read **twice** total.

## Transformation: loop fusion via index renaming
Rename the second loop's indices (i↔j): `x2[j] += A[i][j]*y2[i]`. Now both updates use
the **same** matrix element `A[i][j]`, so a single row-major sweep computes both:
```
for i: for j:  a = A[i][j];  s1 += a*y1[j];  x2[j] += a*y2[i];
```
- A is streamed **once** (half the matrix traffic) in pure row-major order — the
  transposed access is gone entirely.
- `x1[i]` is a scalar reduction (kept in a register `s1`).
- `x2[j]`/`y1[j]` are sequential, vectorizable; `y2[i]` is a loop-invariant broadcast.

## Tiling
A 256×256 (i,j) block tiles the sweep so the touched `x2[]`, `y1[]` slices (2KB each)
stay in L1/L2 while rows of A stream through; `x1[i]` accumulates across j-blocks and
`x2[j]` across i-blocks (associativity-only reassociation, all-close preserved).
Remainder handled by `min` bounds, so any N works. `restrict` enables SIMD.

## Predicted speedup
Eliminating the transposed pass + halving A's memory traffic: this kernel is
memory-bandwidth bound at these sizes, so expect roughly **2.5–4× faster** than ref
(2× from single-pass A traffic, plus extra from removing the strided-A cache misses).
