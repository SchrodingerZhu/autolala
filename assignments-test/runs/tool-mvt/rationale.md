# mvt optimization rationale

## Kernel
`mvt` runs two independent matrix-vector products over a shared NxN matrix `A`:

- Nest 1: `x1[i] += A[i][j] * y_1[j]`  — reads `A` **row-major** (good locality)
- Nest 2: `x2[i] += A[j][i] * y_2[j]`  — reads `A` **column-major / transposed** (poor locality)

`A` is NxN and does not fit in cache, so the original streams the whole matrix
**twice**, and the second pass pays a transposed (strided) access penalty: every
`A[j][i]` load touches a fresh cache line.

## Transformations applied
1. **Loop fusion.** Rename nest 2's loops (`i <-> j`), which is a pure renaming
   and leaves its result unchanged: `x2[j] += A[i][j] * y_2[i]` is identical to
   `x2[i] += A[j][i] * y_2[j]`. After renaming, *both* nests traverse `A` as
   `A[i][j]`, so they can be fused into one `(i, j)` rectangle. Fusion is legal:
   the two nests share only the read-only matrix `A`; their outputs (`x1`, `x2`)
   and input vectors (`y_1`, `y_2`) are disjoint, so there is no cross-nest
   dependence.
2. **32x32 tiling** of the fused `(i, j)` iteration space. Tiling bounds the
   reuse distance of the `x2` / `y_1` / `y_2` vector elements to a single 32x32
   tile, keeping them cache-resident across the inner sweeps.

## Why this cuts data movement
- **Fusion**: `A` is now read **once** instead of twice. The expensive
  transposed second pass over `A` is eliminated entirely — every `A[i][j]` load
  serves both the `x1` and the `x2` update.
- **Tiling**: within a 32x32 block the small working set (a strip of `x1`, a
  strip of `x2`, and slices of `y_1`/`y_2`) stays in cache, capping the largest
  reuse-distance contributions instead of letting them span the full N.

## Analyzer (DMD) verification
Confirmed with `analyze_mlir` (attr `dmd.extract`, block_size 64). All variants
extract successfully; total accesses identical (`8*N^2 + 5*N`), confirming
semantics-preserving (same set of loads/stores, reordered).

| N    | Original    | Fused       | Fused + tiled (this file) |
|------|-------------|-------------|---------------------------|
| 256  | 781,758     | 578,023     | 479,848                   |
| 1024 | 12,561,319  | 9,387,160   | 7,914,817                 |
| 4096 | 202,779,747 | 152,334,641 | 127,969,424               |

The tiled+fused variant also has a ~6x smaller sqrt-weighted `N^2.5` leading
coefficient and the smallest `N^2` body coefficient (~7.6 vs ~11.9 for the
original).

## Predicted improvement
**~1.58-1.63x data-movement reduction** vs the original (1.63x at N=256,
converging to ~1.58x for large N). Fusion alone gives ~1.33x; tiling on top adds
another ~1.19x.
