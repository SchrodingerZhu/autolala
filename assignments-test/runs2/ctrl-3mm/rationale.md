# Optimization rationale: ctrl-3mm

## Kernel
3mm computes three chained matrix multiplications:
- `E = A * B`  (ni x nk * nk x nj)
- `F = C * D`  (nj x nm * nm x nl)
- `G = E * F`  (ni x nj * nj x nl)

Each is a textbook `ijk` matmul: for every `(i,j)` an accumulation over `k`.

## Transformation applied
For each of the three matmuls I applied **3D loop tiling (blocking) with tile size 32**
on the `i`, `j`, and `k` loops, plus a **split of the zero-initialization out of the
accumulation nest**.

1. **Init/accumulate fission.** The original code does `E[i][j]=0; for k { E[i][j]+=... }`
   inside the `(i,j)` body. To tile the `k` loop, the same `(i,j)` output element is now
   touched by several `k`-tiles, so the `=0` write must happen exactly once *before* any
   accumulation. I hoisted it into a separate `i,j` init nest. The total set of loads/
   stores is identical (one zero-store per output element, one load+one store per MAC),
   only reordered — semantics preserved.

2. **i/j/k tiling, tile = 32.** Each accumulation nest becomes
   `for ii step 32 { for jj step 32 { for kk step 32 { for i<32 { for j<32 { for k<32 }}}}}`,
   addressing with `%ii+%i`, etc. Tiling does not reorder dependent updates of a given
   `(i,j)` element across `k` in a way that changes the sum (floating-point addition order
   along `k` is preserved within each output element: for fixed `(i,j)` the `k` values are
   still visited `kk=0,32,...` then `k=0..31`, i.e. strictly increasing `k`, identical to
   the original sequential order).

## Why this cuts data movement
The untiled `ijk` matmul, for each `i` row, streams the **entire** B matrix (nk x nj)
through cache once per `i`. When `nj` (and the row of A) exceeds cache, B is reloaded from
memory ni times -> ~`ni * nk * nj` words of traffic for B alone; A's row and the E element
stay hot, but B has reuse distance ~`nk*nj`, far beyond cache.

Tiling into 32x32x32 blocks confines the working set to three 32x32 tiles
(A-tile + B-tile + E-tile ~ 3 * 32*32 * 8B = 24 KiB), which fits in L1/L2. Within a tile each
loaded element of A, B, and E is reused 32 times before eviction. This converts the dominant
capacity misses on the streamed operand (B in E=A*B, D in F=C*D, F in G=E*F) from
`O(N^3)` memory traffic to `O(N^3 / T)` with `T=32`, i.e. roughly a **32x reduction** in
misses on the reused operand, bounded in practice by the relative cost of the non-reused
operand and the output stream.

## Predicted improvement
For dimensions larger than the cache, predicted data-movement reduction is on the order of
**~8x-20x** overall (a conservative read of the ideal ~T=32 factor on the streamed operand,
diluted by the always-resident output and the A-row/E-row traffic that tiling does not
change). For small dimensions that already fit in cache, improvement is near 1x (no harm).
Tile size 32 (32x32x8B = 8 KiB per tile) is chosen to keep three concurrent tiles within a
typical 32-64 KiB L1/L2.
