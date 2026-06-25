# 2mm Tiling Rationale

## Kernel
PolyBench `2mm`: two chained GEMMs.
1. `tmp[i][j] = alpha * sum_k A[i][k] * B[k][j]`  (i x j x k over ni, nj, nk)
2. `D[i][j] = beta * D[i][j] + sum_k tmp[i][k] * C[k][j]`  (i x j x k over ni, nl, nj)

## Transformation
Loop tiling (blocking) of both GEMMs with tile size 32 on all three loop
dimensions (i, j, k). The scalar-init statements that originally sat between
loop levels were hoisted into their own full passes over the (ii,jj) tile so
they execute before the k-accumulation, preserving the original order of
initialization vs. accumulation:
- `tmp[i][j] = 0.0` becomes a 32x32 init sweep over each (ii,jj) tile before the kk loop.
- `D[i][j] *= beta` becomes a 32x32 scale sweep over each (ii,jj) tile before the kk loop.

No loads/stores were removed, no arithmetic changed, and the same memrefs and
read/write sets are preserved (only reordered). The `{dmd.extract}` attribute
remains on the single outermost `affine.for %loop_once`.

## Why this cuts data movement
The untiled GEMM streams an entire B (or C) column-panel from memory for every
single (i,j) pair. With problem dimension N, each of the N x N output elements
touches an N-length row of A and an N-length column of B, and the reused B/A
data has reuse distance ~O(N^2) elements — far larger than cache, so essentially
every inner-loop access misses once the matrices exceed cache size. Total
traffic scales as O(N^3).

Tiling restricts the working set of the innermost three loops to three 32x32
blocks (A-block, B-block, tmp-block), ~3 * 32 * 32 * 8 B = 24 KB, which fits in
L1/L2. Each loaded block element is reused 32 times (once per element of the
orthogonal tile dimension) before eviction, so the reuse distance drops from
O(N^2) to O(32^2). Main-memory traffic for each GEMM falls from ~O(N^3) to
~O(N^3 / 32) for the streamed operands, i.e. the dominant capacity-miss term is
reduced by roughly the tile factor.

## Predicted improvement
For matrices that substantially exceed cache, predicted data-movement reduction
factor is on the order of the tile size, ~8x-30x depending on N relative to
cache (closer to the tile-size factor of 32 for large N where capacity misses
dominate; lower for moderate N where some reuse already fit in cache). I expect
a conservative analyzer-measured reduction in the ~8x-16x range.
