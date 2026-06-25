# matmul — optimization rationale

Kernel: `C += A*B`, all NxN f64. Reference is naive `i-j-k` (B[k,j] strided,
no working-set bound). All variants use **i-k-j interchange**: innermost `j`
is unit-stride for both `B[k,j]` and `C[i,j]`, and `A[i,k]` is invariant in
the j-loop so it stays in a register. Correctness for any N via `affine.min`
clamps on tile upper bounds (verified all-close at N=96 and N=130).

## Per regime
- **small (192-384)** `opt_small.mlir`: i-k-j interchange + light `ii` tile of
  32, no j/k tiling. Matrices are nearly cache-resident, so heavy 3D tiling
  only adds loop/min overhead; the light i-block keeps active C rows hot.
- **medium (512-1024)** `opt_medium.mlir` (= `opt.mlir`): full 3D tiling
  `ii=64, kk=256, jj=256`. The (kk x jj) B-block is reused across the ii-tile.
- **large (1152-1536)** `opt_large.mlir`: 3D tiling `ii=64, kk=128, jj=256`.
  Smaller `kk` shrinks the B-block to 128x256x8B = 256KB so it stays
  L2-resident under heavier capacity pressure at large N.

## Why faster
Naive order leaves reuse distance ~ N (full row/matrix working set). Tiling
all three dims caps the reuse distance to the tile constant: the dominant
data-movement term drops from RD~N (effectively ~N^4) to RD~const (~N^3).
Interchange adds unit-stride inner access + register reuse of A.

## dmd hint
The analyzer confirmed only the *trend*: interchange alone is ~neutral on the
leading data-movement term, while tiling all three dims collapses every
`sqrt(...)` reuse-distance factor from `sqrt(N^2+...)` to `sqrt(const)`. That
steered the structure toward 3D tiling for medium/large; exact tile sizes are
hardware-cache choices the symbolic model can't predict.

## Predicted average speedup
Roughly 3-6x over naive (largest gains at large N where the naive RD~N term
dominates; small N is more modest since it is already near cache-resident).
