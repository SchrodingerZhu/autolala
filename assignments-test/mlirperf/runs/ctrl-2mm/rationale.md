# 2mm optimization rationale

Single `opt.mlir` for all three regimes.

- **Core transform (all regimes):** interchange both matmuls from the reference i-j-k
  (inner k strides B[k,j]/C[k,j] by N — no SIMD, cache-thrashing) to **i-k-j**, so the
  innermost loop runs over contiguous `j` in B/C and T/D. This is the GEMM-friendly order
  that lets `llc -O3` emit packed FMA over unit-stride rows. This alone removes the
  column-stride loads that dominate the naive cost.
- **Tiling (all regimes):** tile i,k by 64 and j by 256 with `affine.min` upper bounds
  (correct for any N). The 64x64 A-block stays in L1/L2 while a 64x256 B-panel is reused
  across the 64 i-rows; this caps reuse distance so the working set fits cache as N grows.
- **small (160-320):** interchange is the dominant win (data nearly fits); tiles are large
  enough that overhead is negligible.
- **medium (448-768) / large (896-1152):** tiling keeps the reused A-block and B-panel
  resident, avoiding the capacity misses the naive version suffers when a full N-row no
  longer fits — the gap over ref grows with N.
- D=0.9*D kept as a cheap O(N^2) pass (negligible vs the two O(N^3) matmuls).

**Predicted average speedup vs ref:** ~3-6x (small ~2x, medium ~4x, large ~5-6x).
