# syrk optimization rationale

Core idea (all regimes): reorder the compute nest from the reference `i-k-j` to **`i-j-k`
with `k` innermost as a register reduction**. In `i-j-k` the inner `k` loop reads `A[i,:]`
and `A[j,:]` — both *contiguous* rows — and accumulates `C[i,j]` in a single register/vector,
instead of the reference's strided `A[j,k]` column walk. `llc -O3` vectorizes the k-reduction.
Lower-triangle (`j<=i`) is preserved via the `#tri(i)=i+1` upper bound (and `min(jt+32,i+1)`
on tiled j). All tile bounds use `affine.min`, so any N is correct (verified at N=96,130).

- **small (opt_small.mlir):** plain `i-j-k`, scale fused into the k-reduction seed
  (`C=0.9*C` as the iter_arg init, no separate scale pass). A's rows fit in L1/L2 at N<=384,
  so tiling overhead would only hurt.
- **medium (opt_medium.mlir):** tile `i` and `j` by 32, keep `k` full so `C[i,j]` never spills.
  `A[jt-block,:]` is reused across the whole i-block (~256KB working set, fits L2).
- **large (opt_large.mlir):** 3-D tile `it/jt=64`, `kt=256` to cap the A working set
  (~256KB) in L2 when full rows (len up to 1664) no longer fit; separate scale pass.
- **opt.mlir:** single-file fallback = the medium tiled version (robust across regimes).

Predicted average speedup vs ref: ~2.5-4x (driven by contiguous A access + L2 reuse;
biggest gains at medium/large where the reference thrashes on strided `A[j,k]`).
