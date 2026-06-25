# mvt optimization rationale

Core transform (all regimes): **FUSE the two passes into one (i,j) nest** so each `A[i,j]` is
loaded exactly once and drives both updates — `x1[i] += A[i,j]*y1[j]` and (via index swap of
pass 2's `A[j,i]`) `x2[j] += A[i,j]*y2[i]`. This eliminates the second, transposed/stride-N
sweep over A entirely: A traffic (the N²-dominant term) is halved and the cache-hostile column
read `A[j,i]` is gone. Then **tile both i and j** (`step T`, `affine.min` upper bounds so it is
correct for any N) to keep the `x2[jj..]`/`y1[jj..]` slices cache-resident while sweeping i; the
running `x1[i]` sum is carried in an `iter_args` register and `fastmath<fast>` lets llc pipeline
the scalar reduction (benign reassociation, still allclose rtol=1e-6).

Per regime: small `opt_small` T=128, medium `opt_medium`/`opt.mlir` T=256, large `opt_large` T=512
(larger N benefits from longer A row-segment streaming once the j-tile vectors fit cache).
**dmd hint** steered the structure: its spatial model ranked fused+tiled (variant B) lowest in
data moved (~0.13–0.16x of ref at medium/large) and flagged that the scattered `x2[j]` update has
a reuse distance growing like N — which is exactly the N-scaling term tiling the j-loop collapses;
it also anchored T≈128–512 to L1/L2 residency of the four live vector slices. Predicted average
speedup vs ref ≈ 2–3x (traffic-bound; llc targets generic scalar aarch64 so the win is locality,
not SIMD). Verified correct at N=96,130,513,1000,1537 incl. non-tile-multiples.
