# atax optimization rationale

Single `opt.mlir` for all three regimes (small/medium/large).

- **Transformation:** Fuse the two i-j nests on `i` and unroll-and-jam the `i` loop by 8.
  Per row-block: phase 1 computes `T[i]=sum_j A[i,j]*x[j]` (scalar reduction, no T
  reload), then phase 2 does `y[j]+=sum_{i in block} A[i,j]*T[i]` with a tree-reduced
  combine. A tail loop (affine.min upper bound) handles N not a multiple of 8, so it is
  correct for any N (verified allclose at N=96,130).
- **Why faster:** `A` (N×N) is the only Θ(N²) array and the sole memory bottleneck.
  Fusion makes each `A[i,:]` row serve both passes, so A is streamed from DRAM **once**
  instead of twice (ref reads it in two separate sweeps) — the dominant win at every
  regime since A ≫ LLC for all tested N. Unroll-and-jam by 8 amortizes each `y[j]`
  load/store over 8 rows (secondary y/T-traffic cut); x re-reads in phase 1 are cheap
  from L2. Tree-reduced phase-2 adds shorten the per-j dependency chain for better ILP.
- **dmd hint:** The analyzer ranked FUSED+unroll-and-jam-i > FUSED > REF stably across
  N=1024–8192, with A-streaming identified as the driver and j-tiling rejected (it
  re-streams A for no y-reuse benefit). This steered me to fuse-on-i + UJ and to skip
  j-tiling. dmd only modeled UJ=4; I chose 8 for stronger y amortization (register
  pressure still fits AVX), treating the exact factor as hardware-tuned, not predicted.
- **Predicted average speedup vs ref:** ~1.5–1.8× (fusion alone ≈1.4–1.5× from halving
  A traffic, UJ/ILP adding the rest), roughly uniform across the three regimes.
