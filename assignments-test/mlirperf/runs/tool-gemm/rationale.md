# GEMM optimization rationale

All three regimes share one structure: pre-scale `C *= 0.9` (O(N^2)), then a tiled
**i-k-j** matmul (j innermost => B and C stream contiguously, so `llc -O3` vectorizes the
inner loop) with **unroll-and-jam x4 on i**: each `B[k,j]` load feeds 4 C rows, quadrupling
arithmetic reuse of the streamed B — the one lever `llc` cannot synthesize. The UJ path
covers `i < (N floordiv 4)*4`; the <=3 bottom rows use a plain tiled i-k-j nest, so the
kernel is correct for ANY N (verified all-close at N=96,130,193,257,384,511,1153).

- **small (N 192-384):** iT=64, kT=jT=384 — matrices are L2-resident, so tiling is near
  no-op; UJ-x4 + i-k-j is the whole win. ~1.5-2x.
- **medium (N 512-1024):** iT=64, kT=256, jT=128 — a kxj B-panel (~256KB) stays L2-resident
  while the 4-row register block streams from L1. ~2.5-3.5x.
- **large (N 1152-1536):** iT=64, kT=256, jT=96 — tighter j-panel (~192KB) keeps the working
  set in L2 as matrices blow past it; tiling pays off most here. ~3-4x.

Predicted average speedup ~2.5-3x. The dmd hint steered only the TREND: i-k-j unconditionally
beats i-j-k (contiguous streams), tiling removes the N^2 reuse-distance term, and register
blocking (unroll-and-jam) was the single best variant; the relative DMD win grows with N, which
is why I tighten the j-panel from large->small N. I treated dmd as direction-only and chose
absolute tile sizes from L1/L2 capacity, not from its (lower-bound-free) magnitudes.
