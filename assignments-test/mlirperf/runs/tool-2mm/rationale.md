# 2mm optimization rationale

Single `opt.mlir` (one structure serves all three regimes, since per-band working set is N-independent).

**Transformations (all regimes, same kernel):**
- **i-k-j interchange** on both matmuls: makes `j` innermost so `B[k,j]`/`T[i,j]` (mm1) and `C[k,j]`/`D[i,j]` (mm2) stream unit-stride, and `A[i,k]`/`T[i,k]` hoist out of the `j` loop. Access stream drops from naive `3N^3` to ~`2N^3`; lets `llc -O3` vectorize the inner `j` loop cleanly.
- **Tiling** `ii=32, kk=32, jj=128`: caps each array's reuse distance at the tile footprint instead of the naive ~N^2/2 re-stream that real caches lose.
- **Fusion by i-band**: for each 32-row band, fully compute `T[band]=A[band]*B`, scale `D[band]*=0.9`, then immediately `D[band]+=T[band]*C` while that T-band is still cache-resident. This removes T's full-N^2 capacity re-stream between the two passes (only a 32xN band is live across the boundary). Correctness folds the 0.9 scale into a per-band pass; verified all-close at N=96,130 and non-tile-multiple N.

**Why faster:** combines unit-stride vectorizable inner loop + bounded reuse distance + elimination of the N^2 intermediate-array eviction traffic. Win grows with N (compulsory floor unchanged, capacity traffic killed).

**dmd hint steering:** the analyzer modeled i-k-j tiling at ~0.28/0.17/0.11x ref traffic (small/medium/large) and a further ~0.43x from fusing the two matmuls (keeping each T-band resident), and favored smaller tiles with `jj` shrunk toward 64-128. I adopted band-fusion and the 32/32/128 tiling on that trend (treating the exact tile numbers as direction only, not a runtime prediction).

**Predicted avg speedup vs ref:** ~2.5-4x (larger at large N), averaged across regimes.
