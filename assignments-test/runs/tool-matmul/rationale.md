# Optimization Rationale — matmul (ijk)

## Baseline
The input is a classic `i,j,k` matmul: `C[i,j] += A[i,k] * B[k,j]`.
The inner `k` loop reuses `C[i,j]` (scalar reuse, good), but the `B[k,j]`
plane and `A[i,k]` row are re-streamed: as the iteration advances, the
reuse distance for `B` and `C` grows with the array dimensions. The DMD
analyzer reports the dominant term with a reuse distance of `~N*K/64`
inside the sqrt:

  Baseline DMD ≈ Θ( M·N·K · √(N·K) )   (empirical cubic exponent ≈ n^3.55)

So data movement per useful flop grows without bound as the matrix grows —
the working set spills out of any fixed cache.

## Transformation applied
**3D loop tiling / blocking of all three loops (i, j, k) by a tile of 32.**

```
for ii in 0..M step 32
  for jj in 0..N step 32
    for kk in 0..K step 32
      for i in 0..32
        for j in 0..32
          for k in 0..32
            C[ii+i, jj+j] += A[ii+i, kk+k] * B[kk+k, jj+j]
```

This is purely a reordering of the same iteration space (a loop
permutation of a strip-mined nest). The exact same set of loads/stores on
A, B, C is performed — only the order changes — so it is
semantics-preserving (the `+=` reduction on `C[i,j]` is associative over
`k`, and the tiling keeps each `(i,j)` accumulation's `k` order monotone).

## Why it cuts data movement
Tiling confines the active working set of every iteration of the inner
3-deep nest to three 32×32 blocks (one block each of A, B, C ≈ 3·32·32·8 B
≈ 24 KB). Once a tile is resident, every element of A, B, C in that tile is
reused 32 times before eviction, and the reuse distance is bounded by the
**constant tile footprint** instead of growing with N·K. The analyzer
confirms the dominant `MNK·√(...)` terms collapse to a constant
sqrt-prefactor (the `√(NK)` factor disappears):

  Tiled DMD ≈ Θ( M·N·K )   (empirical cubic exponent ≈ n^3.08, flat DMD/MNK)

A partially-tiled variant (j,k only, i left un-tiled) was also tested and
is strictly worse — Θ(MNK·√K) — because the un-tiled `i` re-streams A rows
and C; full 3D tiling is required to bound the reuse distance.

## Predicted improvement
Because the asymptotics differ (√(NK) vs constant), the speedup grows with
problem size:

| M=N=K | baseline / tiled |
|------:|-----------------:|
|   256 |  ~5.6× |
|   512 |  ~7.3× |
|  1024 |  ~9.8× |
|  2048 | ~13.5× |
|  4096 | ~18.6× |

Predicted data-movement reduction: **~6–19×** for square matrices in the
256–4096 range, increasing with size. Tile size 32 is illustrative; the
optimal tile should be tuned to the real L1/L2 capacity, but any
constant-bounded tile yields the Θ(MNK) asymptote.
