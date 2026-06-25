# 3mm locality optimization rationale

## Kernel
PolyBench `3mm`: three chained matrix multiplications inside one `{dmd.extract}` region.
- `E = A*B`   (i in ni, j in nj, k in nk)
- `F = C*D`   (i in nj, j in nl, k in nm)
- `G = E*F`   (i in ni, j in nl, k in nj)

Each is a naive `ijk` matmul: the accumulator `E[i][j]`/`F[i][j]`/`G[i][j]` is initialized to 0
once per `(i,j)`, then read-modify-written every `k` step.

## Bottleneck (from analyzer)
The analyzer (`dmd.extract`, block_size 64) shows total data movement scales as **Θ(N⁴)** when
all dims ~ N. The leading DMD term of each matmul has the form

    work_volume (Θ(N³)) · sqrt( panel_footprint / block_size )

where the streamed right-hand operand of each product — `B[k][j]` in `E=A*B`,
`D[k][j]` in `F=C*D`, `F[k][j]` in `G=E*F` — is re-streamed in full for every outer `i`.
Its reuse distance therefore grows with the **whole N² operand matrix**:
`RD ≈ (1/64)·(second_dim · accum_dim) + …`. That `sqrt(N²)` factor inside every leading
term is what lifts the Θ(N³) arithmetic volume to Θ(N⁴) of data movement. The accumulator
itself sits at reuse distance 1 (free); the cost is entirely the N²-footprint operand reuse.

## Transformation applied: 32×32×32 loop tiling of all three matmuls
For each of the three products I tile the `i`, `j`, and `k` loops with tile size 32
(`affine.for %ii = 0 to N step 32 { affine.for %i = 0 to 32 { … %ii + %i … } }`),
ordered `ii, jj, [init i,j], kk, i, j, k`.

Because the `k`/reduction loop is now split into an outer `kk` tile loop, the per-`(i,j)`
zero-initialization (`E[i][j] = 0.0`) can no longer live inside the `k` body — it would
re-zero the accumulator on every `kk` tile. It is hoisted into a separate `(i,j)` loop nest
that runs once per `(ii,jj)` tile, before the `kk` reduction. Initializing the whole output
tile to 0 first and then accumulating over all `kk` is **exactly equivalent** to the original
"zero once, then accumulate over all k" — same writes, same reads, same arithmetic, same
final values. No loads/stores removed, no domain shrunk, same memrefs.

## Why it cuts data movement
Tiling caps the retained working set of the inner `(i,j,k)` body at three 32×32 tiles
(~3·1024 elements), independent of N. The operand's reuse distance inside the sqrt drops
from Θ(N²) to Θ(tile²) = Θ(1024), a **constant**. The leading DMD term becomes

    Θ(N³) · sqrt(1024 / 64) = Θ(N³) · 4   →   total Θ(N³)

instead of Θ(N⁴). Asymptotically the data-movement reduction factor is **≈ N / (8)** (more
precisely `sqrt(N²/tile²) = N/32` on the operand footprint, partly offset by the fixed
sqrt(tile²/block) = 4 spatial factor). For typical PolyBench sizes (N ≈ 1024) this is a
large multi-×–to–order-of-magnitude reduction in modeled data movement; the dominant
Θ(N⁴) growth term is removed entirely, leaving Θ(N³).

## Predicted improvement
Leading-order DMD changes from **Θ(N⁴)** to **Θ(N³)**. Predicted improvement factor
≈ **N/8** asymptotically (the analyzer-confirmed leading-term ratio), i.e. it grows with
problem size; for N≈1024 this is roughly two orders of magnitude on the modeled traffic.
