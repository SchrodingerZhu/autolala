# gesummv — locality optimization rationale

## Kernel

`gesummv` computes two matrix-vector products that share the same input
vector `x` and the same iteration domain:

```
for i:                       tmp[i] = 0; y[i] = 0
  for j:                     tmp[i] += A[i][j] * x[j]
                             y[i]   += B[i][j] * x[j]
  y[i] = alpha*tmp[i] + beta*y[i]
```

The A and B sweeps are already fused over `(i, j)` in the original, so the
shared `x[j]` is loaded once per inner iteration for both products. The
remaining locality problem is in the `x` vector and the matrix rows.

## Transformation applied: 2-D tiling of both i and j by 32

I tiled **both** the `i` and `j` loops with tile size 32. Because `tmp[i]`
and `y[i]` must be initialized exactly once before the j-sweep and finalized
exactly once after it, I split the per-i body into three loops over the
i-tile to keep everything affine without using `affine.if`:

1. **init** loop (`%i = 0..32`): `tmp[ii+i] = 0`, `y[ii+i] = 0`
2. **tiled accumulation** (`%jj` outer, then `%i`, then `%j`): each `(ii, jj)`
   tile sweeps a 32x32 block of A and B
3. **final** loop (`%i = 0..32`): `y[ii+i] = alpha*tmp[ii+i] + beta*y[ii+i]`

All three are wrapped under a single outer `affine.for %ii` carrying
`{dmd.extract}`.

## Why this cuts data movement

In the original (i outer, j inner, untiled), each i row re-streams the entire
`x[0..N]` vector, and `x` is reloaded N times total. Its reuse distance grows
with N, so `x` traffic is not cache-resident.

After tiling, within one `(ii, jj)` tile:

- the `x[jj..jj+32]` block (32 elements) is **reused across all 32 i rows** of
  the tile, with a reuse distance bounded by the tile footprint instead of N;
- a 32x32 block of A and a 32x32 block of B stay resident for the duration of
  the inner sweep (short reuse distance for the streamed matrix blocks);
- the accumulators `tmp[ii+i]`, `y[ii+i]` for the 32-row strip are touched
  repeatedly with bounded reuse distance.

The set of array reads/writes is unchanged — the analyzer confirms identical
total access count `8*N^2 + 5*N`; only the order (hence reuse distance) changes.

## Analyzer confirmation (block_size = 64)

| Variant | leading sqrt-weighted coeff (the `*N^2.5` term) | pure-`N^2` floor | DMD @ N=4096 |
|---|---|---|---|
| original (untiled)        | ~3.38e-3 | 11.87 | 2.03e8 |
| tile i only by 32         | ~3.38e-3 |  8.87 | 1.52e8 |
| **tile i and j by 32 (chosen)** | **~4.92e-4** | **7.63** | **1.28e8** |

The chosen variant has both the smallest dominant `N^2.5` coefficient
(~7x reduction vs. original) and the lowest `N^2` floor. It wins at every
problem size measured (N = 1024, 4096, 65536). `analyze_mlir` confirmed the
optimized file extracts cleanly and preserves `8*N^2 + 5*N` accesses.

## Predicted improvement

- **Dominant (asymptotic) term:** ~**7x** lower (~3.4e-3 -> ~4.9e-3/10 ≈ 4.9e-4).
- **Practical regime (N^2 floor dominates up to N ≈ 2.4e8):** ~**1.55x**
  reduction in data movement (e.g. 2.03e8 -> 1.28e8 at N=4096, a factor 1.59).

Net predicted data-movement reduction: roughly **1.5x–1.6x** in the realistic
problem-size range, growing toward **~7x** as N becomes very large.

## Notes / caveats

- Tile size 32 is a reasonable default for a 64-byte block; the true optimum
  depends on actual cache size and should be tuned by measurement.
- `affine.store` is modeled by the analyzer as `write` (not read-modify-write);
  since each accumulation emits an adjacent load+store on the same cell, the
  temporal reuse is still captured and the access counts match.
