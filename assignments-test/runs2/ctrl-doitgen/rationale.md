# doitgen optimization rationale

## Kernel semantics

For each `(r, q)`:
1. `sum[p] = 0` for all `p`
2. `sum[p] += A[r][q][s] * C4[s][p]` for all `p, s` (a vector x matrix contraction)
3. `A[r][q][p] = sum[p]` (write back in place)

This is `A[r][q][:] = A[r][q][:] * C4`, a tensor-times-matrix multiply where the
`NP x NP` matrix `C4` is shared by every `(r, q)` slice.

## Transformation applied

1. **Removed the degenerate `loop_once` (`0 to 1`) wrapper** and moved the
   `{dmd.extract}` attribute onto the now-outermost `%r` loop. The wrapper was a
   single-iteration no-op, so this is semantics-preserving and keeps a single
   outermost `affine.for` carrying the attribute.

2. **Tiled the accumulation nest over both the contraction dimension `s` and the
   output dimension `p` with tile size 32** (`pp`/`ss` step-32 loops over 32-wide
   inner loops using `%pp + %p`, `%ss + %s` index expressions). The zero-init and
   write-back loops are kept as separate full-width sweeps so the accumulation
   into `sum[p]` across all `s`-tiles remains correct (sum is initialized once,
   then accumulated over every `ss` tile, then written back once).

The split-and-tile keeps the per-tile init/accumulate/finalize ordering identical
to the original: every `sum[p]` is zeroed before any product is added, and every
`A[r][q][p]` is written only after all `s` have been accumulated. No loads, stores,
math, or domain sizes changed.

## Why it cuts data movement

The hot data in the contraction is the `C4` matrix and the per-`(r,q)` `sum`/`A`
row. In the original `(p, s)` ordering each `p` streams a full length-`NP` column
of `C4` and re-reads the whole `A` row; the working set per `(r,q)` is the entire
`NP x NP` `C4` (`O(NP^2)` doubles), so once `NP^2` exceeds cache, every cache line
of `C4` is evicted before its reuse and the kernel reloads `C4` from memory with
poor spatial locality (column-major stride through a row-major array).

After tiling, the inner working set is a single `32x32` `C4` block plus the 32-wide
`sum` tile and 32-wide `A` tile (`~32^2 + 64` doubles, ~8.7 KB) which fits in L1.
Inside a tile each loaded `C4` cache line is reused across all 32 `p` values, the
`A` element is reused across 32 `p`, and the `sum` element is reused across 32 `s`,
so reuse distances drop from `O(NP)`/`O(NP^2)` down to `O(32)`. This converts the
streaming, capacity-miss-dominated access pattern into a blocked one that captures
spatial and temporal reuse within cache.

## Predicted improvement

The cross-`(r,q)` reuse of `C4` is structural (the `(r,q)` loops must stay outside
because `sum` is per-`(r,q)`), so total `C4` references are unchanged; the win comes
from turning capacity misses into hits by shrinking the resident working set from
`O(NP^2)` to `O(32^2)` and from recovering spatial locality on each cache line.

Predicted data-movement reduction: **~2x to 3x** for `NP` large enough that
`NP^2` overflows the modeled cache (the larger `NP` relative to cache, the closer
to the upper end). For `NP` already cache-resident the factor approaches 1x.
