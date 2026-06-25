# SYRK Optimization Rationale

## Kernel
`C[i][j] = beta*C[i][j] + alpha * sum_k A[i][k]*A[j][k]`, with the symmetric
(triangular) range `0 <= j <= i < n` and accumulation length `k = 0..m`.
Original loop order: `i -> { j (beta scale) ; k -> j (accumulate) }`.

## Transformation applied
**Loop interchange** of the inner accumulation nest: change `i -> k -> j` to
`i -> j -> k` (move `j` *outside* `k`). The triangular `beta`-scaling sweep over
`j` is left untouched and still runs before the accumulation. The redundant
`affine.for %loop_once = 0 to 1` wrapper (executed exactly once) was removed and
`{dmd.extract}` was placed on the now-single-outermost `%i` loop, as required.

No tiling with `min(kk+32, m)` bounds was used: the extractor rejects `min()`
upper bounds, and `m` is a symbolic/dynamic parameter so a 2-level k-tile carries
no modeled locality benefit and cannot guarantee `32 | m`. The interchange is the
clean, machine-scorable win.

## Why it cuts data movement (reuse / locality)
In the original `i -> k -> j` order, for every value of `k` the inner loop sweeps
the entire triangular `C[i][0..i]` row, so each `C[i][j]` is re-loaded and
re-stored once per `k`. Its reuse interval spans a full j-sweep, which is too long
to stay cache-resident; this puts the `C` accumulation traffic into the
leading-order `m*n^2` DMD mass.

After interchanging so `j` is outside `k`, each scalar `C[i][j]` is loaded once,
accumulated over the **entire** `k = 0..m` sweep while resident in a
register / L1 line, and stored once. Its reuse distance collapses to ~1. The
remaining leading-order traffic is dominated by streaming the `A[i][k]` and
`A[j][k]` reads, which is the irreducible part of the computation.

The read/write *set* is unchanged (total accesses identical:
`2*m*n^2 + n^2 + 2*m*n + n`); only the order changes, so semantics are preserved.

## Analyzer verification (mcp__dmd__analyze_mlir, block_size = 64)
- Extraction: **PASS** (no error; round-trip validated).
- Total access counts: **identical** to original (semantics preserved).
- Leading-order `m*n^2` DMD coefficient:
  - Original: constant `~2.722`.
  - Optimized: `~1.10` at m=64, `~1.13` at m=1024 (grows slowly, like `sqrt(m)`).
- Ratio optimized/original on the dominant term: **~0.40–0.43x** for the standard
  SYRK regime (m comparable to or smaller than n); approaches parity only in the
  extreme `m >> n`.

## Predicted improvement factor
**~2.4x reduction** in leading-order data movement (ratio ~0.41) in the practical
`m <= n` regime, driven entirely by capturing `C[i][j]`'s k-accumulation reuse.
