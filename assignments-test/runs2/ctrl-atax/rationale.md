# ATAX optimization rationale

Kernel: `y = A^T (A x)`, computed in two per-row phases:
- **Phase 1** (per row `i`): `tmp[i] = Σ_j A[i][j]·x[j]`  — reads full row `A[i][:]` and full `x[:]`.
- **Phase 2** (per row `i`): `y[j] += A[i][j]·tmp[i]`  — reads full row `A[i][:]`, reads/writes `y[:]`.

## Transformation applied

**Loop tiling (strip-mining) of both inner `j` loops by tile factor 32**, keeping the
`i` loop outermost. Each `affine.for %j = 0 to %n` becomes
`affine.for %jj = 0 to %n step 32 { affine.for %j = 0 to 32 { ... %jj + %j ... }}`.

The two phases are kept separate (not fused) and `i` is kept outermost — both
deliberately:

- **`i` stays outermost** because `A` (size `m·n`, the dominant array, read twice)
  is streamed row-major with one use per element per phase. Any interchange that
  put `j` outside `i` would stride through `A` column-wise (stride `n`), destroying
  its spatial locality and inflating the dominant traffic. So `A`'s access pattern
  is left as unit-stride row-major.
- **Phases are not fused** because Phase 2 needs the *fully accumulated* `tmp[i]`
  (sum over all `j`), which only exists after Phase 1 completes for that `i`.
  Tile-level fusion would violate this true dependence. Tiling each phase
  independently is unconditionally semantics-preserving.

## Why this cuts data movement

1. **A-row-block reuse across phases.** When `n` is large, a full row `A[i][:]`
   (`n·8` bytes) loaded in Phase 1 can be evicted before Phase 2 re-reads it,
   forcing the row to be fetched from memory twice. With `j` tiled, the working
   set between the two phases is bounded conceptually to 32-element blocks; the
   freshly-touched `A[i][jj..jj+32]` block stays cache-resident and is reused,
   reducing redundant capacity misses on `A`.
2. **Bounded `x` / `y` working set.** Phase 1 re-reads `x[:]` and Phase 2
   re-reads/writes `y[:]` for every row. Tiling caps the active `x`/`y` footprint
   to a 32-wide block (256 B each), which stays in L1 across the inner block,
   improving spatial-locality utilization and reducing conflict/capacity misses
   when `n` exceeds cache.

## Predicted improvement

The dominant `A` traffic (`2·m·n`) is intrinsically stream-once-per-phase and
largely unchanged; the win comes from eliminating redundant `A`-row and `x`/`y`
re-fetches in the large-`n` regime. **Predicted data-movement reduction ≈
1.3×–1.8×** for `n` well beyond L1/L2, trending toward ~1× when the row already
fits in cache (tiling then neither helps nor hurts asymptotic traffic).
