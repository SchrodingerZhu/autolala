# SYRK locality optimization rationale

## Kernel
PolyBench `syrk`: for each row `i`, scale the lower-triangle `C[i][0..i] *= beta`,
then accumulate `C[i][j] += alpha * A[i][k] * A[j][k]` for all `k in [0,m)` and
`j in [0,i]`. The original compute nest is ordered `i -> k -> j`.

## Transformation applied
**Loop interchange** of the two innermost loops of the compute nest:
`(i, k, j)` -> `(i, j, k)`. The triangular bound `j <= i` (expressed with
`affine_map<(d0)->(d0+1)>(%i)`) is preserved verbatim; the `loop_once = 0 to 1`
scaffold loop was removed and `{dmd.extract}` moved onto the now-outermost `%i`
loop. Loop-invariant `alpha`/`beta` constants were sunk inside `%i` (no
semantic effect). The set of memrefs, loads, and stores is unchanged.

## Why this cuts data movement
In the original `i -> k -> j` order, the accumulator `C[i][j]` is reloaded and
restored once **per k**: across the whole compute nest each `C[i][j]` cell is
touched `m` times with a reuse distance of `O(i)` (a full triangle row sweep
sits between consecutive accesses to the same cell). For realistic problem
sizes the C row does not stay in the smallest cache level, so those `~m` round
trips to `C[i][j]` repeatedly miss.

After interchange to `i -> j -> k`, for a fixed `(i, j)` the *innermost* loop is
`k`, and every iteration touches the **same** `C[i][j]` cell. Its reuse distance
collapses to 1, so the cell is read once, updated `m` times, and written back
while resident in a register / L1 line. The `m` redundant fetch/spill pairs of
`C` per `(i,j)` are eliminated.

`A[j][k]` is now swept contiguously along row `j` (unit stride in the inner
loop), giving good spatial locality on `A`. `A[i][k]` loses the "load-once,
reuse-across-j" pattern it had in the original, but `A[i][*]` is a single row
that is small and hot, and the streaming access keeps it cache-resident; its
cost is dominated by the `C` traffic saved.

## Predicted improvement
The `C` accumulation traffic is the dominant term. Modeling it as the largest
reused working set, the interchange removes ~`m` redundant C-cell transfers per
`(i,j)`, replacing `O(triangle * m)` C-line round trips with `O(triangle)`.
Predicted data-movement reduction: roughly **2x-4x** on the C accumulation
component for typical `m`, with the overall kernel improving in the **1.5x-3x**
range depending on cache sizes and the `A` access mix.

## Notes
Tiling the `k` dimension would add further A-reuse blocking, but with a symbolic
trip count `%m` (not a known multiple of 32) a constant-trip inner tile would
require an `affine.if`/min-bound guard to stay correct, which the scoring
constraints disallow. The clean interchange is therefore the strongest
fully-analyzable, semantics-preserving choice here.
