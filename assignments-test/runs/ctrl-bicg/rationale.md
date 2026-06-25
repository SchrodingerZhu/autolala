# BICG locality optimization

## Kernel
```
for i in 0..N:
    q[i] = 0
    for j in 0..M:
        s[j] += r[i] * A[i][j]      // s, p indexed by j
        q[i] += A[i][j] * p[j]      // q, r indexed by i
```

## Access / reuse analysis of the original (i outer, j inner)
- `A[i][j]`: each element touched exactly once → N*M traffic, already optimal, row-major sequential.
- `s[j]`, `p[j]`: the *entire* M-length vector is swept for every value of `i`. The
  reuse of a given `s[j]`/`p[j]` across successive `i` has reuse distance ~M. Once
  the M-vector exceeds cache capacity, every `i` reloads them: **N*M traffic each**.
- `q[i]`, `r[i]`: loop-invariant inside the `j` loop (scalar reuse) → ~N traffic each.

The dominant, capacity-limited cost is the repeated streaming of `s` and `p`.

## Transformation applied
1. **Index-set splitting / loop distribution of the initializer**: `q[i] = 0` is
   pulled into its own `i` loop so it still executes exactly once per `i` after the
   loop order changes (semantics preserved).
2. **Strip-mine the `j` loop by 32** and **interchange the `j`-tile loop (`jj`)
   outside the `i` loop**, giving order `jj` (step 32) → `i` → `j` (0..32).

For each 32-wide block of `j`, the whole `i` dimension is swept while that block of
`s[jj..jj+31]` and `p[jj..jj+31]` (32 doubles = 256 B each, fits in L1) stays
resident. Each `s`/`p` element is therefore loaded once → **M traffic each instead
of N*M**.

## Cost traded
`q[i]` and `r[i]` are now touched once per `i` per `j`-tile → N*(M/32) instead of N.
This adds traffic on the two *N*-length vectors, but they are reused with a short
distance within each tile and N fits in cache far more easily than the M-streamed
`s`/`p` did. `A` traffic is unchanged (N*M, touched once either way).

## Semantics
- `s[j]` accumulation over `i` is still in ascending `i` order (fixed j) → identical FP order.
- `q[i]` accumulation over `j` is still in ascending `j` order (fixed i), merely split
  across the outer `jj` tiles → identical FP order.
- Same memrefs, same set of loads/stores, same domains. Only iteration order changed.

## Predicted improvement
The capacity miss traffic of `s` and `p` drops from ~2*N*M to ~2*M. For large M
(M-vector spilling out of cache) the dominant data-movement term falls from
O(N*M) (A) + O(2*N*M) (s,p) to O(N*M) (A) + O(2*M) (s,p). Predicted data-movement
reduction factor: roughly **2x–3x** overall (eliminating two of the three N*M
streaming terms), with the benefit growing as M outgrows the cache.
