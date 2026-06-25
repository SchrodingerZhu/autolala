# atax optimization rationale

`atax` computes `tmp = A*x` then `y = A^T*tmp`, with A an N×N matrix
(N ∈ [2048, 8192], i.e. 32 MB–512 MB of doubles). A is far larger than the
last-level cache, while x and y are tiny (N doubles each, ≤64 KB). This makes the
kernel strongly **memory-bandwidth bound on A**.

## Reference cost
The reference performs two independent sweeps over A:
1. `tmp[i] = Σ_j A[i][j]·x[j]` — one full streaming read of A.
2. `y[j] += A[i][j]·tmp[i]` — a *second* full streaming read of A.

So A is fetched from DRAM **twice**. Per-row, both sweeps touch exactly row `i`.

## Transformation: loop fusion over the row dimension
For each row i, compute `t = tmp[i] = A[i]·x` and then **immediately** scatter
`y[j] += A[i][j]·t` using the same row, which is still resident in cache. A is now
read from DRAM **once** instead of twice → roughly **half the DRAM traffic** on the
dominant array. The dependence (need the full dot product before the y-update) is
respected by doing the dot product first, then the update of the same row; a single
row is ≤64 KB and fits in L2, so the second use hits cache, not memory.

Supporting tweaks: `restrict` pointers (no aliasing → enables vectorization and
fused load/FMA), 4-wide unrolling with 4 independent scalar accumulators to break
the FMA reduction dependency chain and expose ILP, and scalar remainder loops so
any N (including non-multiples of 4) is handled correctly. x/y stay L1-resident and
are reused across all rows.

## Predicted speedup
Memory-bound and traffic on A halved → expect roughly **1.7–2.0×** vs the reference
on large N, where DRAM bandwidth dominates and the saved second pass over A is the
main win. Smaller within-range N (more cache-resident) trend toward the lower end.
```
