# Optimization rationale: PolyBench `covariance`

## Baseline access behavior

The kernel has three phases over `data` (n rows x m cols), `mean` (m), `cov` (m x m).

1. **Mean** — `for j { mean[j]=0; for i { mean[j]+=data[i][j] }; mean[j]/=n }`.
   The inner loop walks a *column* of `data`: consecutive `i` are `m` elements
   apart in row-major memory, so every access is a stride-`m` miss. Effectively
   no spatial reuse; ~`n*m` cold-ish loads of `data`.
2. **Center** — `for i { for j { data[i][j]-=mean[j] } }`. Row-major, already good.
3. **Covariance** — `for i { for j>=i { cov[i][j]=0; for k { cov[i][j]+=data[k][i]*data[k][j] }; ... } }`.
   The dominant term, O(m^2 * n). The k-loop streams two *columns* of `data`
   (stride-`m`), and the `cov[i][j]` accumulator is loaded/stored every k
   iteration with a long surrounding stream.

## Transformations applied

1. **Loop distribution + interchange in Step 1.**
   Split the `mean[j]=0` initialization and the `mean[j]/=float_n` normalization
   into their own `j`-loops, then interchange the accumulation loop from
   `j`-outer/`i`-inner to `i`-outer/`j`-inner:
   `for i { for j { mean[j]+=data[i][j] } }`.
   Now `data[i][j]` is traversed with **unit stride** along `j`, converting the
   column-major (stride-`m`) stream into a fully spatial, cache-line-friendly
   row-major stream. Semantics preserved: the reduction into `mean[j]` is
   associative-order-identical (each `mean[j]` still sums the same `data[i][j]`
   over all `i`); init and normalize are pure per-`j` operations safely hoisted
   into separate full sweeps.

2. **Tiling the reduction (k) loop in Step 3** by a factor of 32:
   `for kk step 32 { for k 0..32 { ... data[kk+k][i] ... } }`.
   This blocks the innermost stream so a 32-element block of each loaded `data`
   column plus the live `cov[i][j]` partial sum stay resident together,
   shortening the reuse distance of the accumulator and giving the loaded
   column block temporal reuse within the tile. The triangular `j>=i` domain and
   the symmetric `cov[j][i]=cov[i][j]` write are left untouched, and i/j are not
   tiled (that would need `affine.if` clipping, which is forbidden), so
   correctness is trivially preserved.

## Why this cuts data movement

- Step 1 changes from stride-`m` column access to stride-1 row access: with a
  cache line holding L doubles, traffic drops by roughly a factor of L
  (~8 for 64B lines) for the `data` reads in the mean phase.
- Step 3 tiling reduces capacity misses on the `cov` accumulator and gives
  in-tile temporal reuse of the streamed `data` block, trimming redundant memory
  traffic in the O(m^2*n) hot loop.

## Predicted improvement

Step 1 spatial-locality fix alone is ~L-fold (≈8x) on that phase's `data`
traffic. Folded over the whole kernel (Step 3 dominates asymptotically but Step 1
is a real fraction of traffic, and the k-tiling further trims Step 3), the overall
predicted data-movement reduction is roughly **2x–3x**.
