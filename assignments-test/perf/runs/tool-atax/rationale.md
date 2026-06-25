# atax optimization rationale

## What the reference does
```
pass1: for i: tmp[i] = sum_j A[i*N+j]*x[j]   // streams all of A
pass2: for i,j: y[j] += A[i*N+j]*tmp[i]       // streams all of A AGAIN
```
A is N*N doubles. For N in [2048,8192] that is 32MB..512MB, far larger than
any cache, so each of the two passes is a full cold DRAM stream of A. A is the
entire memory cost; x, y, tmp (N doubles each) are negligible.

## Transformation: loop fusion (read A once)
I fused the two nests so each row `A[i,:]` is brought from DRAM exactly once.
Per row i:
1. full reduction `t = dot(A[i,:], x)`, store `tmp[i] = t`;
2. immediately reuse the **same, now L1/L2-resident** row for `y[j] += A[i,j]*t`.

`tmp[i]` is the complete dot product over j, computed before the y-update, so
the math is identical to the reference (no reassociation beyond the 4-way
reduction tree in the dot product, which stays within rtol 1e-6). `y[]` is
accumulated across all i exactly as before.

Secondary: 4-wide scalar accumulators in the reduction to expose ILP/vectorization,
`restrict` pointers, row base pointer hoisting, scalar `t` (no reload of tmp[i]),
and scalar remainder loops for arbitrary N (no tile-size assumptions).

I considered tiling the inner j loop, but the two inner loops over a single row
are sequential and one row (16-64KB) already stays resident in L1/L2 between
them, so an extra j-tile loop adds index overhead without reducing traffic; I
left it out.

## Why faster
A's DRAM traffic is halved (read once, not twice) — and since A dominates total
bytes, end-to-end runtime is roughly halved. The analyzer modeled the second
sweep's reuse distance as ~N^2, so fusion removes a full order: data-movement
metric drops from ~N^3 (reference) to ~N^2.25 (fused), ~39x lower at N=8192 in
the model (DRAM-bandwidth bound, so real speedup tracks the ~2x byte reduction).

## Predicted speedup
~1.8-2.0x (memory-bandwidth bound; A read once instead of twice).

## Analyzer
Yes. The dmd affine-locality analyzer compared reference vs fused vs j-tiled
DSL variants: it confirmed the reference's ~N^3 A-reuse term, that fusion
removes it (~N^2.25), and that explicit j-tiling only changes the exponent's
constant within model noise — guiding me to fuse and skip the extra tiling.
