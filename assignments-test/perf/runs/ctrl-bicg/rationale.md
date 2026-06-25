# bicg optimization rationale

## Reference
```c
for(i) for(j){ s[j]+=r[i]*A[i*N+j];  qi+=A[i*N+j]*p[j]; } q[i]=qi;
```
Fused `q = A*p` (row dot-product) and `s = A^T*r` (column scatter). The loop order
(i outer, j inner) is already correct for streaming `A` and `s` contiguously, but
each inner iteration does a read-modify-write of `s[j]` and a load of `p[j]` for
only ONE row of work — `s[]` and `p[]` traffic is the bottleneck, and the loop-
carried RMW on `s[j]` serialized per single FMA limits ILP.

## Transformation
**Register blocking of the outer i-loop by 4 rows** (with a scalar remainder loop
for `N % 4`). For each column `j` the inner loop now:
- loads `p[j]` once and reuses it across 4 row accumulators `q0..q3`;
- loads `s[j]` once, adds the combined `r0*a0+r1*a1+r2*a2+r3*a3` contribution, and
  stores it back once.

This is correctness-preserving (only reassociation of the `s[j]` sum; verified
all-close to machine epsilon for N up to 513 incl. non-multiples of 4).

## Why faster
- `s[]` read-modify-write traffic and `p[]` load traffic are cut ~4x (amortized
  over 4 rows), turning the memory-bound inner loop more compute-bound.
- Four independent `q` accumulators and four independent A streams expose ILP and
  let the FMA units / SIMD lanes stay busy instead of stalling on the single
  serialized `s[j]` dependency.
- `restrict` lets the compiler keep `s[j]` in a register across the 4 updates and
  freely vectorize. A streams are contiguous and prefetcher-friendly.

## Predicted speedup
~1.8–3x vs ref on a modern AVX2/AVX-512 core (the kernel is dominated by the
s[]/p[] memory traffic that blocking removes; the exact factor depends on whether
the 4 active A rows + s + p fit in L1/L2 at the tested N).
