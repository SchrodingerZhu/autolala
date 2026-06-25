# SYRK optimization rationale

## Kernel
`C = 0.9*C + 1.1*A*A^T`, lower triangle only (`j<=i`). Equivalent to
`C[i,j] = 0.9*C[i,j] + 1.1 * sum_k A[i,k]*A[j,k]` for `j<=i`.

## Transformations
1. **Fuse / hoist constants.** The `0.9` scaling stays a cheap O(N^2/2) prologue.
   The factor `1.1` is hoisted out of the k-reduction: we accumulate the raw dot
   product `sum_k A[i,k]*A[j,k]` and multiply by `1.1` once at write-back. This is
   only FP reassociation (within rtol 1e-6) and removes a multiply from the hot loop.
2. **GEMM-style 3-level cache blocking** over j (NC=256), i (MC=192), k (KC=256),
   so the A row-panel (`A[i,*]`) and A^T column-panel (`A[j,*]`) stay L2-resident
   and the strided `A[j,k]` slab stays hot.
3. **4x4 register micro-kernel** streaming k: 16 scalar accumulators held in
   registers across the k-reduction; clang auto-vectorizes to 16 NEON `fmla v.2d`
   (2 doubles/lane) — confirmed in the emitted assembly.
4. **Triangle handling.** Off-diagonal register tiles (fully below diagonal) use the
   fast full micro-kernel; only the O(N) diagonal-straddling and remainder tiles take
   a generic masked path enforcing `j<=i`. Strictly-upper tiles are skipped entirely.
   Non-multiple-of-tile N handled by remainder bounds — verified for N=7..2048.

## Why faster
The reference uses i-k-j with `j` innermost: correct order but no cache blocking, so
A rows are re-streamed from memory and the working set blows past L1/L2 as N grows.
Blocking + a register-resident C tile turns the N^3 traffic into cache hits and feeds
the FMA units at full SIMD width. Roughly half the flops are skipped (lower triangle).

## Analyzer
Yes. The dmd affine analyst confirmed i-k-j beats i-j-k by 1.6x-2.1x in modeled data
movement (the i-j-k reuse distance scales with N), and recommended the GEMM tiling
with mr=nr register block and Mc≈192 / Nc≈256 / Kc≈256 — which I adopted directly.

## Predicted speedup
~4-8x at N in [768,2048] (cache-blocking + SIMD register kernel vs untiled reference).
```
```
