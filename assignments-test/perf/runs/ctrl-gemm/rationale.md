# Optimization rationale — `gemm` (C = 0.9*C + 1.1*A*B)

## Reference problems
The reference does:
1. A separate full pass over C to scale by 0.9 (extra N^2 memory traffic).
2. An i-j-k triple loop where the inner k loop walks `B[k*N+j]` with **stride N**
   (column access). This is cache-hostile, cannot vectorize over k for B, and
   reloads/recomputes `1.1*A` every iteration.

## Transformations applied
- **Loop interchange to i-k-j.** The innermost `j` loop is now unit-stride over
  both `B[k,*]` and `C[i,*]`, so clang vectorizes it with FMAs (AVX/AVX-512 under
  `-march=native`). This is the dominant win — B is read sequentially instead of
  by column.
- **Fused beta scaling.** `C *= 0.9` is folded into the first k-block update
  (`first_k`) instead of a separate N^2 pass, eliminating one full read+write of C.
- **Pre-scale A by 1.1 (packed).** `alpha*A` is packed once per block into a small
  contiguous buffer, turning the inner loop into a pure FMA and giving A contiguous,
  cache-friendly access during packing.
- **Register blocking MR=4 rows.** Each `B[k,j]` value is reused across 4 C rows,
  raising arithmetic intensity (4 FMAs per B load) and amortizing C load/store cost.
- **Cache blocking (KC=256 k, NC=512 j).** Keeps the active B panel and C row-block
  resident in L1/L2 so they aren't streamed from memory repeatedly.
- **Remainder loops** for ragged i/k/j blocks → correct for any N (768–2048),
  including non-multiples of tile sizes (verified at N=769,1023,1025,2047, etc.).

## Why faster
Converts a strided, non-vectorizable inner loop into a streaming vectorized FMA
loop, removes one full C pass, and reuses B/A heavily in registers and cache.

## Predicted speedup
Roughly **8–20x** over the naive reference at N in [768,2048]: the strided->unit-stride
interchange plus vectorization alone typically yields ~6–10x, register blocking and
the fused scaling/cache blocking add the rest. Verified numerically all-close.
