# 3mm optimization rationale

## What ref does
Three matmuls (E=A*B, F=C*D, G=E*F) with the naive `i-j-k` dot-product order.
The innermost `k` loop strides B by N (column walk) — every inner iteration
touches a new cache line of B, so B is re-streamed from memory N times per
output row and SIMD can't be used on the reduction without a horizontal sum.

## Transformations applied
- **Loop interchange to `i-k-j`**: the inner loop now sweeps `j` contiguously
  over a row of B and accumulates into a contiguous row of C. This is a pure
  AXPY, so clang auto-vectorizes it with FMAs (full AVX/AVX-512 width) and B/C
  accesses are unit-stride.
- **Cache blocking (IB=64, KB=256, JB=512)**: tiles the i/k/j space so the
  working set of B and C stays resident in L1/L2 and is reused across the block
  instead of being re-fetched from DRAM.
- **Register blocking ×4 on i**: four C rows share each loaded B[k][j] value,
  so each streamed B element drives 4 FMAs — raising arithmetic intensity and
  hiding load latency. Scalar remainder loop handles leftover rows.
- **`restrict` + one `memset`** zero of C lets the compiler keep accumulators
  in vector registers across the j tile.
- Remainder handling via `min` bounds keeps it correct for any N (non-multiples
  of tile sizes verified at N=577,1023,700).

## Why faster / predicted speedup
Vectorized unit-stride inner loop + 4-way register reuse + L1/L2-resident tiles
turn a memory-bound column-walk into a compute-bound FMA stream. Expected
~6-15x over ref for N in [512,1536] (larger N → bigger win as ref thrashes cache
hardest). Result is bit-identical to ref here (max relative error 0).
