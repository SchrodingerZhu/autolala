# syrk optimization rationale

**Problem:** `C[i][j] = 0.9*C[i][j] + 1.1*sum_k A[i][k]*A[j][k]` for `j<=i`.

**Transformations**
- **Reformulate as row·row dot products.** The reference's `(i,k,j)` order reads
  `A[j*N+k]` with stride-N (column) access — a cache miss per element. Computing
  `C[i][j]` as the dot product of rows `i` and `j` of A makes both operands
  contiguous (stride-1), so streaming reads hit L1/L2.
- **4×4 register blocking** on `(i,j)`: each k-step loads 4 `A[i]` and 4 `A[j]`
  values and forms 16 products, raising arithmetic intensity from ~2 flops/load
  to ~16, and keeping 16 accumulators resident in vector registers.
- **k-tiling (KB=256)** keeps the active 4-row strips of A within cache across the
  inner reduction; the lower-triangle structure is preserved by guarding edge
  blocks (`jj<=ii`) while the bulk (fully-below-diagonal) blocks run the fast path.
- `restrict` + separated β-scale pass let clang vectorize/FMA the kernel freely.

**Why faster:** eliminates stride-N column traffic, ~8× higher reuse per loaded
value, FMA-friendly 4×4 microkernel. **Predicted speedup ~4–8× vs ref** at
N in [768,2048] (ref is memory-bound on column access; opt is compute-bound).

Verified all-close (exact match) for N from 1 to 1023 incl. non-tile-multiples.
