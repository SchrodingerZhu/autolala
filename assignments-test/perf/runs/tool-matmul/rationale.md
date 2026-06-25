# Optimization rationale — matmul `C += A*B`

## Reference problem
The reference is the naive `ijk` triple loop. With `j` outside `k`, the access
`B[k*N+j]` is **column-strided**: each inner-`k` step jumps `N` doubles. B is
re-streamed for every `j`, so its reuse distance is ~N²/8 — no L1/L2 cache holds
an N² working set for N≥768, giving a cache miss on essentially every B load.

## Transformations applied
1. **Loop interchange to `ikj`.** Hoisting `k` above `j` makes all three accesses
   unit-stride row streams (`A[i,k]` scalar-reused across `j`, `B[k,*]` and
   `C[i,*]` streamed contiguously). Reuse distance of the hot stream drops from
   quadratic (~N²/8) to linear (~N/4).
2. **Cache (macro) blocking on (k, j, i)** with `KC=256`, `NC=1024`, `MC=256`.
   This keeps the active A tile / B panel / C tile resident so reuse is actually
   *captured* by the cache; the analyzer showed this makes data-movement per
   access flat in N instead of growing.
3. **Register micro-kernel (4×4).** The innermost block accumulates a 4×4 tile of
   C entirely in 16 registers across the `k` loop, with A and B values loaded into
   registers and reused MR/NR times each. This removes C load/store traffic from
   the inner loop and exposes 16 independent FMA chains for the scheduler /
   auto-vectorizer (no intrinsics; clang `-O3 -march=native` vectorizes the
   unrolled body). FP results stay all-close (only k-order is preserved per
   element; reassociation is benign and in practice bit-identical here).
4. **Full remainder handling.** Every tile loop computes a clamped extent
   (`mr,nr,kk,mc,nc,nc`), and a general edge micro-kernel handles partial tiles,
   so any N (including primes / non-multiples) is correct.

## How the analyzer informed the choice
Using the `dmd` affine analyzer on DSL variants:
- `ijk` DMD/access = 20.5 (N=768) → 45.2 (N=2048), dominated by a `(7/32)N³`
  region at RD ≈ N²/8 (the column-strided B read).
- `ikj` DMD/access = 12.3 → 31.6 (~1.4× better; RD of hot stream ≈ N/4).
- **Tiled `ikj`** DMD/access ≈ **2.1 and flat in N** (~10× lower), with the
  smaller tile ranking best (32<64<128<256 on DMD/access).

So the analyzer drove: pick `ikj`, then tile. It models footprint/data-movement
only, so the register-blocking (constant-factor C-traffic removal) was added on
top per its guidance that this is orthogonal but important on hardware. Macro
tile sizes were set to L2-friendly byte budgets; the small inner register tile
plays the role of the analyzer's preferred small L1 tile.

## Predicted speedup
Versus the naive column-strided reference, expect roughly **8–15×** at N in
[768,2048] on a typical machine: the interchange alone removes the dominant
cache-miss stream, blocking captures reuse (flat DMD), and register blocking +
vectorization add a large constant factor. Verified bit-exact against the
reference for N from 1 to 1025 including primes and tile non-multiples.
