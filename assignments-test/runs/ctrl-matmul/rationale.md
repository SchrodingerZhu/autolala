# Optimization Rationale

## Kernel
Square-ish dense matmul `C[i,j] += A[i,k] * B[k,j]` as a naive `i,j,k` loop nest.

## Transformation applied
**3D loop tiling (blocking)** of all three loops `i`, `j`, `k` with a tile
size of 32, producing the loop order:

```
ii (step 32) -> jj (step 32) -> kk (step 32) -> i (0..32) -> j (0..32) -> k (0..32)
```

The intra-tile body uses `%ii + %i`, `%jj + %j`, `%kk + %k` index expressions
on the original memrefs. No loads/stores were added or removed and the
arithmetic is unchanged, so semantics are preserved.

## Why this cuts data movement
In the original `i,j,k` nest:
- `A[i,k]`: the innermost `k` streams a full row of A per `(i,j)`; that row is
  re-read for every `j` (N times) and never stays resident across `j` because
  between reuses we touch O(N*K) other data — reuse distance far exceeds cache.
- `B[k,j]`: the inner `k` walks a full column of B; the whole B matrix
  (K*N elements) is swept for every `i`, so B is reloaded M times from memory.
- `C[i,j]`: stays in a register across the `k` loop (good), but A and B do not.

After tiling, each innermost computation works on a 32x32 block of A, a 32x32
block of B, and a 32x32 block of C. With tile size T=32 and f64 elements, the
three blocks occupy `3 * 32 * 32 * 8 B ≈ 24 KB`, which fits comfortably in a
typical 32 KB L1 / 256 KB L2. Consequences:

- An A block is loaded once and reused across all 32 `j` iterations (T-fold
  reuse held in cache instead of refetched).
- A B block is loaded once and reused across all 32 `i` iterations.
- The reuse distance for every array element drops from O(matrix) to
  O(tile), so almost all reuse becomes cache hits.

Memory traffic for the classic untiled matmul is dominated by reloading B
(≈ M*N*K accesses, each a miss) — O(N^3) for square N. Blocking reduces the
slow-memory traffic to roughly `2*N^3/T + N^2`, i.e. the B (and A) reload
factor shrinks by ~T.

## Predicted improvement
Slow-memory data movement reduced by approximately a factor of **T = 32**
(order ~10-32x in practice once the fixed N^2 capacity term and finite cache
associativity are accounted for). Confident the kernel still extracts and
analyzes: it uses only affine.for / affine.load / affine.store / arith.* /
memref, constant tile bounds, `%base + %iv` index expressions, and a single
outermost `{dmd.extract}` loop.
