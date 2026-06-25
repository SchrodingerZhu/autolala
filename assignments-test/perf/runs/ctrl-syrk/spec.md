# Kernel `syrk` — optimize for single-core native performance

## What it computes
syrk: C = 0.9*C + 1.1*A*A^T, lower triangle only (A, C are N x N).

## Exact function signature (your opt.c MUST match this verbatim)
```c
void kernel(int N, double* A, double* C);
```
Arrays are flat row-major `double*`. `N` is the problem size, passed at runtime.

## Test conditions
- Compiled with: `clang -O3 -march=native -funroll-loops` (single translation unit, no LTO).
- Run pinned to ONE core (`taskset -c 0`). Optimize for single-core performance only.
- Evaluated at a problem size somewhere in the range **N in [768, 2048]** — the exact
  size is NOT disclosed, so DO NOT hardcode or special-case a specific N. Your code must
  be correct and fast for any N in that range, including N that are NOT multiples of any
  tile size you pick (handle remainder/boundary iterations).

## Correctness bar
Your kernel's full output array(s) must match the reference within
`numpy.allclose(rtol=1e-6, atol=1e-9)`. You may reorder floating-point operations
(tiling, blocking, interchange) — small reassociation is fine — but the result must
stay all-close. Do NOT change what is computed.

## Deliverables (write into THIS directory)
- `opt.c` — contains ONLY the optimized `kernel(...)` (same signature). No `main`.
- `rationale.md` — the transformation(s) you applied and why they speed it up.
