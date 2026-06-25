---
name: author-affine-dsl
description: Write or fix an AutoLALA DSL kernel from a description or from a fragment that fails to parse/validate. Use when the user wants to express a loop nest in the DSL by hand, asks for the DSL grammar/syntax, or has a DSL snippet that the analyzer rejects and needs corrected.
---

# Author an AutoLALA DSL Kernel

Help the user express a loop nest in the DSL `dmd-core` consumes, then validate it.
Read [the glossary](../../../docs/affine-analysis-glossary.md) §4 for the full
grammar and §6 for the limits.

## Tools

- `mcp__dmd__validate_dsl` — `{ dsl }` → `{ valid, params, arrays, parallel }` on
  success, or `{ valid:false, error }`. Always validate before declaring done.
- `mcp__dmd__analyze_dsl` — once valid, analyze it (or hand to `analyze-affine`).

## Grammar in one screen

```
params M, N, K;              // declare every symbolic parameter, comma-separated
array A[M, K];               // rank = number of extents; extents are expressions
array B[K, N];
array C[M, N];

for i in 0 .. M {            // half-open [lower, upper); bounds are affine exprs
  for j in 0 .. N {
    for k in 0 .. K step 1 { // optional `step <integer literal>`
      read  A[i, k];         // load
      read  B[k, j];
      update C[i, j];        // read-modify-write (use for accumulation)
    }
  }
}
```

- **Statements:** `read X[..];` (load), `write X[..];` (store),
  `update X[..];` (load+store of the same cell — use for reductions/`+=`).
- **Guards:** `if i < j { … } else { … }` — conditions joined with `&&`
  (`<`, `<=`, `==`, `>=`, `>`). Each side is an affine expression.
- **Parallel:** `parallel(8) for p in 0 .. M { … }` — **at most one** parallel
  loop in the whole program.
- **Expressions:** `+`, `-`, `*`, `/` (floor division), unary `-`, parentheses,
  integer literals, and names of declared params / enclosing loop variables.

## Rules to keep it valid

1. **Declare before use.** Every name in a bound, extent, index, or guard must be a
   declared `param` or an enclosing loop variable.
2. **Affine only.** Indices/bounds must be affine: `c0*var + c1*var + … + const`.
   No variable×variable products, no `var / var`, no `mod`, no `ceildiv`.
3. **Rank matches extents.** `array A[M, N];` ⇒ index it as `A[e1, e2]`.
4. **`step` takes an integer literal**, not an expression.
5. **One parallel loop**, and it cannot be nested inside another `parallel`.
6. **Accumulation ⇒ `update`.** `C[i,j] += A[i,k]*B[k,j]` is `update C[i, j];`,
   not separate `read`+`write` (and never just `write`, which would miss the read
   reuse).

## Procedure

1. Map the math to loops: one `for` per index, ranges from the iteration space.
2. Declare `params` and `array`s (rank = dimensionality, extents = sizes).
3. Emit accesses: inputs `read`, outputs `write`, accumulators `update`. Add `if`
   guards for triangular/banded iteration.
4. `validate_dsl`. If `valid:false`, map the `error` to a rule above and fix:
   - "unknown identifier" → declare it or fix a typo (rule 1).
   - "non-affine" / parse error near `*` or `/` → rule 2.
   - rank mismatch → rule 3.
5. Once valid, analyze (hand to `analyze-affine`) or return the kernel.

## Worked example — triangular solve body

> "For an N×N lower-triangular matrix, read L[i,j] and x[j] and update x[i], for
> j < i."

```
params N;
array L[N, N];
array x[N];
for i in 0 .. N {
  for j in 0 .. N {
    if j < i {
      read   L[i, j];
      read   x[j];
      update x[i];
    }
  }
}
```

`validate_dsl` → expect `valid:true` with `params:["N"]`, two arrays (ranks 2 and
1). Then analyze.
