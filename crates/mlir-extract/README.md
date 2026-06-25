# mlir-extract

Translate an attribute-tagged MLIR `affine.for` loop nest into the AutoLALA DSL
("the language") consumed by `dmd-core`.

This is the `melior`-based successor to the old `raffine` ingestion path: instead
of building a private loop tree through a C++ helper, it reads the affine
dialect directly through the MLIR C API and emits DSL source text, which the rest
of the workspace (`dmd-cli`, `dmd-playground`) already understands.

## Usage

```bash
# Print the DSL for the loop tagged `dmd.extract`:
cargo run -p mlir-extract -- kernel.mlir

# Use a different marker attribute, write to a file:
cargo run -p mlir-extract -- kernel.mlir --attr my.marker -o kernel.dsl
```

Tag the loop nest you want with a unit attribute:

```mlir
affine.for %i = 0 to %M {
  affine.for %j = 0 to %N {
    affine.for %k = 0 to %K {
      %a = affine.load %A[%i, %k] : memref<?x?xf32>
      %b = affine.load %B[%k, %j] : memref<?x?xf32>
      %c = affine.load %C[%i, %j] : memref<?x?xf32>
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %c, %p : f32
      affine.store %s, %C[%i, %j] : memref<?x?xf32>
    }
  }
} { dmd.extract }
```

produces

```text
params p0, p1, p2, A_d0, A_d1, B_d0, B_d1, C_d0, C_d1;
array A[A_d0, A_d1];
array B[B_d0, B_d1];
array C[C_d0, C_d1];

for i0 in 0 .. p0 {
    for i1 in 0 .. p1 {
        for i2 in 0 .. p2 {
            read A[i0, i2];
            read B[i2, i1];
            read C[i0, i1];
            write C[i0, i1];
        }
    }
}
```

See `examples/` for runnable inputs.

## Translation

| MLIR | DSL |
| --- | --- |
| `affine.for %i = lo to hi step s` | `for iN in lo .. hi step s { ... }` |
| `affine.if #set(...)`             | `if c0 && c1 { ... } else { ... }` |
| `affine.load A[...]`              | `read A[...];` |
| `affine.store v, A[...]`          | `write A[...];` |
| `arith.*`, `math.*`, `complex.*`  | ignored (compute only; locality cares about memory) |

Identifier conventions:

- **Induction variables** of the extracted loops become `i0, i1, ...`.
- **Symbols** — any operand that is not an enclosing induction variable, i.e.
  loop-invariant within the nest (function arguments, outer induction variables)
  — become symbolic parameters `p0, p1, ...`.
- **Arrays** are named `A, B, C, ...` in first-access order. Static memref dims
  become integer extents; **dynamic** dims (`?`) become fresh parameters
  `A_d0, A_d1, ...`, because a memref type does not record which SSA value
  supplies its size.

The emitted DSL is re-parsed and semantically validated against `dmd-core` as a
self-check before it is printed (disable with `--no-validate`).

## Rejected (unsupported) constructs

Anything the language cannot express is rejected with an `ariadne` diagnostic
that underlines the offending operation, using the operation's MLIR source
location:

- non-affine / non-memory operations inside the nest (`memref.load`, `scf.for`,
  `func.call`, `affine.apply`, ...);
- `min` / `max` loop bounds (multi-result bound maps);
- `mod` / `ceildiv` in any affine expression;
- non-constant multiplication or division;
- loop-carried values (`iter_args` / reductions);
- zero-rank memref accesses.

```text
Error: the `mod` operator cannot be expressed in the target language
   ╭─[ kernel.mlir:5:12 ]
 5 │       %v = affine.load %A[%i mod 4] : memref<10xf32>
   │            ─────┬─────
   │                 ╰─────── the `mod` operator cannot be expressed in the target language
   │
   │ Help: rewrite the access without modulo, or extract a tiled form that avoids it
───╯
```

## Build requirements

`melior` / `mlir-sys` link against LLVM/MLIR 22. The workspace
`.cargo/config.toml` points `mlir-sys` at the apt.llvm.org install by default:

```toml
[env]
MLIR_SYS_220_PREFIX = "/usr/lib/llvm-22"
TABLEGEN_220_PREFIX = "/usr/lib/llvm-22"
```

These are defaults; export the same variables to override them for a
non-standard install. Only `mlir-extract` reads them — the other workspace
crates ignore them.
