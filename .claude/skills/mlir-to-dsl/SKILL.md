---
name: mlir-to-dsl
description: Tag an MLIR affine loop nest and extract it to AutoLALA DSL, translating extraction rejections into concrete fixes. Use when the user has MLIR (affine.for / affine.load / memref) and wants the DSL form, or when extraction fails and they need to know how to make a loop analyzable (mod, ceildiv, memref.load, scf.for, min/max bounds).
---

# MLIR → AutoLALA DSL

Get a tagged MLIR affine loop into DSL the analyzer can consume, and turn any
rejection into an actionable fix. Read [the glossary](../../../docs/affine-analysis-glossary.md)
§5–6 for what is accepted and the known limits.

## Tools

- `mcp__dmd__extract_from_mlir` — `{ mlir, attr? }` → `{ dsl }`, or a diagnostic
  with `line`/`column` and a `help:` note.
- `mcp__dmd__validate_dsl` — sanity-check the extracted DSL.
- CLI fallback: `cargo run -p mlir-extract -- input.mlir --attr dmd.extract`
  (prints an ariadne diagnostic with the source span underlined).

## Procedure

1. **Tag the target loop.** The extractor pulls the single op carrying the marker
   attribute (default `dmd.extract`). Attach it to the **outermost** `affine.for`
   of the nest you want:

   ```mlir
   affine.for %i = 0 to %M {
     ...
   } { dmd.extract }
   ```

   If the user passes a custom marker, set `attr` accordingly.

2. **Extract.** Call `extract_from_mlir`. On success, `validate_dsl` the result,
   then hand to the `analyze-affine` skill.

3. **On rejection**, the diagnostic names the op and points at the span. Apply the
   matching fix and re-extract:

   | Rejection | Fix |
   | --- | --- |
   | `memref.load` / `memref.store` | Raise to `affine.load` / `affine.store` (use affine maps for the indices). |
   | `scf.for` / generic loop | Convert the loop to `affine.for` with affine bounds. |
   | `mod` in an index | Rewrite without modulo, or extract a **tiled** form that avoids it. |
   | `ceildiv` | Only floor division `/` is expressible — reformulate. |
   | `min`/`max` affine bound | Split the loop so each piece has a single affine bound. |
   | other op inside the nest | If it is pure compute (`arith.*`,`math.*`,`complex.*`) it is already ignored; otherwise lower/remove it before extraction. |

4. **Post-extraction correctness pass** (the extractor cannot infer these):
   - **`store` → `write`, never `update`.** If an `affine.store` writes back a cell
     it just loaded (a reduction/accumulation like `C[i,j] += …`), change the
     emitted `write A[…]` to `update A[…]` so the reuse of that cell is counted.
   - **Dynamic memref extents** become independent params (`A_d0`, `A_d1`). If two
     dynamic dims are actually equal, unify them by editing the `array` decl /
     params so the analysis sees the real shape.

5. **Report** the DSL plus any hand-edits you made and why.

## Worked example — store that is really an update

```mlir
func.func @mm(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %k = 0 to %K {
        %a = affine.load %A[%i, %k] : memref<?x?xf32>
        %b = affine.load %B[%k, %j] : memref<?x?xf32>
        %c = affine.load %C[%i, %j] : memref<?x?xf32>
        %p = arith.mulf %a, %b : f32          // ignored (pure compute)
        %s = arith.addf %c, %p : f32          // ignored
        affine.store %s, %C[%i, %j] : memref<?x?xf32>
      }
    }
  } { dmd.extract }
  return
}
```

Extraction yields `read C[i,j]; … write C[i,j];` for the accumulator. Since `C[i,j]`
is loaded and stored each `k`, collapse those to **`update C[i, j];`**. Then
`validate_dsl` and analyze.

## Limits

`mod`/`ceildiv`, non-affine `*`/`/`, multiple/nested `parallel`, and `min`/`max`
bounds are unsupported. Dynamic extents are independent params. Quote the
diagnostic's line/column to the user so the fix is unambiguous.
