---
name: analyze-affine
description: Analyze the locality / data-movement behavior of an affine loop kernel given as MLIR or AutoLALA DSL. Use when the user supplies a loop nest (matmul, stencil, gemm, conv, etc.) and asks about cache behavior, reuse, data movement, memory traffic, how it scales, or where the locality bottleneck is. Entry point for affine-program analysis.
---

# Analyze an Affine Kernel

Turn an affine loop nest into a locality verdict: the DMD (data-movement)
formula, the dominant reuse-distance regions, warm-vs-compulsory traffic, whether
it scales, and the likely bottleneck.

**Read [the glossary](../../../docs/affine-analysis-glossary.md) first** — it
defines every term and JSON field referenced below. The analysis is *symbolic*:
results are expressions in the kernel's named parameters (`M`, `N`, `K`).

## Tools

This skill drives the `dmd` MCP server (registered in `.mcp.json`):

- `mcp__dmd__analyze_mlir` — extract a tagged MLIR loop **and** analyze it in one call.
- `mcp__dmd__analyze_dsl` — analyze AutoLALA DSL source directly.
- `mcp__dmd__validate_dsl` — parse/validate DSL without analysis (cheap pre-check).
- `mcp__dmd__extract_from_mlir` — MLIR → DSL only (use the `mlir-to-dsl` skill for fixes).

If the MCP server is unavailable, fall back to the CLI: `cargo run -p dmd-cli -- --json [--block-size N] [--num-sets N] < kernel.dsl` (and `cargo run -p mlir-extract -- input.mlir` for extraction).

## Procedure

1. **Detect the input form.**
   - MLIR (`module`, `func.func`, `affine.for`, `memref<…>`) → ensure the target
     loop nest carries the `dmd.extract` attribute, then call `analyze_mlir`. If
     it is not tagged, ask the user which loop, or apply the `mlir-to-dsl` skill
     to tag + extract first.
   - DSL (`params`, `array`, `for … in … { read/write/update }`) → `analyze_dsl`.
   - Unsure → run `validate_dsl`; a clean parse confirms DSL.

2. **Choose options.** Start with defaults (`block_size=1`, `num_sets=1`). Set
   `block_size` > 1 to model spatial locality (cache lines); widen `num_sets` to
   study conflicts. Raise `max_operations` only if analysis aborts on a big nest.

3. **Run the analysis.** Prefer `analyze_mlir` / `analyze_dsl` so you get the full
   report **and** the digest in `summary`.

4. **Interpret** (full checklist in the glossary §3):
   - Quote `dmd_formula_plain` — the headline. Note which named dims survive.
   - **Scales?** Named dims in the formula ⇒ locality depends on problem size.
   - **Traffic split:** `warm` vs `compulsory`. Mostly compulsory ⇒ little reuse;
     mostly warm ⇒ exploitable reuse.
   - **Bottleneck:** in `rd_distribution`, the entry whose `value_plain` (RD)
     grows fastest with a large `count_plain` is where a real cache loses reuse.
   - If `parallel` is present, read the CRI model (`thread_count`, `schedule`,
     `model_kind`) instead of plain RD/DMD.
   - Always surface `notes`.

5. **Report.** Lead with a one-sentence verdict, then: the DMD formula, the
   bottleneck loop/array, the warm/compulsory split, and (if asked to improve it)
   hand off to the `affine-transform-advisor` skill.

## Worked example

DSL `ijk` matmul:

```
params M, N, K;
array A[M, K]; array B[K, N]; array C[M, N];
for i in 0 .. M { for j in 0 .. N { for k in 0 .. K {
  read A[i, k]; read B[k, j]; update C[i, j];
} } }
```

`analyze_dsl` → read `dmd_formula_plain`; expect `B`'s reuse distance to scale
with `N` (B is re-streamed per `i`), flagging the bottleneck that motivates
interchange/tiling. Confirm any fix with `compare_variants` (see transform-advisor).

## Failure handling

- **Extraction rejected** (span + `help`): an unsupported op (`memref.load`,
  `mod`, `scf.for`, …). Switch to the `mlir-to-dsl` skill and apply the suggested
  fix; quote the offending line/column to the user.
- **`validate_dsl` reports `valid:false`:** fix the DSL per the error (use the
  `author-affine-dsl` skill) before analyzing.
- **`analysis failed` / timeout:** the nest exceeded `max_operations`. Raise it,
  or reduce the kernel; never fan out analyses in parallel (Barvinok is serial).

## Limits to state up front

One parallel loop only; `mod`/`ceildiv` and non-affine `*`/`/` unsupported;
dynamic memref extents become independent params; `affine.store` ⇒ `write` (not
`update`). Metrics are symbolic and approximate — compare directionally.
