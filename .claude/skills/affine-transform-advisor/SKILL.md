---
name: affine-transform-advisor
description: Propose loop transformations (tiling, interchange, fusion, parallelization) for an affine kernel and re-evaluate them with the analyzer to recommend the best one. Use when the user asks how to improve locality / reduce data movement / speed up a loop nest, or wants tiled/reordered variants compared. Advisor-only — it suggests and ranks, it does not edit the user's source.
---

# Affine Transformation Advisor

Given a kernel and its analysis, hypothesize transformations, synthesize each as a
DSL variant, and **measure** them with the analyzer before recommending one. This
skill is advisor-only: it proposes and ranks variants; it does not modify the
user's code unless they explicitly ask.

Read [the glossary](../../../docs/affine-analysis-glossary.md) (§3 interpretation,
§4 DSL grammar) first.

## Tools

- `mcp__dmd__compare_variants` — `{ variants: [{label, dsl}], block_size?, num_sets?, max_operations? }`
  → a side-by-side table of `dmd_formula`, `total/warm/compulsory_accesses` per
  variant. The primary tool of this skill.
- `mcp__dmd__analyze_dsl` — deep-dive a single variant's RD/RI distributions.
- `mcp__dmd__validate_dsl` — validate each synthesized variant before comparing.

## Procedure

1. **Diagnose first.** Run `analyze-affine` (or `analyze_dsl`) on the baseline.
   Identify the dominant `rd_distribution` region — the loop/array whose reuse
   distance scales worst. That points at the transformation:

   | Symptom in the report | Hypothesis |
   | --- | --- |
   | A reused array's RD scales with an **outer** loop's trip count | **Tile** that outer loop — RD drops to the tile size. |
   | Reuse is carried by the wrong innermost loop (e.g. `B[k,j]` re-streamed per `i`) | **Interchange** to put the reuse-carrying loop innermost. |
   | A `write` array is immediately `read` by a following nest | **Fuse** the two nests to keep the value warm. |
   | Independent outer iterations, large kernel | **Parallelize** the outer loop (`parallel(T) for`) and read the CRI model. |

2. **Synthesize variants as DSL.** Write the baseline plus one DSL per hypothesis.
   Tiling example (split `for i in 0..M` by tile `T`):

   ```
   for ii in 0 .. M step T {        // step is an integer literal
     for i in ii .. (ii + T) {      // bound expressions are affine
       ...
     }
   }
   ```

   Interchange = reorder the `for` headers. Keep array decls/params identical
   across variants so the comparison is apples-to-apples. `validate_dsl` each one.

3. **Compare.** Call `compare_variants` with all variants and the same options.
   Use a fixed, descriptive `label` per variant (`ijk-baseline`, `ikj-interchange`,
   `i-tiled-32`).

4. **Rank and recommend.** Metrics are symbolic — judge **directionally**:
   - Lower data movement (`dmd_formula`) in the dominant term wins.
   - More `warm` (and proportionally less `compulsory`) traffic ⇒ better reuse.
   - A variant whose DMD drops a named dim (e.g. removes an `N` factor) is a real
     scaling win, not just a constant-factor one.
   Recommend the winner and **quote the metric delta** vs baseline. If two are
   close, deep-dive both with `analyze_dsl` and compare RD distributions.

5. **Caveat.** State that tile sizes here are modeled symbolically; the *direction*
   (tiling helps) is trustworthy, the *exact* best tile size is not — that needs
   measurement on hardware.

## Worked example — matmul interchange

Compare `ijk` vs `ikj`:

- `ijk`: innermost `k` reuses nothing across iterations of `j`; `B[k,j]` is
  re-streamed → RD scales with `N`.
- `ikj`: innermost `j` streams `B[k,j]` and `C[i,j]` with unit stride; `A[i,k]` is
  loop-invariant in `j` → smaller RD.

`compare_variants([{label:"ijk",dsl:…},{label:"ikj",dsl:…}])` should show `ikj`
with lower data movement / more warm traffic. Recommend `ikj`, quoting both
`dmd_formula`s.

## Do not

- Do not edit the user's MLIR/source unless they ask — synthesize variants as
  standalone DSL and compare.
- Do not assert a winner without `compare_variants` backing it.
- Do not fan out analyses concurrently (Barvinok is serialized; `compare_variants`
  already sequences them).
