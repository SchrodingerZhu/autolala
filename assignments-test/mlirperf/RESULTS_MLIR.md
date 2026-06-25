# DMD-Tool Effectiveness — MLIR-Affine / Locality-Isolated Refactor

**Question.** With the C SIMD/register confounds removed — agents write **MLIR affine**
loop nests, `llc -O3` does the low-level vectorization for everyone, so the only lever is
**loop structure / locality** (exactly what `dmd` models) — does giving an agent the dmd
analyzer help it optimize for real single-core runtime?

## Design (per the three constraints)

1. **dmd is a hint only.** Tool agents were told to treat dmd output purely as a
   directional hint about data-movement *trend*, not a runtime predictor.
2. **Agents write MLIR affine, not C.** I (coordinator) supply the C driver
   (builds memref descriptors, calls the MLIR C-interface) and the exact lowering
   command. Agents can only restructure affine loops — no C vectorization tricks.
3. **Three size regimes.** Each kernel is tested at small / medium / large (each a
   range, exact N hidden); agents may submit one kernel or three tuned versions
   (`opt_small/medium/large.mlir`). Score = **average speedup over the three regimes**.

- 6 kernels × {tool, ctrl} = **12 agents**, identical `claude` base; only the dmd
  permission differs. Neither group may benchmark/time — both optimize blind to
  wall-clock; `check.sh` lets them verify correctness only.
- **Ground truth = real runtime**: lower MLIR→native (llvm-22), link driver, verify
  full output `numpy.allclose`, time with `hyperfine` pinned to one core (machine is
  aarch64). Speedup = `t_ref / t_opt`.

## Results — average speedup vs the naive affine reference

| kernel | tool (s/m/l) | ctrl (s/m/l) | winner |
|--------|-------------:|-------------:|--------|
| matmul | **4.43** (3.74/4.30/5.25) | **6.12** (4.96/6.02/7.38) | ctrl |
| gemm | **5.86** (4.91/5.74/6.92) | 4.28 (3.73/4.31/4.80) | tool |
| 2mm | 4.26 (3.33/3.85/5.59) | 4.39 (3.41/3.98/5.77) | ctrl |
| mvt | **2.62** (2.18/2.67/3.00) | 1.67 (1.45/1.70/1.86) | tool |
| atax | 1.48 (1.34/1.40/1.69) | **1.64** (1.55/1.49/1.88) | ctrl |
| syrk | **4.29** (3.90/4.80/4.18) | 4.01 (3.23/4.41/4.37) | tool |

**All 12 submissions built and were numerically correct (12/12), at every regime,
including non-tile-multiple N.**

## Aggregate

| metric | TOOL | CTRL | T/C |
|--------|-----:|-----:|----:|
| geometric-mean avg-speedup | **3.50** | 3.29 | **1.06** |
| mean avg-speedup | 3.82 | 3.69 | 1.04 |
| head-to-head wins | 3 | 3 | (0 ties) |
| correct / 6 | 6 | 6 | — |
| output tokens (contestant) | 149,760 | 92,369 | **1.62×** |
| output tokens (incl. analyzer sub-agents) | 232,404 | 92,369 | **2.52×** |

## Verdict

**When the task is restricted to the locality dimension the tool actually models, dmd's
value becomes real but marginal — and still not obviously worth its cost.** The tool
group edges the control on geometric mean (+6%) but the head-to-head is a dead **3–3
tie**, and it spent **1.6–2.5× more tokens** to get there.

This is the informative contrast across the three experiments on these kernels:

| ground truth | kernel medium | who wins | why |
|---|---|---|---|
| symbolic DMD (the tool's own metric) | MLIR | **tool** (8.5 vs 6.8 geo) | tautological — scored on what the tool optimizes |
| **real runtime** | **C** | **ctrl** (5.9 vs 5.6 geo, 6–3 H2H) | C lets unaided agents win on SIMD/register/ILP the model can't see |
| **real runtime** | **MLIR affine** (this) | **~tie**, tool +6% | confounds removed → only locality remains, so the tool's signal finally shows |

Even here the tool doesn't dominate, for two visible reasons in the per-kernel data:
- **`matmul` (ctrl 6.12 vs tool 4.43):** the control agent added aggressive
  **unroll-and-jam ×4** on top of tiling; the tool agent leaned on the analyzer's
  data-movement framing and tiled without the same register-level ILP. Even in affine
  MLIR, an ILP lever the model doesn't quantify decided the kernel.
- **`gemm`/`mvt`/`syrk` (tool ahead):** here the locality call dmd is good at — fuse the
  two passes / kill the transposed sweep / pick i-k-j — was the dominant lever, and the
  hint nudged the tool agent onto it cleanly.

**Takeaway.** dmd is a genuine aid for the *structural/locality* decision (loop order,
tiling, fusion) and, constrained to that, roughly matches a strong unaided optimizer.
But it is blind to the ILP/register/SIMD levers that often decide real runtime, and it
carries a 1.6–2.5× token premium. Best used as a cheap directional sanity-check on loop
structure, not as the optimizer of record — measure runtime for the final call.

_Artifacts: `mlirperf/kernels/*` (ref.mlir + driver.c), `mlirperf/runs/<slot>/{opt*.mlir,
rationale.md}`, `mlirperf/batch_eval.py`, `results.json`, `mlir_metrics.json`._
