# DMD-Tool Effectiveness — Real-Performance Refactor

**Question (revised).** When the ground truth is **real single-core wall-clock runtime**
(not a symbolic data-movement model), does giving an agent the `dmd` locality analyzer
make it a better or cheaper kernel optimizer than an agent reasoning from first
principles?

## Design

- **12 PolyBench-style kernels**, each a C `kernel()` over flat `double*` arrays.
- **24 agents** = 12 kernels × {`tool`, `ctrl`}, identical `claude` base, identical
  performance-only task. The sole manipulation: `tool` may use the `dmd` analyzer
  (MCP or CLI); `ctrl` may not. Neither group may benchmark/time its own code — both
  optimize "blind" to wall-clock; the only extra signal `tool` gets is the analyzer.
- Agents were told a **size *range*, not the exact eval N**, and had to stay correct
  for any N (remainder handling) — so they couldn't overfit a single size.

## Evaluation (ground truth)

`eval.py`: build the agent's `opt.c` with `clang -O3 -march=native -funroll-loops`,
linked to a fixed driver.
1. **Correctness** = `numpy.allclose(ref_out, opt_out, rtol=1e-6, atol=1e-9)` on the
   **full output arrays** (not a checksum) at multiple sizes incl. non-tile-multiples.
2. **Performance** = `hyperfine` wall-clock, pinned to one core (`taskset -c 0`),
   speedup = `t_ref / t_opt` at an undisclosed size in the kernel's range.

## Results — speedup vs the naive reference

| kernel | tool × | ctrl × | winner |
|--------|------:|------:|--------|
| matmul | 18.69 | 19.51 | ctrl |
| gemm | 18.84 | 20.91 | ctrl |
| 2mm | 9.84 | **17.40** | ctrl |
| 3mm | 16.64 | 16.53 | tie |
| mvt | 2.61 | 2.46 | tool |
| atax | **0.94** | 1.01 | ctrl (tool *regressed* below 1×) |
| bicg | 1.02 | 1.01 | tie |
| gemver | 2.38 | 2.74 | ctrl |
| gesummv | 1.00 | 1.02 | tie |
| syrk | 24.88 | 23.76 | tool |
| doitgen | 2.65 | **4.49** | ctrl |
| covariance | 38.13 | 20.71 | tool |

**Every one of the 24 kernels built and was numerically correct (24/24).**

## Aggregate

| metric | TOOL | CTRL | T/C |
|--------|-----:|-----:|----:|
| **geometric-mean speedup** | 5.56× | **5.91×** | 0.94 |
| mean speedup | 11.47× | 10.96× | 1.05 |
| head-to-head wins (>3%) | 3 | **6** | (3 ties) |
| regressions (<1×) | **1** | 0 | — |
| correct / 12 | 12 | 12 | — |
| output tokens (contestant only) | 173,504 | 86,821 | **2.00×** |
| output tokens (incl. analyzer sub-agents) | 372,458 | 86,821 | **4.29×** |

## Verdict

**On real single-core runtime, the dmd locality tool did not help — and slightly hurt.**
The control group has the higher geometric-mean speedup, wins twice as many head-to-head
matchups (6 vs 3), produced zero regressions, and did it at **half to a quarter of the
token cost**. The tool group's one sub-1× regression (`atax`) and its worst relative
losses (`2mm` 9.8× vs 17.4×, `doitgen` 2.65× vs 4.49×) all trace to the same cause:

> The analyzer optimizes **modeled data movement** (reuse distance / cache traffic).
> Real single-core throughput here is governed by **vectorization, register/ILP
> blocking, and streaming bandwidth** — which the model does not capture. When the
> tool steered an agent toward a DMD-optimal *tiling*, it sometimes displaced the
> register-blocked, SIMD-friendly micro-kernel that the unaided agents reached directly
> and that the hardware actually rewards.

Where the tool won (`covariance` 38×, `syrk`, `mvt`) the locality insight — kill the
transposed access, block for cache — happened to coincide with the runtime win, but the
control agents independently found the same class of transform.

### Contrast with the earlier (symbolic-DMD) experiment

The first experiment scored variants with the dmd analyzer itself, and the tool group
came out ahead — unsurprising, since that *is* the metric the tool optimizes. **Swapping
the ground truth to measured wall-clock erases the tool's edge.** The practical lesson:
a data-movement model is a proxy; for an automated *performance* optimizer it adds cost
and an occasional misleading signal, while the decisive levers (SIMD, registers, ILP,
bandwidth) live outside the model. Use the analyzer to *explain* locality, not to *pick*
the fastest kernel — measure runtime for that.

_Artifacts: `perf/kernels/*` (ref + driver), `perf/runs/<slot>/{opt.c,rationale.md}`,
`perf/eval.py`, `perf/perf_results.json`, `perf/perf_metrics.json`._
