# DMD-Tool Effectiveness — Results

**Setup.** 12 `claude` agents, matched in pairs over 6 affine kernels
(`matmul, gemm, mvt, bicg, syrk, gesummv`, drawn from `../autolala`). One agent per
pair was *required* to use the `dmd` locality analyzer (`tool`), the other was
*forbidden* it and had to reason from cache-locality first principles (`ctrl`).
Every produced `optimized.mlir` was scored by the same deterministic ground-truth
pipeline (`score.py`: `mlir-extract → dmd-cli --json`, DMD formula evaluated at
N = 65536, block size 64). Lower DMD = less data movement. `R = baseline_DMD /
optimized_DMD` (improvement factor; R<1 = regression).

## Manipulation check (from transcripts, not self-reports)

| group | `mcp__dmd__*` calls | ran dmd engine? | how |
|-------|--------------------:|-----------------|-----|
| `tool` (6) | each spawned `affine-locality-analyst` children | **yes** | delegated; children ran `mlir-extract`+`dmd-cli` (MCP server was unreachable in-context, so they used the identical CLI engine as a fallback) |
| `ctrl` (6) | **0** | **no** | pure reasoning; transcripts confirm zero analysis runs |

The independent variable held: the tool group measured data movement, the control
group never did.

## Per-kernel outcome

| kernel | baseline DMD | tool R | ctrl R | winner | note |
|--------|-------------:|-------:|-------:|--------|------|
| matmul | 1.08e17 | **49.1×** | **49.1×** | tie | both did textbook 32³ tiling → byte-identical result |
| syrk   | 7.66e14 | **1.90×** | **1.90×** | tie | both interchanged k/j → identical result |
| gemm   | 3.70e16 | 38.9× | **43.0×** | ctrl | analyzer nudged tool to a slightly more conservative tiling |
| bicg   | 3.12e10 | 1.17× | **1.20×** | ctrl | marginal; tool burned a huge token budget for a worse result |
| mvt    | 1.13e12 | **55.5×** | 31.3× | **tool** | tool found fusion+tiling; ctrl only tiled |
| gesummv| 5.47e10 | **1.64×** | 0.66× | **tool** | **ctrl shipped a REGRESSION** (made it 1.5× worse); tool's measurement caught it |

All 12 variants preserved total memory accesses (validity gate passed) — no agent
cheated by deleting computation. `gesummv/ctrl` is a *legal* transform that genuinely
increases modeled data movement.

## Group aggregates

| metric | TOOL | CTRL | T/C |
|--------|-----:|-----:|----:|
| geometric-mean improvement (geoR) | **8.54×** | 6.81× | 1.25× |
| regressions (R<1) out of 6 | **0** | 1 | — |
| output tokens (self + analyzer children) | 327,793 | 38,051 | **8.61×** |
| output tokens (contestant transcript only) | 150,214 | 38,051 | 3.95× |
| pooled efficiency E = Σln(R) / kTok | 0.039 | **0.303** | 0.13× |

## Sophisticated metric: where does measurement pay for itself?

Token-efficiency `E = ln(R)/kTok` (log-improvement per 1k output tokens) **favors the
control group ~8×** — because the cheap, textbook wins (matmul, syrk, the bulk of
gemm) are available to a competent reasoner *for free*, and the analyzer's
delegation-heavy workflow is expensive. The tool's value is **not** uniform; it
concentrates in two regimes:

1. **Regression insurance.** The single most valuable event in the experiment is
   negative: `ctrl-gesummv` confidently shipped a transform that *worsened* locality
   (R = 0.66). The tool group's measurement loop caught and avoided this (R = 1.64).
   Across the board, **tool regressions = 0 vs ctrl = 1/6 (17%)**. For an automated
   optimization pipeline, a 17% "confidently-wrong" rate is the expensive failure
   mode, and the analyzer eliminates it.

2. **Non-obvious transform discovery.** On `mvt` the analyzer's feedback led the tool
   agent to *fuse* the two matrix-vector passes (so `A` streams once) on top of
   tiling — R = 55× vs the control's tiling-only R = 31×. Measurement surfaced a win
   that first-principles reasoning left on the table.

On the other four kernels the analyzer added cost without changing the decision
(2 ties) or slightly hurt it (2 marginal ctrl wins, incl. one where the tool agent
recursively spawned a sub-analyst and spent **140k tokens** to land *below* the
control's 16k-token answer on bicg).

### Composite verdict — "Value of Measurement" (VoM)

> **The dmd tool buys reliability, not raw speed.** It lifts geometric-mean DMD
> reduction by **1.25×** and drives the confidently-wrong regression rate from
> **17% → 0%**, at a cost of **~4–8.6× more output tokens**. It changed the outcome
> on **2 / 6** kernels (both the hard, non-textbook ones); on the 4 kernels with a
> well-known optimal transform it was pure overhead.

**Implication.** Gate the analyzer on difficulty: let an unaided agent take the
first pass cheaply, and invoke the dmd tools (a) to *verify no regression* before
accepting any transform, and (b) to *break ties / explore* on kernels where the
locality-optimal transform is non-obvious (fusion, transposed accesses, multi-nest
kernels). That captures the tool's entire measured upside — regression insurance +
hard-case discovery — while avoiding its 8× cost on the easy majority.

_Artifacts: `baselines.json`, `scores.json`, `results_final.json`, `token_audit.json`,
`score.py`, `runs/<slot>/{optimized.mlir,rationale.md}`._
