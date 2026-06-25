# DMD-Tool Effectiveness Experiment

**Question.** Do the `dmd` MCP locality-analysis tools make a Claude agent measurably
better and/or cheaper at optimizing affine loop nests, versus an agent that must
reason from first principles with no analyzer feedback?

## Design — matched-pair A/B

- **6 kernels** drawn from `../autolala` polybench/symbolic + a canonical matmul:
  `matmul, gemm, mvt, bicg, syrk, gesummv`.
- **2 conditions** per kernel → **12 contestant agents** (`claude` base agent for all,
  so the only difference is the independent variable):
  - **`tool`** — *must* use `mcp__dmd__analyze_mlir` / `analyze_dsl` / `compare_variants`
    to measure data movement and iterate against the analyzer's feedback.
  - **`ctrl`** — *forbidden* from any `mcp__dmd__*` tool and from running `dmd-cli`,
    `mlir-extract`, or `score.py`. Pure reasoning about cache locality.
- The manipulation is **verified post-hoc** from each agent's transcript (did `mcp__dmd__*`
  tool calls appear or not). Non-compliant runs are flagged.

## Task given to each agent

Transform `original.mlir` into a semantics-preserving, lower-data-movement
`optimized.mlir` (loop interchange / tiling / fusion / skewing — **no deleting
computation**). Keep the `{dmd.extract}` tag on the outer loop so it stays scorable.
Write `rationale.md`.

## Ground-truth scoring (deterministic, agent-free)

`score.py`: `optimized.mlir --(mlir-extract)--> DSL --(dmd-cli --json)--> DMD formula`,
then evaluate the symbolic DMD formula at every parameter = **N = 65536**, block size
**64**. Lower DMD = less predicted data movement. Validated to rank
`naive matmul (1.1e17) > interchanged (3.6e16) > tiled (2.2e15)` correctly.

**Validity gate.** Optimized kernel must (a) extract, (b) analyze, and (c) preserve
total memory accesses within 2× of baseline (guards against degenerate "optimizations"
that just drop accesses).

## Metrics

For agent *a* on kernel *k* (baseline DMD `D0`, optimized DMD `D`):
- **Improvement ratio** `R = D0 / D` (>1 is better; ≤1 = no help / regression).
- **Cost** = output tokens spent by the agent (from its `subagents/agent-*.jsonl`).
- **Token efficiency** `E = ln(R) / (output_tokens / 1000)` — log-improvement per kilo-token.

Group-level (tool vs ctrl):
- mean `R`, success rate (`R>1` under validity gate), mean tokens, mean `E`.
- **Tool ROI** = `E_tool / E_ctrl` and `meanR_tool / meanR_ctrl`.

Results land in `RESULTS.md` + `results.json`.
