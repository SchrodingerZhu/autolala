# Plan: Claude Skills & Tools for Analyzing Affine Programs

A design for exposing this workspace's affine-program analysis stack to Claude
agents (Claude Code CLI, and API/SDK agents) so they can ingest MLIR or DSL,
run locality / data-movement (DMD) analysis, interpret the results, and propose
loop transformations.

## 1. Goal

Let an agent go from *"here is an affine kernel"* to *"here is its locality
behavior, the bottleneck, and a better-tiled variant"* without a human knowing
the CLI flags or the JSON layout. The agent should:

1. Ingest a program (tagged MLIR loop **or** DSL source).
2. Run RI / RD / DMD / parallel-CRI analysis.
3. Read the structured output and explain it in domain terms.
4. Propose and *re-evaluate* transformation variants.
5. Fail legibly on unsupported input (point at the offending op).

## 2. Capability inventory (what already exists)

| Component | Surface | Use to the agent |
| --- | --- | --- |
| `mlir-extract` (lib + bin) | `extract_dsl_from_source(src, attr)`; `report_error`; ariadne CLI | MLIR affine loop → DSL; structured rejection w/ source spans |
| `dmd-core` | `parse_program`, `validate_program`, `analyze_program`/`analyze_source`, `AnalysisReport` (serde) | DSL parse/validate/lower/analyze; RI, RD, DMD, parallel CRI |
| `dmd-cli` | `--input`/stdin, `--json`, `--block-size`, `--num-sets`, `--max-operations`, `--approximation-method` | scriptable JSON frontend |
| `dmd-playground` | Axum `POST /api/tasks`, `GET /api/tasks/{id}` | bounded, async HTTP analysis backend |

The executable substrate is therefore **already done**. The work is *packaging*
it into agent-consumable surfaces and *teaching the model* how to drive and read
it.

## 3. Architecture: three layers

```
  ┌─ Subagents ──────────────────────────────────────────┐
  │  affine-locality-analyst / affine-optimizer           │  autonomy + orchestration
  │     └─ uses Skills + Tools, sweeps variants           │
  ├─ Skills (.claude/skills/*) ──────────────────────────┤
  │  analyze-affine / mlir-to-dsl / transform-advisor     │  procedural + domain knowledge
  │     └─ instructions + interpretation guide            │
  ├─ Tools (MCP server / CLI-as-tool) ───────────────────┤
  │  extract / validate / analyze / compare               │  typed, deterministic execution
  └───────────────────────────────────────────────────────┘
```

Each layer is independently useful; build bottom-up.

### Layer 1 — Tools (the executable contract)

**Option A — CLI-as-tool (fast interim).** Agents call `mlir-extract` and
`dmd-cli --json` over `Bash`, with a project allowlist (see the
`fewer-permission-prompts` skill). Zero new code; the agent must know the flags
and parse JSON. Good for a Phase-1 win.

**Option B — `crates/dmd-mcp`, a stdio MCP server (recommended, durable).**
A new crate wrapping `dmd-core` and `mlir-extract` *in-process* and exposing
typed tools over MCP (works in Claude Code, the Agent SDK, and headless/cron):

- `extract_from_mlir(mlir, attr="dmd.extract")` → `{ dsl, diagnostics[] }`
- `validate_dsl(dsl)` → `{ ok, errors[] }` (each error: message + span)
- `analyze_dsl(dsl, options)` → `AnalysisReport` (RI / RD / DMD / access counts / parallel CRI)
- `analyze_mlir(mlir, attr, options)` → extract **then** analyze in one call
- `compare_variants(variants: {label, dsl}[], options)` → metric table (DMD, warm/compulsory, peak RD) for ranking transformations

Implementation notes:
- Built on `tower-mcp` (Tower-native MCP). Tool input types derive
  `schemars::JsonSchema`, so the advertised `inputSchema` is generated from typed
  Rust structs rather than hand-written. Reuse `dmd-core` types directly — no
  subprocess, no flag guessing.
- **Serialize analysis calls behind a mutex.** Per the README, in-process
  Barvinok init "is not stable under parallel in-process calls"; the server must
  not run two analyses concurrently.
- Enforce guardrails inside the tool: input-size cap, `max-operations`, and a
  wall-clock timeout (mirror the playground's bounded execution).
- Register via project `.mcp.json` so it loads automatically.

**Option C — remote tool** backed by the existing playground HTTP API, for
shared/hosted use. Lower priority.

### Layer 2 — Skills (procedural + domain knowledge)

Skills live in `.claude/skills/<name>/SKILL.md` (project-scoped). Each carries a
`description` trigger, step-by-step instructions that call Layer-1 tools, worked
examples, an **output-interpretation guide**, and failure handling.

- **`analyze-affine`** — entry point. Detect MLIR vs DSL; (extract →) validate →
  analyze; summarize: the DMD formula, dominant reuse-distance regions, warm vs
  compulsory traffic, and whether the kernel *scales* (named dims survive in the
  formula). When to widen `--num-sets` / change `--approximation-method`.
- **`mlir-to-dsl`** — tag the target loop with `dmd.extract`, run extraction,
  and translate ariadne rejections into concrete fixes (e.g. `mod` → tile; `min`
  bound → split loop; `memref.load` → raise to `affine.load`).
- **`affine-transform-advisor`** — from RI/RD, hypothesize tiling / interchange /
  fusion; synthesize variant DSLs; call `compare_variants`; recommend the winner
  with the metric delta.
- **`author-affine-dsl`** — help write DSL kernels from a description (grammar,
  `params`/`array`/`for`/`if`/`read|write|update`), then validate.

Skills must *teach the domain*, not just the commands — the interpretation guide
is the high-value part (see Layer-4 glossary).

### Layer 3 — Subagents (autonomy + scale)

Agent definitions in `.claude/agents/*.md`, tooled with `Bash` + the MCP server:

- **`affine-locality-analyst`** — ingest a program (or a directory of kernels),
  run analyses, and produce a report that cites source loops by location.
- **`affine-optimizer`** — propose N transformation variants, evaluate each
  (worktree isolation if it edits files), and rank by DMD/locality. A natural fit
  for a fan-out workflow: discover variants → analyze in parallel → synthesize.

## 4. Prerequisite: stabilize the data contract

Agents act on structure, not prose. Before/with Layer 1:

- **Freeze versioned JSON schemas** for `ExtractResult`, `Diagnostic`
  (message + `{start_line,start_col,end_line,end_col}` + help), and
  `AnalysisReport` (RI entries, RD entries, DMD terms, access counts, parallel
  CRI laws). `AnalysisReport` is already serde-derived — document field meanings
  and **units**.
- **Make errors machine-readable**: a stable `kind` discriminant plus span, so a
  skill can branch on the failure instead of regex-matching English. `dmd-core`
  errors carry byte offsets; `mlir-extract` errors carry spans — surface both in
  the tool layer.

## 5. Interpretation knowledge base (Layer 4)

`docs/affine-analysis-glossary.md`, referenced by every skill:

- **Glossary**: reuse interval (RI), reuse distance (RD), data-movement distance
  (DMD), CRI, compulsory vs warm traffic, `block-size`, `num-sets`,
  approximation methods.
- **Formula semantics**: DMD as a sum of `sqrt` over RD regions; what the
  `--approximation-method=scale` filter drops and why; how the scaling check
  keeps named dims (`M`, `N`, `K`) visible.
- **Parallel model**: `parallel(T) for` round-robin CRI, single-parallel-loop
  limit, the racetrack/negative-binomial RI laws.
- **Known limits** (so the agent sets expectations): one parallel loop;
  `mod`/`ceildiv` and non-affine mul/div unsupported; dynamic memref extents
  become *independent* params; load→read, store→write (no auto `update`).

## 6. Evaluation & safety

- **Golden corpus**: matmul, stencil, gemm, conv, triangular-`if`, parallel
  kernels — each with expected metric ranges. A harness runs the tools and
  checks both the numbers and the agent's interpretation against rubric points.
- **Determinism / cost**: Barvinok is serialized and can be slow → enforce
  `max-operations` + timeouts at the tool boundary; never fan out analyses
  in-process.
- **Resource guardrails**: input-size and recursion caps in the MCP tool;
  read-only by default (the analyst subagent shouldn't edit files).

## 7. Rollout phases

| Phase | Deliverable | Outcome |
| --- | --- | --- |
| 0 | Confirm/stabilize `dmd-cli --json` schema + structured errors | machine-readable contract |
| 1 | `analyze-affine` skill driving the CLIs via Bash allowlist | usable end-to-end today |
| 2 | `crates/dmd-mcp` MCP server (in-process, mutex-guarded) + `.mcp.json` | typed, reusable tools |
| 3 | Skill suite (`mlir-to-dsl`, `transform-advisor`, `author-dsl`) + subagents | autonomous analysis & optimization |
| 4 | Glossary doc + golden corpus + eval harness | trustworthy, regression-guarded |

## 8. Concrete next deliverables

- `crates/dmd-mcp/` — MCP server (tools in §3, Layer 1, Option B).
- `.claude/skills/analyze-affine/SKILL.md` (+ the three siblings).
- `.claude/agents/affine-locality-analyst.md`, `.claude/agents/affine-optimizer.md`.
- `docs/affine-analysis-glossary.md`.
- `.mcp.json` registering `dmd-mcp`; Bash allowlist entries for the interim CLIs.
- Golden corpus + harness under `crates/dmd-mcp/tests/` (or `examples/`).

## 9. Open decisions (need a call)

1. **First surface**: CLI-as-tool (ship this week) vs MCP server (more work, more
   durable)? Recommendation: do both — CLI in Phase 1, MCP in Phase 2.
2. **MCP backing**: in-process `dmd-core` (fast, but must serialize Barvinok) vs
   subprocess `dmd-cli` (isolation, slower). Recommendation: in-process +
   mutex; subprocess only if isolation/timeouts prove hard in-process.
3. **Skill scope**: project `.claude/skills` (shipped with the repo) vs personal.
   Recommendation: project, so collaborators inherit them.
4. **Transform autonomy**: advisor-only (suggest) vs optimizer (edit + verify in
   a worktree)? Recommendation: start advisor-only.
