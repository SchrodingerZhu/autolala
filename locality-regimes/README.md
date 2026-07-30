# locality-regimes

**Read `REPORT.md` (or `REPORT.pdf`).** The algebraic locality compiler
turns a kernel's memory behavior into exact formulas in all loop bounds
at once; this branch uses it as an instrument and reports what it
*finds*. Headline findings, all derived (nothing profiled), all passing
the framework's built-in conservation gates:

1. **Linear attention is provably context-length-free**: 99% of its
   accesses have reuse-distance formulas containing no n; its only
   cliff is in head dimension — state residency at d ≈ 62 (fp64) / 90
   (fp32) / 128 (bf16) for a 32 KB L1, exactly the head sizes in use,
   with a 43x per-token traffic jump on crossing.
2. **Softmax attention's cliff is n\* = C/d** (50x per-token jump,
   landing precisely between the predicted sample points), and below
   it 100% of DRAM traffic is the materialized score matrix — the
   FlashAttention payoff, now with constants and a validity region.
3. **Chunked linear attention has a computable memory-free chunk
   window**; the recurrent form is the traffic floor, and the tables
   catch a case (chunk 256 at 1 MB) where the naive working-set
   argument predicts free and the phase structure actually costs 2.5x.
4. Across PolyBench: staircase boundaries are quantized at rational
   powers of the footprint; no scalar data-movement score can rank
   kernels (structural argument + concrete misrankings); parallel
   slicing laws by parameter substitution (row-slicing provably
   traffic-neutral, column-slicing merges p caches at p = n²·8B/C).

Pipeline: `run_suite.py` → `regimes.py` → `derived.py` /
`parallel_study.py` / `anchor_checks.py` / `attention_study.py` →
`tables/`. Exact Fraction arithmetic throughout; region-domain gating;
conservation + sum-rule self-checks (six inconsistent stencil outputs
excluded, not absorbed). Raw analyzer JSON (`data/`) is gitignored;
regenerate with `run_suite.py`.
