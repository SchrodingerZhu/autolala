# locality-regimes

**Read `REPORT.md` (or `REPORT.pdf`).** It is written for a reader who
knows only that affine kernels (matmul, convolutions, stencils) matter
and that their scaling behavior is worth measuring — and it builds up,
from scratch, the discovery this branch studies: for affine kernels,
memory behavior has *closed form*. A compiler reduces an n³-step
execution to a few rows of formulas (reuse distances and their
frequencies, symbolic in every loop bound), and from those rows you can
read off — without running anything — miss ratios at any cache and
problem size, the problem sizes where performance falls off a cliff,
whether and when tiling pays and by how much (36–106x for matmul, with
the exact condition n > sqrt(C/8)), which parallel decompositions are
provably useless and which core count merges private caches into one,
and conservation laws that catch broken analyses automatically.

Everything is derived fresh from `dmd-cli` output (infinite repeat,
scale approximation, 64-byte lines) over 22 conserving PolyBench
kernels plus a matmul variant family. Pipeline and tables:
`run_suite.py` → `regimes.py` → `derived.py` / `parallel_study.py` /
`anchor_checks.py` → `tables/`. Raw analyzer JSON (`data/`, 29 MB) is
gitignored; regenerate with `run_suite.py`.
