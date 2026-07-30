# locality-regimes

A fresh analysis of how to read and use the output of the algebraic
locality compiler (symbolic reuse-interval distributions -> cache-size
and miss-ratio polynomials), replacing the earlier data-movement-
complexity reading. Built entirely from raw `dmd-cli` output under the
paper's canonical configuration: infinite repeat, scale approximation,
block size 8.

**Read `REPORT.md` (or `REPORT.pdf`).** In one sentence: collapse the
symbolic distribution to a scalar and its leading term and you are
governed by exactly the levels whose probability vanishes (RI Sum
Invariance forces this, and scalar rankings demonstrably invert at real
cache sizes); read it instead as a finite list of locality *regimes* —
polynomial cache-size boundaries with rational miss-ratio plateaus —
and prediction, problem-size planning, cache provisioning,
transformation accounting (interchange 4.5x, tiling 36–106x, pointwise
in cache size), co-scaling laws (a sqrt(2)-rule with per-kernel validity
windows), and parallel slicing analysis (by parameter substitution)
each become a single polynomial evaluation or inversion.

Pipeline: `run_suite.py` -> `data/` (raw analyzer JSON) -> `regimes.py`
-> `regimes/` (exact level structure) -> `derived.py`,
`parallel_study.py`, `anchor_checks.py` -> `tables/`. All symbolic
statements are exact (Fraction arithmetic, fitted on a residue class and
verified on held-out points); all concrete numbers are evaluated from
the raw guarded quasi-polynomials. Mass conservation and RI Sum
Invariance run as self-checks; kernels whose analyzer output fails them
are excluded and listed rather than silently used.
