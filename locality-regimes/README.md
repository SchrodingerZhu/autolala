# locality-regimes

**Read `REPORT.md` (or `REPORT.pdf`).** Its claim, in one line: for
fifty years the reuse-distance distribution — the object that
determines a kernel's miss ratio at every cache size — could only be
*measured* (one number per run, one curve per profiled input, or
exponents with no constants); for affine kernels it can now be
*derived*, as a four-row family of formulas symbolic in every loop
bound, and formulas do what no measurement campaign can: make for-all
statements (gemm's miss ratio provably flat from 32 KB to 31 MB;
row-parallel matmul's traffic provably independent of worker count),
correct accepted classifications (a kernel certified "no headroom" by
asymptotics hides a 16x waste at L1; no scalar locality score can rank
kernels, provably), upgrade folklore to theorems (the sqrt(2) rule gets
per-kernel validity windows), compose under substitution (the exact
core count where private caches merge into one, and the cache-line
floor that blocks it), and audit themselves (conservation laws caught
six broken analyzer outputs).

Everything is derived fresh from `dmd-cli` output (infinite repeat,
scale approximation, 64-byte lines) over 22 conserving PolyBench
kernels plus a matmul variant family. Pipeline and tables:
`run_suite.py` → `regimes.py` → `derived.py` / `parallel_study.py` /
`anchor_checks.py` → `tables/`. Raw analyzer JSON (`data/`, 29 MB) is
gitignored; regenerate with `run_suite.py`.
