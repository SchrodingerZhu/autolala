#!/usr/bin/env python3
"""Parallel extrapolation by parameter substitution (case study: gemm).

The cache polynomials are functions of every loop bound separately.  A
data-parallel decomposition that gives each of p workers a contiguous slice
of one loop is, for the per-worker access stream, exactly the original
kernel with that loop's bound replaced by n/p.  The per-worker miss-ratio
staircase therefore needs no new analysis: it is a parameter substitution
in the already-derived polynomials.  Aggregate traffic through p private
caches of C lines each is p * accesses(worker) * mr_worker(C).

This models private caches and ignores coherence and interleaving; it is a
first-order reading of the symbolic model, not a validated multicore
model.  Its value is the boundary structure: substitution shows which
regime boundaries shrink with p (aggregate cache capacity helps) and which
are invariant (it cannot help).

gemm (PolyBench, C_out = alpha*A*B + beta*C_out) in the extracted DSL has
bounds p0 (i), p1 (j), p2 (k).  The array written is (i x j), the inputs
are A (i x k) and B (k x j).  The whole-matrix regime boundary of the
kernel is the re-sweep of B (k x j) across iterations of i.

Output: tables/parallel.md
"""
import os
from fractions import Fraction

from derived import Kernel, fmt_mr, load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables")

N = 2016
CACHE = 512          # 32 KB in 64-byte lines
CACHE2 = 16384       # 1 MB
PS = [1, 2, 4, 8, 16, 32, 63, 126, 252, 504, 1008]


def env_for(k, sliced, p):
    env = {}
    for name in k.param_names:
        env[name] = Fraction(N)
    env[sliced] = Fraction(N, p)
    assert env[sliced].denominator == 1
    return env


def main():
    k = load("sym_gemm", "inf")
    lines = [f"# gemm under p-way slicing, n = {N} (block 8, infinite repeat)",
             "",
             "Per-worker miss ratio from parameter substitution; aggregate "
             "traffic = p x worker accesses x worker miss ratio, private "
             f"caches of {CACHE} lines (32 KB) and {CACHE2} lines (1 MB).",
             ""]
    for sliced, label, note in [
            ("p0", "i-slice (rows of the output)",
             "the re-swept matrix B (k x j) stays whole in every worker"),
            ("p1", "j-slice (columns of the output)",
             "each worker re-sweeps only an n x n/p slice of B"),
            ("p2", "k-slice (reduction; needs partial-sum combination)",
             "each worker re-sweeps an n/p x n slice of B")]:
        lines.append(f"\n## {label}\n")
        lines.append(f"Boundary reading: {note}.\n")
        lines.append("| p | worker accesses | worker mr @ 32 KB | aggregate "
                     "traffic @ 32 KB (lines) | worker mr @ 1 MB | aggregate "
                     "@ 1 MB |")
        lines.append("|---|---|---|---|---|---|")
        for p in PS:
            env = env_for(k, sliced, p)
            entries, total = k.raw_entries_env(env)
            mr1 = k.mr_env(CACHE, env)
            mr2 = k.mr_env(CACHE2, env)
            t1 = p * total * mr1
            t2 = p * total * mr2
            lines.append(f"| {p} | {float(total):.3g} | {fmt_mr(mr1)} | "
                         f"{float(t1):.3g} | {fmt_mr(mr2)} | {float(t2):.3g} |")
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/parallel.md", "w").write("\n".join(lines))
    print("wrote tables/parallel.md")


if __name__ == "__main__":
    main()
