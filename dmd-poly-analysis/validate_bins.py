#!/usr/bin/env python3
"""End-to-end audit: the analyzer's RD distribution, evaluated at a concrete
size, must reproduce the brute-force trace histogram exactly (exact method).

For every kernel-model requested: bind all params to n, evaluate every bin —
enumerating exposed iterators where present — into a value -> count map, and
compare with dsl_sim's histogram. Distances and populations must both match.
"""
import json
import os
import sys
from collections import Counter
from fractions import Fraction

import dsl_sim
from qpeval import Piecewise, compile_expr, call, free_names
from region_enum import enum_region

HERE = os.path.dirname(os.path.abspath(__file__))


def analyzer_hist(rec, params, n):
    """value -> count reconstructed from the bins at size n (exact)."""
    hist = Counter()
    for e in rec["rd"]:
        iters = sorted((free_names(e["value_plain"])
                        | set().union(*[free_names(r["domain_plain"])
                                        | free_names(r["count_plain"])
                                        for r in e["regions"]] or [set()]))
                       - set(params))
        env = {p: Fraction(n) for p in params}
        if not iters:
            v = Piecewise(e["value_plain"])(env)
            m = Piecewise(e["mass_plain"])(env)
            if m:
                hist[Fraction(v)] += m
            continue
        vfn = compile_expr(e["value_plain"], "val")
        lim = 2 * n + 64
        for r in e["regions"]:
            for pt, c in enum_region(r["domain_plain"], r["count_plain"],
                                     iters, env, lim, "exact", Fraction):
                hist[Fraction(call(vfn, pt))] += c
    return {k: v for k, v in hist.items() if v}


def check(name, n, model):
    d = json.load(open(f"{HERE}/results/{name}.json"))
    rec = d.get(model)
    if not rec or rec.get("method") != "exact":
        return f"{name}.{model}@{n}: skipped (no exact record)"
    dsl = open(f"{HERE}/dsl/{name}.dsl").read()
    pline = next(l for l in dsl.splitlines() if l.startswith("params"))
    params = [p.strip().rstrip(";") for p in pline.split(None, 1)[1].split(",")]
    binds = {p: n for p in params}
    sim = dsl_sim.stats(dsl, binds, repeat=2 if model == "inf" else 1)
    ah = analyzer_hist(rec, params, n)
    sh = {Fraction(k): v for k, v in sim["hist"].items()}
    if ah == sh:
        return f"{name}.{model}@{n}: OK ({len(sh)} distinct distances, " \
               f"{sum(sh.values())} reuses)"
    only_a = {k: v for k, v in ah.items() if sh.get(k) != v}
    only_s = {k: v for k, v in sh.items() if ah.get(k) != v}
    return (f"{name}.{model}@{n}: MISMATCH\n"
            f"  analyzer-only/diff: {dict(sorted(only_a.items())[:8])}\n"
            f"  sim-only/diff:      {dict(sorted(only_s.items())[:8])}")


if __name__ == "__main__":
    kernels = sys.argv[1:] or ["sym_gemm", "sym_syrk", "sym_trisolve",
                               "sym_jacobi-1d", "sym_mvt", "sym_trmm"]
    for name in kernels:
        for model in ("single", "inf"):
            for n in (24, 26):
                try:
                    print(check(name, n, model), flush=True)
                except Exception as exc:
                    print(f"{name}.{model}@{n}: ERROR {exc!r}", flush=True)
