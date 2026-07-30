#!/usr/bin/env python3
"""Extract the locality-regime structure of each kernel from raw dmd-cli JSON.

Input:  data/<kernel>.<model>.json   (model in {single, inf})
Output: regimes/<kernel>.<model>.json

For one kernel the analyzer emits a fine-grained list of (rd, mass) pairs:
rd is the exact number of distinct cache blocks touched inside one reuse
window (the LRU stack distance of that reuse, block granularity), mass the
number of accesses with that rd; both are polynomials in the size parameter
once all program parameters are bound to a single n.

This module:
 1. binds every parameter to n and recovers exact polynomial coefficients
    for every rd value, every mass, and the total access count;
 2. clusters rd entries into LEVELS by asymptotic scale: two entries belong
    to the same level iff their rd polynomials have the same degree and the
    same leading coefficient (they then differ by lower-order terms only);
 3. orders levels by rd magnitude and computes, exactly,
      portion  p_k(n) = (sum of masses at level k) / total(n)
      plateau  m_k(n) = cold(n)/total(n) + sum_{j>k} p_j(n)
    so that m_k is the miss ratio of an LRU cache just large enough to hold
    every reuse at levels 1..k (cold = compulsory accesses, zero under the
    infinite-repeat model);
 4. does the same clustering for the RI (time-scale) distribution and marks
    levels whose RI grows as fast as the whole trace: those reuses span a
    constant fraction of one execution and are imaginary (cross-run) or
    outer-loop-carried reuses.

Sampling for the exact fits uses n = multiples of 8400 so that one branch
of every quasi-polynomial (moduli up to 8 observed, plus slack) is active;
the recovered polynomials are exact on that residue class and asymptotically
representative.
"""
import json
import math
import os
import sys
from fractions import Fraction

import sympy as sp

from qp import (Piecewise, QPError, domain_satisfiable, fit_polynomial,
                poly_str)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = STEP = 8400
NREF = 84000  # reference size for ordering levels

# Kernels whose parameters live on different scales: map param -> multiplier
# of n (as a Fraction).  Everything else binds every parameter to n.
BINDINGS = {
    # 2D convolution: image p0 x p0, filter p1 x p1 with p1 < p0; bind the
    # filter side to n/4 so the generic (large-parameter) branches stay active
    "sym_convolution": {"p1": Fraction(1, 4)},
}

n = sp.Symbol("n", positive=True)


def to_sympy(coeffs):
    return sum(sp.Rational(c) * n ** i for i, c in enumerate(coeffs))


def bind_all(pw, binding):
    """Return f(n) evaluating the piecewise QP with parameters bound to
    multiples of n according to `binding` (default multiplier 1)."""
    def f(nv):
        env = {}
        for p in pw.params:
            m = binding.get(p, Fraction(1))
            v = Fraction(nv) * m
            if v.denominator != 1:
                raise ValueError(f"binding {p}={m}*n not integral at n={nv}")
            env[p] = v
        return pw(env)
    return f


def fit(pw, binding, base):
    return fit_polynomial(bind_all(pw, binding), max_degree=8,
                          base=base, step=base)


def load_entries(dist, binding, base, param_names):
    """Compile and exactly fit one distribution: [(v_coeffs, w_coeffs)].

    Each entry is gated by the satisfiability of its region domains: piece
    guards are printed gisted against those domains, so outside them the
    polynomials are meaningless.  Non-parameter names in a domain are
    existentially quantified (Fourier-Motzkin over the rational relaxation).
    """
    out = []
    for e in dist:
        v = Piecewise(e["value_plain"])
        w = Piecewise(e["mass_plain"])
        domains = [r["domain_plain"] for r in e.get("regions", [])]

        def gated(pw, doms=domains):
            def f(nv):
                penv = {}
                for p in param_names:
                    val = Fraction(nv) * binding.get(p, Fraction(1))
                    if val.denominator != 1:
                        raise ValueError(f"non-integral binding at n={nv}")
                    penv[p] = val
                if doms and not any(domain_satisfiable(d, penv) for d in doms):
                    return Fraction(0)
                return pw({p: penv.get(p, Fraction(nv)) for p in pw.params})
            return f

        wc = fit_polynomial(gated(w), max_degree=8, base=base, step=base)
        if wc is not None and all(c == 0 for c in wc):
            continue
        vc = fit_polynomial(gated(v), max_degree=8, base=base, step=base)
        if vc is None or wc is None:
            raise QPError(f"non-polynomial entry: {e['value_plain'][:60]}")
        out.append((vc, wc))
    return out


def cluster_levels(entries):
    """Group entries by (degree, leading coefficient) of the value."""
    groups = {}
    for vc, wc in entries:
        deg = len(vc) - 1
        key = (deg, vc[-1])
        groups.setdefault(key, []).append((vc, wc))
    levels = []
    for (deg, lc), members in groups.items():
        mass = [Fraction(0)] * (1 + max(len(w) for _, w in members))
        for _, wc in members:
            for i, c in enumerate(wc):
                mass[i] += c
        while len(mass) > 1 and mass[-1] == 0:
            mass.pop()
        # exact mass-weighted average rd, as a sympy rational function
        num = sum(to_sympy(vc) * to_sympy(wc) for vc, wc in members)
        den = sum(to_sympy(wc) for _, wc in members)
        avg = sp.cancel(num / den) if den.subs(n, NREF) != 0 else None
        vmax = max((to_sympy(vc) for vc, _ in members),
                   key=lambda e: e.subs(n, NREF))
        if avg is None:
            avg = vmax
        levels.append({
            "deg": deg, "lc": lc, "members": members,
            "mass": mass, "avg": avg, "vmax": vmax,
        })
    levels.sort(key=lambda L: L["vmax"].subs(n, NREF))
    return levels


def series_str(expr, order=2):
    """Plain asymptotic string of a sympy rational function, e.g. 1/3 - 1/(3n)."""
    expr = sp.cancel(sp.together(expr))
    try:
        s = sp.series(expr, n, sp.oo, order).removeO()
    except Exception:
        return str(expr)
    s = sp.nsimplify(sp.expand(s))
    return str(s).replace("**", "^")


def analyze(kernel, model):
    path = os.path.join(HERE, "data", f"{kernel}.{model}.json")
    d = json.load(open(path))
    binding = BINDINGS.get(kernel.split(".")[0], {})
    # keep every bound parameter on a fixed residue class (moduli up to 8):
    # scale the sampling base by each binding denominator
    base = BASE
    for m in binding.values():
        base *= m.denominator
    total_pw = Piecewise(d["total_accesses_plain"])
    total_c = fit(total_pw, binding, base)
    total = to_sympy(total_c)

    dslp = os.path.join(HERE, "dsl", kernel.split(".")[0] + ".dsl")
    header = open(dslp).read().split(";")[0]
    param_names = [p.strip() for p in header.replace("params", "").split(",")]

    rd_levels = cluster_levels(
        load_entries(d["rd_distribution"], binding, base, param_names))
    ri_levels = cluster_levels(
        load_entries(d["ri_distribution"], binding, base, param_names))

    trace_deg = len(total_c) - 1
    covered = sum(to_sympy(L["mass"]) for L in rd_levels)
    ri_cold = sp.expand(total - sum(to_sympy(L["mass"]) for L in ri_levels))
    if model == "inf":
        # infinite repeat leaves no first-touch accesses; whatever mass is
        # not in the distribution was dropped by degenerate-region filtering
        slack = sp.expand(total - covered)
        cold_used = sp.Integer(0)
    else:
        # compulsory portion as the residual of the RD distribution, so the
        # staircase is self-consistent; the analyzer filters degenerate
        # regions from the RI and RD lists independently, so the RI-based
        # cold count can differ by a lower-order sliver (reported below)
        cold_used = sp.expand(total - covered)
        slack = sp.expand(ri_cold - cold_used)

    out = {"kernel": kernel, "model": model,
           "total": poly_str(total_c),
           "trace_deg": trace_deg,
           "cold_portion": series_str(cold_used / total),
           "unaccounted_portion": series_str(slack / total),
           "levels": []}

    tail = sp.Integer(0)  # mass strictly above current level
    for L in rd_levels:
        tail += to_sympy(L["mass"])
    running = tail
    for k, L in enumerate(rd_levels, 1):
        mass = to_sympy(L["mass"])
        running -= mass
        portion = sp.cancel(mass / total)
        plateau = sp.cancel((running + cold_used) / total)
        out["levels"].append({
            "k": k,
            "rd_scale": f"{L['lc']} n^{L['deg']}" if L["deg"] else str(L["lc"]),
            "rd_avg": series_str(L["avg"], order=3),
            "rd_max": str(sp.expand(L["vmax"])).replace("**", "^"),
            "n_entries": len(L["members"]),
            "portion": series_str(portion),
            "miss_after": series_str(plateau),
            # exact pieces for numeric evaluation elsewhere
            "mass_coeffs": [str(c) for c in L["mass"]],
            "members": [{"v": [str(c) for c in vc], "w": [str(c) for c in wc]}
                        for vc, wc in L["members"]],
        })

    out["ri_levels"] = []
    for k, L in enumerate(ri_levels, 1):
        portion = sp.cancel(to_sympy(L["mass"]) / total)
        out["ri_levels"].append({
            "k": k,
            "ri_scale": f"{L['lc']} n^{L['deg']}" if L["deg"] else str(L["lc"]),
            "portion": series_str(portion),
            "spans_trace": L["deg"] >= trace_deg,
            "mass_coeffs": [str(c) for c in L["mass"]],
        })
    out["total_coeffs"] = [str(c) for c in total_c]
    return out


def main():
    os.makedirs(os.path.join(HERE, "regimes"), exist_ok=True)
    kernels = sys.argv[1:]
    if not kernels:
        kernels = sorted({f[:-len(".single.json")] if f.endswith(".single.json")
                          else f[:-len(".inf.json")]
                          for f in os.listdir(f"{HERE}/data")
                          if f.endswith((".single.json", ".inf.json"))})
    for k in kernels:
        for model in ("single", "inf"):
            path = os.path.join(HERE, "data", f"{k}.{model}.json")
            if not os.path.exists(path):
                continue
            try:
                res = analyze(k, model)
            except (QPError, Exception) as e:  # noqa: BLE001 - record and move on
                res = {"kernel": k, "model": model, "error": f"{type(e).__name__}: {e}"}
            with open(os.path.join(HERE, "regimes", f"{k}.{model}.json"), "w") as f:
                json.dump(res, f, indent=1)
            tag = "ok" if "error" not in res else "ERR " + res["error"][:80]
            print(f"{k}.{model}: {tag}", flush=True)


if __name__ == "__main__":
    main()
