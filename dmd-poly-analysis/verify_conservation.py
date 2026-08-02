#!/usr/bin/env python3
"""Suite-wide mass-conservation audit of the analyzer output.

For every kernel and model, with all program parameters bound to N:
  * sum(rd bin masses)  must equal  warm          (exactly, when method=exact)
  * warm + compulsory   must equal  total         (by construction; checked)
  * every bin mass must be non-negative
checked on several N spanning all residue classes mod the kernel's modulus.
Kernels analyzed with the scale approximation report their worst deviation
instead (the approximation is not conservation-exact); the table names the
method so downstream tables can gate on it.
"""
import glob
import json
import os
import sys
from fractions import Fraction

from qpeval import Piecewise, detect_modulus, free_names

HERE = os.path.dirname(os.path.abspath(__file__))

# per-kernel parameter overrides (everything else is bound to N)
BINDINGS = {
    # two-scale kernel: filter extent fixed at 9, image scales with N
    "sym_convolution": {"p1": 9, "A_d0": 9, "A_d1": 9},
}


def check_model(name, rec, ns=(512, 515, 1000, 2016)):
    binds = BINDINGS.get(name, {})
    total_pw = Piecewise(rec["total"])
    warm_pw = Piecewise(rec["warm"])
    comp_pw = Piecewise(rec["compulsory"])
    masses = [Piecewise(e["mass_plain"]) for e in rec["rd"]]
    rows = []
    for n in ns:
        env = {}
        for nm in (total_pw.free_names() | warm_pw.free_names()
                   | comp_pw.free_names()
                   | set().union(*(m.free_names() for m in masses))
                   if masses else total_pw.free_names()):
            env[nm] = Fraction(binds.get(nm, n))
        total = total_pw(env)
        warm = warm_pw(env)
        comp = comp_pw(env)
        mass_vals = [m(env) for m in masses]
        summass = sum(mass_vals, Fraction(0))
        neg = sum(1 for v in mass_vals if v < 0)
        rows.append(dict(n=n, total=total, warm=warm, comp=comp,
                         summass=summass, negative_bins=neg))
    return rows


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    report = {}
    fails = []
    for path in sorted(glob.glob(f"{HERE}/results/*.json")):
        name = os.path.basename(path)[:-5]
        if only and name not in only:
            continue
        d = json.load(open(path))
        if d.get("status") != "ok":
            continue
        for model in ("single", "inf"):
            rec = d.get(model)
            if not rec:
                continue
            method = rec.get("method", "?")
            try:
                rows = check_model(name, rec)
            except Exception as exc:
                fails.append((name, model, method, f"EVAL FAIL: {exc}"))
                continue
            worst = 0.0
            bad = False
            for r in rows:
                if r["warm"] != 0:
                    dev = abs(float(r["summass"] / r["warm"]) - 1.0)
                    worst = max(worst, dev)
                ok_sum = (r["summass"] == r["warm"]) if method == "exact" \
                    else abs(float(r["summass"] / r["warm"]) - 1.0) < 0.05 \
                    if r["warm"] != 0 else True
                ok_tot = r["warm"] + r["comp"] == r["total"]
                if not (ok_sum and ok_tot and r["negative_bins"] == 0):
                    bad = True
            report[f"{name}.{model}"] = dict(
                method=method, worst_dev=worst,
                rows=[{k: str(v) for k, v in r.items()} for r in rows])
            status = "FAIL" if bad else "ok"
            if bad:
                fails.append((name, model, method, f"worst dev {worst:.3g}"))
            print(f"{name:24s} {model:6s} [{method:5s}] {status:4s} "
                  f"worst |summass/warm - 1| = {worst:.3g}")
    json.dump(report, open(f"{HERE}/conservation.json", "w"), indent=1)
    print(f"\n{len(fails)} failing kernel-models")
    for f in fails:
        print("  FAIL:", *f)


if __name__ == "__main__":
    main()
