#!/usr/bin/env python3
"""Finite-range ("what does doubling N cost from here?") analysis.

The asymptotic order d predicts DMD(2N)/DMD(N) = 2^d in the limit. Over a real
octave [N, 2N] the accurate quantity is the LOCAL exponent -- the log-log slope
p(N) = dlogF/dlogN, here measured exactly as p(N) = log2(F(2N)/F(N)) from the
symbolic formula. We also compute the local headroom p_DMD(N) - p_acc(N), and the
cache-crossing size N* = sqrt(C/c) from the dominant reuse-distance coefficient,
which is where the smooth model hides a cliff in the real miss curve.

DMD(N) is evaluated cleanly as sum over reuse-distance bins of
multiplicity(N) * sqrt(distance(N)) -- the sum-of-sqrt construction, no formula
cancellation."""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_math import evalf, order, _pyexpr, _env, dmd_dominant, lead_coeff, _round_order

LINE_BYTES = 64            # 64-byte cache line (8 f64)
LLC_BYTES = 2 * 1024 * 1024  # 2 MB last-level, matches the cachegrind simulation


def dmd_value(rec, N):
    """DMD(N) = sum over RD bins of multiplicity(N) * sqrt(distance(N))."""
    total = 0.0
    for e in rec["rd"]:
        v = evalf(e["value_plain"], N)
        if v <= 0:
            continue
        root = math.sqrt(v)
        mult = 0.0
        for r in e["regions"]:
            try:
                if eval(_pyexpr(r["domain_plain"]), {"__builtins__": {}}, _env(r["domain_plain"], N)):
                    mult += evalf(r["count_plain"], N)
            except Exception:
                pass
        total += mult * root
    return total


def dominant_rd_bin(rec):
    """The reuse-distance bin with the fastest-growing distance (sets N*)."""
    best = None
    for e in rec["rd"]:
        o = order(e["value_plain"])
        if o is not None and (best is None or o > best[0]):
            best = (o, e["value_plain"])
    return best  # (growth_order, value_expr) or None


def threshold_N(value_expr, rho):
    """N* where the dominant reuse distance (in lines) fills the LLC:
       distance(N)*LINE_BYTES = LLC_BYTES.  distance ~ c*N^rho (lines)."""
    c = lead_coeff(value_expr, rho)  # leading coeff of distance in *lines*
    if not c or c <= 0 or rho <= 0:
        return None, c
    # c*N^rho * LINE_BYTES = LLC_BYTES
    return (LLC_BYTES / (c * LINE_BYTES)) ** (1.0 / rho), c


def analyze(name, sweep):
    rec = json.load(open(f"results/{name}.json"))["single"]
    d = dmd_dominant(rec["rd"], access_order=order(rec["total"]))["order"]
    print(f"\n===== {name.replace('sym_','')}   (asymptotic DMD order d = {d:.2f}) =====")
    print(f"{'N':>6s} {'DMD(N)':>12s} {'DMD2N/DMDN':>10s} {'local p(N)':>10s} "
          f"{'naive 2^d':>9s} {'acc slope':>9s} {'local head':>10s}")
    for N in sweep:
        F, F2 = dmd_value(rec, N), dmd_value(rec, 2 * N)
        A, A2 = evalf(rec["total"], N), evalf(rec["total"], 2 * N)
        if F <= 0 or F2 <= 0 or A <= 0 or A2 <= 0:
            continue
        ratio = F2 / F
        p = math.log2(ratio)
        ap = math.log2(A2 / A)
        print(f"{N:6d} {F:12.3e} {ratio:10.3f} {p:10.3f} {2**d:9.3f} {ap:9.3f} {p-ap:10.3f}")
    dom = dominant_rd_bin(rec)
    if dom:
        rho = round(dom[0])
        Nstar, c = threshold_N(dom[1], rho)
        if Nstar:
            print(f"  dominant reuse distance ~ {c:.4g} * N^{rho} lines; "
                  f"fills {LLC_BYTES//1024//1024} MB LLC at  N* = {Nstar:.0f}")
        else:
            print(f"  dominant reuse distance growth rho={rho} (bounded -> no cache-crossing size)")


if __name__ == "__main__":
    analyze("sym_gemm", [64, 128, 256, 512, 1024, 2048, 4096])   # Class A, headroom 1
    analyze("sym_mvt",  [64, 128, 256, 512, 1024, 2048, 4096])   # Class B, headroom 1/2
    analyze("sym_syrk", [64, 128, 256, 512, 1024, 2048, 4096])   # Class C, headroom 0
