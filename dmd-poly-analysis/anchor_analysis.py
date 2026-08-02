#!/usr/bin/env python3
"""Anchoring the leading DMD term for triangular (class-C) kernels.

Why the old route fails here. The analyzer's RD bins for triangular kernels
carry *per-iteration-point* counts (the bin value forced loop iterators into
parameter space), so evaluating a count at a single point -- what
analyze_math.py does -- misses that the true bin population is a SUM over the
exposed iterators. On top of that, triangular populations are quasi-polynomials
like N^3/6 - N^2/2 + ... whose log-log slope converges slowly, so neither the
exponent nor the coefficient anchors.

The fix has two parts:

1.  dmd-core now emits `mass_plain` per RD bin: the exact, parameter-only bin
    population (cardinality of the un-projected piece domain, computed by isl).

2.  This script anchors each bin EXACTLY: on a residue class of N (steps of 8,
    which absorbs the mod-8 cache-line pieces) a mass is an exact polynomial,
    so finite differences over rational arithmetic give its true degree and
    leading coefficient -- no slope fitting, no rounding.

From the anchored bins we build the kernel's DMD *term spectrum*
    term = mass * sqrt(value),   order = deg(mass) + deg(value)/2,
report every order with its summed coefficient, the asymptotically dominant
term, the runner-up, and the CROSSOVER size where the dominant term actually
overtakes (for triangular kernels this can sit far beyond practical sizes,
which is exactly why single-number "order" is a poor characterization).

We also report the local exponent p(N) = log2(DMD(2N)/DMD(N)) and its
CURVATURE q(N) = p(2N) - p(N): a pure power law is a straight line in log-log
coordinates (q = 0), so q is the model-independent diagnostic for "has the
leading term anchored yet" -- q > 0 means a faster term is still mixing in.

Requires results/*.json produced by the mass-emitting dmd-cli.
"""
import json, math, os, re, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))

CLASS_C = ["sym_syrk", "sym_syr2k", "sym_trmm", "sym_symm",
           "sym_cholesky", "sym_lu", "sym_trisolve", "sym_lu_decomp"]
ANCHORS = ["sym_gemm", "sym_mvt"]

PARAM_NAMES = {"p0", "p1", "p2", "p3"}
# tokens that look like identifiers but are operators/keywords, not variables:
# floor()/mod() etc. and the boolean connectives inside piecewise guards.
FUNCS = {"floor", "sqrt", "min", "max", "mod", "and", "or", "not"}


def free_names(expr):
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)) - FUNCS


def param_only(expr):
    return free_names(expr) <= PARAM_NAMES


# ---------------------------------------------------- exact rational evaluation

def _rationalize(expr):
    """Turn every integer literal into a Fraction so eval is exact; map the DSL
    `mod` operator (in domain guards) to Python `%`."""
    expr = expr.replace("^", "**").replace(" mod ", " % ")
    return re.sub(r"(?<![\w.])(\d+)", r"F(\1)", expr)


def exact_eval(expr, N):
    """Exact rational value of a parameter-only piecewise formula at all
    params = N.  Handles the `guard [dom] => val` piecewise tail syntax and
    floor()."""
    env = {"F": Fraction, "floor": lambda x: Fraction(math.floor(x)),
           "min": min, "max": max}
    for nm in free_names(expr):
        env[nm] = Fraction(N)

    def ev(piece):
        piece = piece.strip().rstrip("+-").strip()
        if not piece:
            return Fraction(0)
        return eval(_rationalize(piece), {"__builtins__": {}}, env)

    i = expr.find("[")
    total = ev(expr if i < 0 else expr[:i])
    if i >= 0:
        for seg in expr[i + 1:].split("["):
            dom, _, val = seg.partition("] =>")
            dom = re.sub(r"(?<![<>=!])=(?!=)", "==", dom.replace("^", "**"))
            if eval(_rationalize(dom), {"__builtins__": {}}, env):
                total += ev(val)
    return total


# --------------------------------------- exact degree + leading coefficient

def anchor_poly(expr, base=2048, h=8, max_deg=8):
    """Exact degree and leading coefficient of a parameter-only
    quasi-polynomial, via finite differences on the residue class
    N = base + k*h (h a multiple of the mod-8 period, base large enough that
    all size guards are settled).  Returns (degree, leading_coeff: float),
    both EXACT (no fitting).  degree -1 means identically zero on the class."""
    vals = [exact_eval(expr, base + k * h) for k in range(max_deg + 2)]
    diffs = [vals]
    while len(diffs[-1]) > 1:
        prev = diffs[-1]
        diffs.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    deg = -1
    for d in range(len(diffs) - 1, -1, -1):
        if any(x != 0 for x in diffs[d]):
            deg = d
            break
    if deg < 0:
        return -1, 0.0
    lead = diffs[deg][0] / (math.factorial(deg) * Fraction(h) ** deg)
    return deg, float(lead)


# ------------------------------------------------------------- term spectrum

def spectrum(rec):
    """Anchored DMD term spectrum of one analysis record.
    Returns (terms, n_skipped) where each term is
    dict(order, coeff, mass_deg, mass_coeff, val_deg, val_coeff)."""
    terms, skipped = [], 0
    for e in rec["rd"]:
        mass, val = e.get("mass_plain", ""), e["value_plain"]
        if not (param_only(mass) and param_only(val)):
            skipped += 1
            continue
        md, mc = anchor_poly(mass)
        if md < 0 or mc <= 0:
            continue
        vd, vc = anchor_poly(val)
        if vd < 0 or vc < 0:
            continue  # negative-value artifact bins carry no movement
        terms.append(dict(order=md + vd / 2.0, coeff=mc * math.sqrt(vc),
                          mass_deg=md, mass_coeff=mc, val_deg=vd, val_coeff=vc))
    return terms, skipped


def group_orders(terms):
    """Aggregate coefficients of terms sharing the same order, descending."""
    agg = {}
    for t in terms:
        agg[t["order"]] = agg.get(t["order"], 0.0) + t["coeff"]
    return sorted(agg.items(), key=lambda kv: -kv[0])


def crossover(o1, c1, o2, c2):
    """N where c1*N^o1 (faster, smaller-coeff) overtakes c2*N^o2."""
    if o1 <= o2 or c1 <= 0 or c2 <= 0:
        return None
    return (c2 / c1) ** (1.0 / (o1 - o2))


# ---------------------------------------- model DMD(N), local exponent, curvature

def model_dmd(rec, N):
    """Exact DMD(N) under the scale model: sum of mass*sqrt(value) over bins
    (parameter-only bins; per-point bins are excluded and reported)."""
    tot = 0.0
    for e in rec["rd"]:
        mass, val = e.get("mass_plain", ""), e["value_plain"]
        if not (param_only(mass) and param_only(val)):
            continue
        m, v = float(exact_eval(mass, N)), float(exact_eval(val, N))
        if m > 0 and v > 0:
            tot += m * math.sqrt(v)
    return tot


def slope_curvature(f, sweep):
    """[(N, p(N), q(N))] with p = log2(f(2N)/f(N)), q = p(2N) - p(N)."""
    val = {N: f(N) for N in sweep}
    out = []
    for N in sweep:
        if 2 * N in val and val[N] > 0 and val[2 * N] > 0:
            p = math.log2(val[2 * N] / val[N])
            q = None
            if 4 * N in val and val[4 * N] > 0:
                q = math.log2(val[4 * N] / val[2 * N]) - p
            out.append((N, p, q))
    return out


# ---------------------------------------------------------------- reporting

def fmt_q(q):
    return f"{q:+7.3f}" if q is not None else "      -"


def analyze_kernel(name, sweep=(64, 128, 256, 512, 1024, 2048, 4096, 8192)):
    rec = json.load(open(f"{HERE}/results/{name}.json"))["single"]
    short = name.replace("sym_", "")
    a_deg, a_coeff = anchor_poly(rec["total"])
    terms, skipped = spectrum(rec)
    orders = group_orders(terms)
    print(f"\n===== {short}  (accesses: exactly {a_coeff:g} * N^{a_deg} + lower) =====")
    if skipped:
        print(f"  [{skipped} bin(s) skipped: value or mass not parameter-only]")

    print("  anchored DMD term spectrum (exact degrees & coefficients):")
    for o, c in orders[:5]:
        head = o - a_deg
        print(f"    ~ {c:10.4g} * N^{o:<4.3g}   (headroom {head:+.1f})")
    if len(orders) > 1:
        (o1, c1), (o2, c2) = orders[0], orders[1]
        nx = crossover(o1, c1, o2, c2)
        if nx and o1 > o2:
            print(f"  dominant N^{o1:g} term overtakes the N^{o2:g} term only at "
                  f"N ~ {nx:,.0f}")

    print("  model DMD(N): local exponent p(N) and curvature q(N) = p(2N)-p(N):")
    rows = slope_curvature(lambda N: model_dmd(rec, N), sweep)
    print("      N    p(N)    q(N)")
    for N, p, q in rows:
        print(f"  {N:5d}  {p:6.3f} {fmt_q(q)}")

    # ground truth from the exact-trace simulator, if present
    sim_path = f"{HERE}/exact/{short}.json"
    ratio_rising = None
    if os.path.exists(sim_path):
        sim = {r["N"]: r for r in json.load(open(sim_path))}
        if len(sim) >= 3:
            print("  EXACT trace (no approximation) vs the scale model:")
            print("    p,q = local exponent & curvature of the exact DMD; "
                  "ratio = exact / scale-model DMD;")
            print("    maxRD = largest reuse distance (lines): exact vs what the "
                  "scale model resolves.")
            print("      N    p(N)    q(N)   exact/scale   maxRD(exact/scale)")
            svals = sorted(sim)
            ratios, maxrd_gaps = [], []
            for N in svals[:-1]:
                if 2 * N not in sim:
                    continue
                p = math.log2(sim[2 * N]["dmd"] / sim[N]["dmd"])
                q = (math.log2(sim[4 * N]["dmd"] / sim[2 * N]["dmd"]) - p
                     if 4 * N in sim else None)
                sy = model_dmd(rec, N)
                r = sim[N]["dmd"] / sy if sy > 0 else float("nan")
                ratios.append(r)
                sy_max = scale_max_rd(rec, N)
                if sy_max > 0:
                    maxrd_gaps.append(sim[N]["max_rd"] / sy_max)
                print(f"  {N:5d}  {p:6.3f} {fmt_q(q)}     {r:6.2f}      "
                      f"{sim[N]['max_rd']:6d} / {sy_max:5.0f}")
            # Robust discriminator: does exact DMD pull away from the scale
            # model across the whole sweep? A ratio that keeps growing (>~2x
            # over the sweep) means the scale model loses a growing share of the
            # movement -> it under-resolves reuse and its headroom reading is a
            # spurious under-estimate. A ratio that settles near a constant means
            # the scale model is faithful and its headroom is trustworthy.
            grow = ratios[-1] / ratios[0] if len(ratios) >= 2 and ratios[0] > 0 else 1.0
            gap = max(maxrd_gaps) if maxrd_gaps else 1.0
            # Two independent signatures of the scale model under-resolving reuse:
            #  (a) exact DMD pulls away across the sweep (ratio growth), and
            #  (b) exact reuse distances far exceed what the scale model resolves.
            # Either firing flags a spurious headroom reading; both fire together
            # for the whole-matrix-reuse triangular kernels.
            ratio_rising = grow > 1.8 or gap > 4.0
            if ratio_rising:
                print(f"    exact/scale ratio grows {ratios[0]:.2f} -> {ratios[-1]:.2f} "
                      f"({grow:.1f}x), and exact reuse distances reach ~{gap:.0f}x what the")
                print(f"    scale model resolves: it UNDER-RESOLVES the whole-matrix reuse, so its")
                print(f"    'headroom {orders[0][0]-a_deg:+.1f}' is a scale-approximation artifact "
                      f"-- real DMD grows faster (positive q).")
            else:
                print(f"    exact/scale ratio settles near {ratios[-1]:.2f} "
                      f"({grow:.1f}x over the sweep) and max reuse distances match "
                      f"({gap:.1f}x): scale headroom trustworthy.")
    return dict(name=short, access_deg=a_deg, access_coeff=a_coeff,
                orders=[(o, c) for o, c in orders], skipped=skipped,
                ratio_rising=ratio_rising)


def scale_max_rd(rec, N):
    """Largest reuse distance (lines) the scale model resolves at this N."""
    mx = 0.0
    for e in rec["rd"]:
        if param_only(e["value_plain"]):
            mx = max(mx, float(exact_eval(e["value_plain"], N)))
    return mx


if __name__ == "__main__":
    names = sys.argv[1:] or (CLASS_C + ANCHORS)
    names = [n if n.startswith("sym_") else f"sym_{n}" for n in names]
    summary = []
    for name in names:
        if not os.path.exists(f"{HERE}/results/{name}.json"):
            print(f"\n===== {name}: no results file, skipping =====")
            continue
        try:
            summary.append(analyze_kernel(name))
        except Exception as exc:
            print(f"\n===== {name}: FAILED ({exc}) =====")
    json.dump(summary, open(f"{HERE}/anchor_table.json", "w"), indent=1)
    print("\nwrote anchor_table.json")
