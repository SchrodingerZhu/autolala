#!/usr/bin/env python3
"""Rigorous order analysis of the RI/DMD formulas.

Core identity under the scale model:  DMD = sum_i coeff_i * sqrt(RD_i) * mult_i.
So if access count ~ N^a and DMD ~ N^d, the dominant reuse-distance exponent is
    r = 2*(d - a)
because the dominant DMD term is (N^a accesses) * sqrt(RD ~ N^r). Tiling caps RD at a
constant (tile footprint), driving r -> 0 and DMD order d -> a; the asymptotic DMD
reduction from tiling is therefore ~ N^(d-a). We estimate a and d by log-log slope at
large N (the asymptotic regime where the scale formula is dominated by its leading term).
"""
import json, math, re, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))


def _env(expr, N):
    e = {"sqrt": math.sqrt, "floor": math.floor, "min": min, "max": max}
    for nm in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)):
        if nm not in e:
            e[nm] = float(N)
    return e


def _pyexpr(s):
    """Translate DSL operators to Python: `^`->`**`, and a bare `=` (DSL equality)
    -> `==`, without touching `>=`, `<=`, `==`, `!=`."""
    s = s.replace("^", "**")
    return re.sub(r"(?<![<>=!])=(?!=)", "==", s)


def _ev(piece, N):
    piece = _pyexpr(piece).strip().rstrip("+-").strip()
    if not piece:
        return 0.0
    return float(eval(piece, {"__builtins__": {}}, _env(piece, N)))


def evalf(expr, N):
    """Evaluate a plain formula (with optional piecewise [dom]=>val tail) at all params=N."""
    i = expr.find("[")
    total = _ev(expr if i < 0 else expr[:i], N)
    if i >= 0:
        for seg in expr[i + 1:].split("["):
            dom, _, val = seg.partition("] =>")
            if eval(_pyexpr(dom), {"__builtins__": {}}, _env(dom, N)):
                total += _ev(val, N)
    return total


def order(expr, lo=4000, hi=8000):
    """Asymptotic exponent via log-log slope between two large N (robust to lower-order terms)."""
    try:
        a, b = evalf(expr, lo), evalf(expr, hi)
        if a <= 0 or b <= 0:
            return None
        return math.log(b / a) / math.log(hi / lo)
    except Exception:
        return None


def is_symbolic(expr):
    return bool(re.search(r"[A-Za-z_]", expr.replace("sqrt", "").replace("floor", "")))


def dist_max_value_order(dist):
    """Largest value order across an RI or RD distribution."""
    best = None
    for entry in dist:
        o = order(entry["value_plain"])
        if o is not None and (best is None or o > best):
            best = o
    return best


def lead_coeff(expr, o):
    """Leading coefficient C such that expr ~ C * N^o at large N. Robust for a
    single positive polynomial (no cancellation): C = expr(N) / N^o for large N."""
    if o is None:
        return None
    N = 8192.0
    try:
        v = evalf(expr, N)
    except Exception:
        return None
    return v / (N ** o) if o != 0 else v


def _round_order(o):
    """RI/RD values and access counts are genuine polynomials, so their growth
    exponents are (near) integers; snap to the nearest integer to denoise the
    log-log estimate before extracting a clean leading coefficient."""
    return None if o is None else round(o)


def dmd_dominant(rd, access_order=None):
    """Dominant DMD term via the exact sum-of-sqrt construction (no cancellation):
       DMD = sum over RD entries of  multiplicity(entry) * sqrt(rd_value(entry)).
    Each term grows like N^(m + v/2) with m = growth of the multiplicity and
    v = growth of the reuse-distance value. The dominant growth order is the max
    of (m + v/2); the leading COEFFICIENT is the sum of the leading coefficients
    of every term that attains that order (so DMD ~ coeff * N^order).

    A reuse-bin multiplicity is a count of accesses, so it cannot grow faster than
    the total access count; we clamp m <= access_order to reject spurious
    high-order terms produced by log-log noise on complex piecewise counts.
    Returns dict(order, coeff, rd_growth, count_growth)."""
    m_cap = None if access_order is None else round(access_order)
    terms = []  # (order, coeff, v, m)
    for entry in rd:
        vo = _round_order(order(entry["value_plain"]))
        if vo is None or vo < 0:
            vo = 0
        v_c = lead_coeff(entry["value_plain"], vo) or 0.0
        # dominant region = the one whose multiplicity grows fastest
        best_reg = None
        for reg in entry.get("regions", []):
            mo = _round_order(order(reg["count_plain"]))
            if mo is None:
                continue
            if m_cap is not None and mo > m_cap:
                mo = m_cap  # a bin count cannot exceed the total access count
            if best_reg is None or mo > best_reg[0]:
                best_reg = (mo, lead_coeff(reg["count_plain"], mo) or 0.0)
        if best_reg is None:
            continue
        mo, m_c = best_reg
        term_order = mo + 0.5 * vo
        term_coeff = m_c * math.sqrt(v_c) if v_c > 0 else m_c
        terms.append((term_order, term_coeff, float(vo), float(mo)))
    if not terms:
        return dict(order=None, coeff=None, rd_growth=None, count_growth=None)
    d = max(t[0] for t in terms)
    top = [t for t in terms if abs(t[0] - d) < 1e-6]
    coeff = sum(t[1] for t in top)
    # report the RD/count growth of the single largest-coefficient dominant term
    lead = max(top, key=lambda t: t[1])
    return dict(order=d, coeff=coeff, rd_growth=lead[2], count_growth=lead[3])


def analyze_one_mode(mode):
    """mode is the dict from results/<k>.json for one model (single/inf).
    Returns the growth-rate + leading-coefficient summary, or None if unusable."""
    if mode is None or not is_symbolic(mode["total"]):
        return None
    a = order(mode["total"])
    dom = dmd_dominant(mode["rd"], access_order=a)
    dd = dom["order"]
    gap = (dd - a) if (a is not None and dd is not None) else None
    return dict(access_order=a, dmd_order=dd, dmd_coeff=dom["coeff"], headroom=gap,
                reuse_distance_growth=dom["rd_growth"], count_growth=dom["count_growth"],
                rd_max_order=dist_max_value_order(mode["rd"]),
                ri_max_order=dist_max_value_order(mode["ri"]),
                n_rd=mode["n_rd"], n_ri=mode["n_ri"])


def build_rows():
    rows = []
    for path in sorted(glob.glob(f"{HERE}/results/*.json")):
        name = os.path.basename(path)[:-5]
        d = json.load(open(path))
        if d.get("status") != "ok":
            continue
        # New JSON stores single/inf; fall back to top-level for older files.
        single = d.get("single", d)
        inf = d.get("inf")
        s = analyze_one_mode(single)
        if s is None:
            continue  # const kernels are numeric -> no growth rate
        i = analyze_one_mode(inf)
        rows.append(dict(name=name, family=d.get("family", "?"),
                         single=s, inf=i, total=single["total"]))
    rows.sort(key=lambda x: (x["single"]["headroom"]
                             if x["single"]["headroom"] is not None else -9), reverse=True)
    return rows


def fmt(v):
    return f"{v:5.2f}" if isinstance(v, float) else "  -  "


if __name__ == "__main__":
    rows = build_rows()
    json.dump(rows, open(f"{HERE}/order_table.json", "w"), indent=1)

    print("Growth rate = exponent p such that the quantity ~ N^p at large N.")
    print("DMD ~ coeff * N^order.  'headroom' = DMD order - access order.\n")
    hdr = (f"{'kernel':20s} | {'acc':>5s} {'DMD':>5s} {'coeff':>8s} {'head':>5s} "
           f"{'RDgro':>5s} | {'DMDinf':>6s} {'coefINF':>8s} {'hINF':>5s}")
    print(hdr); print("-" * len(hdr))
    for x in rows:
        s, i = x["single"], x["inf"] or {}
        cf = lambda v: f"{v:8.4f}" if isinstance(v, float) else "    -   "
        print(f"{x['name']:20s} | {fmt(s['access_order'])} {fmt(s['dmd_order'])} "
              f"{cf(s['dmd_coeff'])} {fmt(s['headroom'])} {fmt(s['reuse_distance_growth'])} | "
              f"{fmt(i.get('dmd_order'))} {cf(i.get('dmd_coeff'))} {fmt(i.get('headroom'))}")
    print(f"\n{len(rows)} symbolic kernels. headroom = DMD order - access order = "
          f"half the reuse-distance growth rate.")
    print("Within one order class, rank by coeff (the leading constant factor of DMD).")

    # Where does infinite-repeat change the story?
    print("\nKernels where infinite-repeat raises the DMD growth rate (reveals hidden reuse):")
    any_diff = False
    for x in rows:
        s, i = x["single"], x["inf"]
        if i and s["dmd_order"] and i["dmd_order"] and i["dmd_order"] - s["dmd_order"] > 0.15:
            print(f"  {x['name']:20s} single DMD~N^{s['dmd_order']:.2f} -> "
                  f"inf DMD~N^{i['dmd_order']:.2f}")
            any_diff = True
    if not any_diff:
        print("  (none: infinite-repeat leaves every leading DMD growth rate unchanged)")
