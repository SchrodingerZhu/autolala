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


def _ev(piece, N):
    piece = piece.replace("^", "**").strip().rstrip("+-").strip()
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
            if eval(dom.replace("^", "**"), {"__builtins__": {}}, _env(dom, N)):
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


def dmd_order_from_rd(rd):
    """DMD order via the exact sum-of-sqrt construction (no formula cancellation):
       DMD = sum_rd  mult(rd) * sqrt(rd_value);  order = max_rd [ ord(mult) + 0.5*ord(value) ].
       Returns (dmd_order, dominant_value_order, dominant_mult_order)."""
    best = None
    for entry in rd:
        vo = order(entry["value_plain"])
        if vo is None or vo < 0:
            vo = 0.0
        mo = None
        for reg in entry.get("regions", []):
            o = order(reg["count_plain"])
            if o is not None and (mo is None or o > mo):
                mo = o
        if mo is None:
            continue
        term = mo + 0.5 * vo
        if best is None or term > best[0]:
            best = (term, vo, mo)
    return best if best else (None, None, None)


rows = []
for path in sorted(glob.glob(f"{HERE}/results/*.json")):
    name = os.path.basename(path)[:-5]
    d = json.load(open(path))
    if d.get("status") != "ok":
        continue
    if not is_symbolic(d["total"]):
        continue  # const kernels -> numeric, no order; skip for the order table
    a = order(d["total"])
    dd, dom_rd_order, dom_mult_order = dmd_order_from_rd(d["rd"])
    gap = (dd - a) if (a is not None and dd is not None) else None
    rows.append(dict(name=name, family=d["family"], n_ri=d["n_ri"], n_rd=d["n_rd"],
                     access_order=a, dmd_order=dd, gap=gap,
                     dom_rd_order=dom_rd_order, dom_mult_order=dom_mult_order,
                     ri_max_order=dist_max_value_order(d["ri"]),
                     rd_max_order=dist_max_value_order(d["rd"]),
                     total=d["total"]))

rows.sort(key=lambda x: (x["gap"] if x["gap"] is not None else -9), reverse=True)
json.dump(rows, open(f"{HERE}/order_table.json", "w"), indent=1)

print(f"{'kernel':20s} {'acc':>4s} {'dmd':>4s} {'gap':>4s} | dominant DMD term: "
      f"{'mult~N^':>7s} {'* sqrt(RD~N^':>12s}) | {'RDmax':>5s} {'RImax':>5s}")
print("-" * 92)
for x in rows:
    f = lambda v: f"{v:4.2f}" if isinstance(v, float) else " -  "
    print(f"{x['name']:20s} {f(x['access_order'])} {f(x['dmd_order'])} {f(x['gap'])} | "
          f"{'':18s}{f(x['dom_mult_order']):>7s} {f(x['dom_rd_order']):>12s}  | "
          f"{f(x['rd_max_order']):>5s} {f(x['ri_max_order']):>5s}")
print(f"\n{len(rows)} symbolic kernels analyzed (DMD order from RD sum-of-sqrt construction)")
