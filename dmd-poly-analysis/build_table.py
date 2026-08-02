#!/usr/bin/env python3
"""Value-backed classification table from the EXACT trace.

For every tractable kernel we run the approximation-free stack-distance simulator
over a size-doubling sweep and measure, directly from the real trace:
  * work order a         -- exponent of the access count (from the symbolic total)
  * data order           -- exponent of the memory footprint (distinct lines touched)
  * rho  = growth exponent of the MAXIMUM reuse distance  (log2 max_rd(2N)/max_rd(N))
  * exact DMD headroom   = p(N) - a, with p(N) = log2 DMD(2N)/DMD(N)
  * curvature q(N)       = p(2N) - p(N)      (>0 => exponent still rising)
The classifier is rho (the real reuse-distance growth): rho~2 -> A (tile),
rho~1 -> B (interchange/fuse), rho~0 -> C (already local)."""
import json, math, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exact"))
import gen_sim
from analyze_math import order

HERE = os.path.dirname(os.path.abspath(__file__))

# sweep sizes by work order, chosen to keep the exact trace ~<2e8 accesses
SWEEP = {4: [24, 32, 48, 64], 3: [48, 64, 96, 128, 192], 2: [128, 256, 512, 1024, 2048]}

# convolution (N-K bounds collapse when all params->N) and heat-3d (too heavy) skip
SKIP = {"convolution", "heat-3d"}


def run(name, sizes):
    exe = gen_sim.build_kernel(name)
    recs = []
    for N in sizes:
        p = subprocess.run([exe, str(N)], capture_output=True, text=True)
        if p.returncode != 0:
            break
        recs.append(json.loads(p.stdout))
    return recs


def foot_order(recs):
    """Footprint (distinct-line) order from cold misses = distinct lines touched."""
    pts = [(r["N"], r["cold"]) for r in recs if r.get("cold", 0) > 0]
    if len(pts) < 2:
        return None
    (n1, c1), (n2, c2) = pts[0], pts[-1]
    return math.log2(c2 / c1) / math.log2(n2 / n1)


def growth(recs, key):
    """Exponent of recs[key] between the last two sizes (asymptotic end)."""
    if len(recs) < 2:
        return None
    a, b = recs[-2], recs[-1]
    if a[key] <= 0 or b[key] <= 0:
        return None
    return math.log2(b[key] / a[key]) / math.log2(b["N"] / a["N"])


def classify(reuse, data, rho):
    """Arithmetic-intensity classifier (matches the direct work-vs-data intuition):
      * genuine reuse (each datum touched many times) over a 2-D+ footprint  -> A
        (tiling makes the reused sub-matrix resident; wins ~N)
      * genuine reuse over a 1-D footprint                                    -> B
        (reuse distance only ~N; interchange/fuse wins ~sqrt(N))
      * touched ~once: no reuse to tile. A growing reuse distance from a
        mis-ordered access -> B (interchange); bounded -> C (already local).
    rho is shown alongside but is not the classifier: a kernel can reach a large
    max reuse distance with only O(1) touches per datum (mvt/gemver read A
    transposed), where the fix is reordering, not tiling."""
    if reuse is None:
        return "?"
    if reuse >= 0.5:
        return "A" if (data is not None and data >= 1.5) else "B"
    if rho is not None and rho >= 0.5:
        return "B"
    return "C"


def main():
    names = sys.argv[1:] or [os.path.basename(p)[4:-4]
                             for p in sorted(__import__("glob").glob(f"{HERE}/dsl/sym_*.dsl"))]
    rows = []
    for name in names:
        if name in SKIP:
            continue
        res = json.load(open(f"{HERE}/results/sym_{name}.json"))
        a = order(res["single"]["total"])
        if a is None:
            continue
        sizes = SWEEP.get(round(a), SWEEP[3])
        try:
            recs = run(name, sizes)
        except Exception as e:
            print(f"{name}: sim failed ({e})", file=sys.stderr)
            continue
        if len(recs) < 3:
            print(f"{name}: too few points", file=sys.stderr)
            continue
        rho = growth(recs, "max_rd")
        # local DMD exponent p between consecutive (possibly non-doubling) sizes,
        # and curvature q = change in that exponent across the last two intervals.
        def p_at(i):
            a0, b0 = recs[i], recs[i + 1]
            if a0["dmd"] <= 0 or b0["dmd"] <= 0:
                return None
            return math.log2(b0["dmd"] / a0["dmd"]) / math.log2(b0["N"] / a0["N"])
        p_last = p_at(len(recs) - 2)
        q_last = (p_at(len(recs) - 2) - p_at(len(recs) - 3)) if len(recs) >= 3 else None
        data = foot_order(recs)
        reuse = (a - data) if data is not None else None
        rows.append(dict(name=name, a=a, data=data, rho=rho,
                         headroom=(p_last - a) if p_last else None,
                         q=q_last, cls=classify(reuse, data, rho),
                         Nmax=recs[-1]["N"]))
    rows.sort(key=lambda r: (-{"A": 2, "B": 1, "C": 0, "?": -1}[r["cls"]], -(r["rho"] or 0)))
    json.dump(rows, open(f"{HERE}/classification.json", "w"), indent=1)

    print(f"{'kernel':16s} {'work':>5s} {'data':>5s} {'reuse×':>6s} "
          f"{'maxRD ρ':>8s} {'DMD head':>8s} {'curv q':>7s} {'class':>5s}  (exact, N≤{'':1s})")
    print("-" * 78)
    cur = None
    for r in rows:
        if r["cls"] != cur:
            cur = r["cls"]
            print(f"--- Class {cur} " + "-" * 60)
        reuse = (r["a"] - r["data"]) if r["data"] is not None else None
        f = lambda v, w=5, p=2: (f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}s}")
        print(f"{r['name']:16s} {f(r['a'])} {f(r['data'])} {f(reuse)} "
              f"{f(r['rho'],8)} {f(r['headroom'],8)} {f(r['q'],7)} {r['cls']:>5s}  (N≤{r['Nmax']})")
    print("\nrho = growth of the largest reuse distance (the classifier): "
          "~2->A tile, ~1->B interchange/fuse, ~0->C local.")
    print("reuse× = work order - data order = how many times each datum is reused "
          "(>=1 => tileable).")
    print("curv q>0 => the DMD exponent is still rising at N<=Nmax (finite-size, "
          "not yet anchored).")


if __name__ == "__main__":
    main()
