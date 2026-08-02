#!/usr/bin/env python3
"""Principal contribution terms of every kernel, from the RD distribution.

Model. The analyzer bins every warm access (a reuse) by its reuse distance
(RD, distinct cache lines in the reuse window). A bin with distance V(N) and
population M(N) is a *principal contribution term*:

  * misses:  the bin's M(N) accesses miss exactly when the cache holds fewer
             than V(N) lines, so the term contributes M(N) misses on the cache
             range C < V(N) and none above — (boundary, portion) = (V, M).
  * DMD:     the bin contributes M(N) * sqrt(V(N)), order deg(M) + deg(V)/2.

Bins whose value depends on a loop iterator are split on the iterator's
residue class mod the line size: on each class the value is either a constant
level (stencil boundary structure) or a ramp family (triangular kernels),
which we keep as a family term with its exact mass and distance range.

All polynomial identification is exact: values on the residue class
N = base + k*h are interpolated with rational arithmetic (no fitting).
Requires results/*.json from the fixed, exact-counting analyzer.

Output: terms_table.json + a printed digest per kernel.
"""
import glob
import json
import math
import os
import sys
from fractions import Fraction

from qpeval import (Piecewise, compile_expr, call, detect_modulus, free_names,
                    is_dyadic)
from region_enum import enum_region
from schedule_map import parse_accesses, sources_to_text

HERE = os.path.dirname(os.path.abspath(__file__))

BINDINGS = {  # parameter overrides; every other parameter is bound to N
    "sym_convolution": {"p1": 9, "A_d0": 9, "A_d1": 9},
}

MAX_DEG = 6


# ---------------------------------------------------------------- exact fits

def interp_poly(samples):
    """Exact polynomial through (x, y) samples: coefficient list c0..cd
    (monomial, ascending), via an exact-rational Vandermonde solve (n <= 8)."""
    xs = [Fraction(x) for x, _ in samples]
    ys = [Fraction(y) for _, y in samples]
    n = len(xs)
    A = [[x ** j for j in range(n)] for x in xs]
    b = ys[:]
    # Gaussian elimination
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        inv = Fraction(1) / A[col][col]
        A[col] = [a * inv for a in A[col]]
        b[col] *= inv
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * c for a, c in zip(A[r], A[col])]
                b[r] -= f * b[col]
    poly = b
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def fit_class_poly(f, base, h, max_deg=MAX_DEG):
    """Exact polynomial of N -> f(N) on the class N = base + k*h.
    Returns (coeffs ascending, exact: bool)."""
    pts = [(base + k * h, f(base + k * h)) for k in range(max_deg + 2)]
    poly = interp_poly(pts[: max_deg + 1])
    # verify on the held-out point
    x, y = pts[-1]
    val = sum(c * x ** j for j, c in enumerate(poly))
    return poly, val == y


def poly_str(poly, var="n"):
    if not any(c != 0 for c in poly):
        return "0"
    parts = []
    for j in range(len(poly) - 1, -1, -1):
        c = poly[j]
        if c == 0:
            continue
        cs = str(c) if c.denominator == 1 else f"({c.numerator}/{c.denominator})"
        if j == 0:
            parts.append(cs)
        elif j == 1:
            parts.append(f"{cs}·{var}" if abs(c) != 1 else
                         (f"-{var}" if c < 0 else var))
        else:
            parts.append(f"{cs}·{var}^{j}" if abs(c) != 1 else
                         (f"-{var}^{j}" if c < 0 else f"{var}^{j}"))
    out = " + ".join(parts).replace("+ -", "- ")
    return out


def poly_deg_lead(poly):
    for j in range(len(poly) - 1, -1, -1):
        if poly[j] != 0:
            return j, poly[j]
    return -1, Fraction(0)


def snap_order(order, tol=0.08):
    """Snap a numerically estimated growth order to the nearest quarter when
    within tol (the suite's true orders are multiples of 1/4); otherwise keep
    the raw estimate."""
    q = round(order * 4) / 4
    return q if abs(q - order) <= tol else order


# ------------------------------------------------------------- bin analysis

class KernelTerms:
    def __init__(self, name, rec, dsl_text):
        self.name = name
        self.rec = rec
        self.method = rec.get("method", "?")
        self.accesses = parse_accesses(dsl_text)
        params_line = next(l for l in dsl_text.splitlines()
                           if l.strip().startswith("params"))
        self.params = {p.strip().rstrip(";") for p in
                       params_line.split(None, 1)[1].split(",")}
        self.binds = BINDINGS.get(name, {})
        all_text = " ".join(e["value_plain"] + " " + e["mass_plain"] +
                            " ".join(r["domain_plain"] + r["count_plain"]
                                     for r in e["regions"])
                            for e in rec["rd"])
        self.h = detect_modulus(all_text)
        self.base = max(32 * self.h, 256)
        self.base -= self.base % self.h

    def env(self, n, extra=None):
        e = {p: Fraction(self.binds.get(p, n)) for p in self.params}
        if extra:
            e.update(extra)
        return e

    def total_poly(self):
        pw = Piecewise(self.rec["total"])
        f = lambda n: pw(self.env(n))
        return fit_class_poly(f, self.base, self.h)

    # -- level terms (parameter-only value)
    def level_term(self, e, idx):
        vpw = Piecewise(e["value_plain"])
        mpw = Piecewise(e["mass_plain"])
        # anchor on the class where the bin is live: try shifting the base
        # through the residue classes until the mass is nonzero
        for shift in range(0, self.h, 1):
            base = self.base + shift
            mvals = [mpw(self.env(base + k * self.h)) for k in range(3)]
            if any(v != 0 for v in mvals):
                break
        else:
            return []  # dead bin on every sampled class
        mf = lambda n: mpw(self.env(n))
        vf = lambda n: vpw(self.env(n))
        mpoly, mexact = fit_class_poly(mf, base, self.h)
        vpoly, vexact = fit_class_poly(vf, base, self.h)
        mdeg, mlead = poly_deg_lead(mpoly)
        vdeg, vlead = poly_deg_lead(vpoly)
        if mdeg < 0 or (mlead < 0):
            return []
        return [dict(
            bin=idx, kind="level",
            n_class=(base % self.h, self.h),
            sources=sources_to_text(e.get("sources", []), self.accesses),
            value_expr=e["value_plain"], value_poly=poly_str(vpoly),
            value_deg=vdeg, value_lead=float(vlead),
            mass_expr=e["mass_plain"], mass_poly=poly_str(mpoly),
            mass_deg=mdeg, mass_lead=float(mlead),
            mass_lead_frac=str(Fraction(mlead)),
            dmd_order=mdeg + vdeg / 2.0,
            dmd_coeff=float(mlead) * math.sqrt(max(float(vlead), 0.0)),
            exact_fit=bool(mexact and vexact),
        )]

    # -- iterator-involved bins: enumerate over the exposed iterator(s)
    def iter_terms(self, e, idx):
        iters = sorted((free_names(e["value_plain"])
                        | set().union(*(free_names(r["domain_plain"])
                                        | free_names(r["count_plain"])
                                        for r in e["regions"])))
                       - self.params) if e["regions"] else \
            sorted(free_names(e["value_plain"]) - self.params)
        if not iters or len(iters) > 3:
            return None
        texts = [e["value_plain"]] + [r["domain_plain"] + r["count_plain"]
                                      for r in e["regions"]]
        fast = all(is_dyadic(t) for t in texts)
        mode = "fast" if fast else "exact"
        conv = float if fast else Fraction
        vfn = compile_expr(e["value_plain"], "val", mode)

        def sweep(n):
            """[(value, count, iters)] at parameter value n, enumerating only
            region-domain points. With the fast (float) path all quantities
            are dyadic, hence still exact."""
            env = {k: conv(v) for k, v in self.env(n).items()}
            lim = int(2 * n + 4 * self.h)
            out = []
            for r in e["regions"]:
                for pt, c in enum_region(r["domain_plain"], r["count_plain"],
                                         iters, env, lim, mode, conv):
                    out.append((call(vfn, pt), c,
                                tuple(int(pt[v]) for v in iters)))
            return out

        # bucket by iterator residue class; decide level vs family per class
        samples = [self.base + k * self.h for k in range(MAX_DEG + 2)]
        per_n = {n: sweep(n) for n in samples}
        # split into (residue tuple) classes
        classes = {}
        for n in samples:
            for value, count, ivs in per_n[n]:
                r = tuple(iv % self.h for iv in ivs)
                classes.setdefault(r, {}).setdefault(n, []).append(
                    (value, count, ivs))
        terms = []
        # classes with identical constant value at every n merge into levels
        level_groups = {}
        families = {}
        for r, by_n in sorted(classes.items()):
            values_by_n = {n: {v for v, _, _ in by_n.get(n, [])}
                           for n in samples}
            if all(len(vs) <= 1 for vs in values_by_n.values()):
                vf_map = {n: Fraction(next(iter(values_by_n[n]), 0))
                          for n in samples}
                mf_map = {n: Fraction(sum(c for _, c, _ in by_n.get(n, [])))
                          for n in samples}
                level_groups.setdefault(
                    tuple(vf_map[n] for n in samples), []
                ).append((r, mf_map))
            else:
                families[r] = by_n
        for vkey, group in sorted(level_groups.items()):
            mass_by_n = {n: sum(mf[n] for _, mf in group) for n in samples}
            if all(v == 0 for v in mass_by_n.values()):
                continue
            vpoly = interp_poly(list(zip(samples, vkey))[: MAX_DEG + 1])
            mpoly = interp_poly(list(mass_by_n.items())[: MAX_DEG + 1])
            mx, my = samples[-1], mass_by_n[samples[-1]]
            mexact = sum(c * mx ** j for j, c in enumerate(mpoly)) == my
            vx = sum(c * mx ** j for j, c in enumerate(vpoly))
            vexact = vx == vkey[-1]
            mdeg, mlead = poly_deg_lead(mpoly)
            vdeg, vlead = poly_deg_lead(vpoly)
            if mlead < 0:
                continue
            terms.append(dict(
                bin=idx, kind="level",
                iter_classes=[r for r, _ in group], modulus=self.h,
                sources=sources_to_text(e.get("sources", []), self.accesses),
                value_expr=e["value_plain"], value_poly=poly_str(vpoly),
                value_deg=vdeg, value_lead=float(vlead),
                mass_expr="(enumerated over iterator classes)",
                mass_poly=poly_str(mpoly),
                mass_deg=mdeg, mass_lead=float(mlead),
                mass_lead_frac=str(Fraction(mlead)),
                dmd_order=mdeg + vdeg / 2.0,
                dmd_coeff=float(mlead) * math.sqrt(max(float(vlead), 0.0)),
                exact_fit=bool(mexact and vexact),
            ))
        if families:
            # one family term for the whole bin remainder: exact mass, exact
            # distance range, numeric DMD order from the enumerated sums
            mass_by_n, dmd_by_n, vmax_by_n, vmin_by_n = {}, {}, {}, {}
            for n in samples:
                tot = Fraction(0)
                dd = 0.0
                vmax, vmin = Fraction(0), None
                for r, by_n in families.items():
                    for value, count, _ in by_n.get(n, []):
                        tot += Fraction(count)
                        if value > 0:
                            dd += float(count) * math.sqrt(float(value))
                        value = Fraction(value)
                        vmax = max(vmax, value)
                        vmin = value if vmin is None else min(vmin, value)
                mass_by_n[n], dmd_by_n[n] = tot, dd
                vmax_by_n[n] = vmax
                vmin_by_n[n] = vmin if vmin is not None else Fraction(0)
            mpoly = interp_poly(list(mass_by_n.items())[: MAX_DEG + 1])
            vmaxpoly = interp_poly(list(vmax_by_n.items())[: MAX_DEG + 1])
            mdeg, mlead = poly_deg_lead(mpoly)
            vdeg, vlead = poly_deg_lead(vmaxpoly)
            n1, n2 = samples[-2], samples[-1]

            def held_out_ok(poly, by_n):
                x, y = n2, by_n[n2]
                return sum(c * x ** j for j, c in enumerate(poly)) == y

            def slope_deg(by_n):
                a, b = float(by_n[n1]), float(by_n[n2])
                if a <= 0 or b <= 0:
                    return 0, 0.0
                d = round(2 * math.log(b / a) / math.log(n2 / n1)) / 2
                return d, b / (n2 ** d)

            mexact = held_out_ok(mpoly, mass_by_n)
            vexact = held_out_ok(vmaxpoly, vmax_by_n)
            # an interpolation through non-polynomial samples fabricates a
            # high-degree fit; fall back to a log-slope degree (nearest half)
            if not mexact:
                mdeg, mlead = slope_deg(mass_by_n)
                mpoly = None
            if not vexact:
                vdeg, vlead = slope_deg(vmax_by_n)
                vmaxpoly = None
            raw_order = (math.log(dmd_by_n[n2] / dmd_by_n[n1])
                         / math.log(n2 / n1)) if dmd_by_n[n1] > 0 else 0.0
            # The order comes from the exact (held-out-verified) degrees:
            # a ramp family with mass ~ N^md and top distance ~ N^vd
            # contributes at order md + vd/2. The finite-size slope is kept
            # for reference; at the anchor sizes it can sit well above the
            # asymptotic order (lower-order mass terms still cancelling).
            order = mdeg + vdeg / 2.0
            coeff = dmd_by_n[n2] / (n2 ** order) if order else dmd_by_n[n2]
            note = None
            if abs(raw_order - order) > 0.25:
                note = (f"finite-size slope {raw_order:.2f} at "
                        f"n≈{samples[0]}–{n2}; converges to n^{order:g}")
            vmax_str = poly_str(vmaxpoly) if vmaxpoly is not None else \
                f"~{vlead:.4g}·n^{vdeg:g} (slope estimate)"
            mass_str = poly_str(mpoly) if mpoly is not None else \
                f"~{mlead:.4g}·n^{mdeg:g} (slope estimate)"
            terms.append(dict(
                bin=idx, kind="family", modulus=self.h,
                sources=sources_to_text(e.get("sources", []), self.accesses),
                value_expr=e["value_plain"],
                value_range=[poly_str(interp_poly(
                    list(vmin_by_n.items())[: MAX_DEG + 1])),
                    vmax_str],
                value_deg=vdeg, value_lead=float(vlead),
                mass_expr=e["mass_plain"], mass_poly=mass_str,
                mass_deg=mdeg, mass_lead=float(mlead),
                mass_lead_frac=str(Fraction(mlead)),
                dmd_order=order, dmd_order_measured=round(raw_order, 3),
                dmd_coeff=coeff, note=note,
                exact_fit=bool(mexact and vexact),
            ))
        return terms

    def conservation(self):
        """summass/warm and (warm+compulsory)/total at two anchor sizes."""
        warm = Piecewise(self.rec["warm"])
        comp = Piecewise(self.rec["compulsory"])
        total = Piecewise(self.rec["total"])
        masses = [Piecewise(e["mass_plain"]) for e in self.rec["rd"]]
        out = []
        for n in (self.base, self.base + self.h):
            env = self.env(n)
            w, c, t = warm(env), comp(env), total(env)
            s = sum((m(env) for m in masses), Fraction(0))
            out.append(dict(n=n, ratio=float(s / w) if w else None,
                            closes=bool(w + c == t)))
        return out

    def extract(self):
        total_poly, total_exact = self.total_poly()
        a_deg, a_lead = poly_deg_lead(total_poly)
        terms = []
        skipped = []
        for idx, e in enumerate(self.rec["rd"]):
            iters = free_names(e["value_plain"]) - self.params
            if not iters:
                terms.extend(self.level_term(e, idx))
            else:
                t = self.iter_terms(e, idx)
                if t is None:
                    skipped.append(idx)
                else:
                    terms.extend(t)
        # miss-contribution view: boundary in lines/bytes, portion of accesses
        for t in terms:
            t["miss_boundary_lines"] = t.get("value_poly") or \
                (t.get("value_range") or ["?", "?"])[1]
            t["miss_boundary_bytes_at"] = f"64·({t['miss_boundary_lines']})"
            if a_lead > 0:
                t["portion_deg"] = t["mass_deg"] - a_deg
                t["portion_lead"] = float(Fraction(t["mass_lead_frac"]) /
                                          a_lead)
        terms.sort(key=lambda t: (-t["dmd_order"], -t["dmd_coeff"]))
        spectrum = {}
        for t in terms:
            key = snap_order(t["dmd_order"])
            spectrum[key] = spectrum.get(key, 0.0) + t["dmd_coeff"]
        spec = sorted(spectrum.items(), key=lambda kv: -kv[0])
        return dict(
            kernel=self.name, method=self.method,
            modulus=self.h, anchor_base=self.base,
            access_poly=poly_str(total_poly), access_deg=a_deg,
            access_lead=float(a_lead), access_exact=total_exact,
            conservation=self.conservation(),
            n_bins=len(self.rec["rd"]), skipped_bins=skipped,
            terms=terms,
            spectrum=[[o, c] for o, c in spec],
            dmd_order=spec[0][0] if spec else None,
            headroom=(spec[0][0] - a_deg) if spec else None,
        )


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    out = {}
    if os.path.exists(f"{HERE}/terms_table.json"):
        out = json.load(open(f"{HERE}/terms_table.json"))
    for path in sorted(glob.glob(f"{HERE}/results/sym_*.json")):
        name = os.path.basename(path)[:-5]
        if only and name not in only:
            continue
        d = json.load(open(path))
        if d.get("status") != "ok":
            continue
        dsl = open(f"{HERE}/dsl/{name}.dsl").read()
        for model in ("single", "inf"):
            rec = d.get(model)
            if not rec:
                continue
            try:
                res = KernelTerms(name, rec, dsl).extract()
            except Exception as exc:
                print(f"{name}.{model}: FAILED {exc!r}")
                continue
            out[f"{name}.{model}"] = res
            top = ", ".join(f"{c:.3g}·N^{o:g}" for o, c in res["spectrum"][:3])
            print(f"{name:22s} {model:6s} [{res['method']:5s}] a={res['access_deg']} "
                  f"d={res['dmd_order']:.2f} head={res['headroom']:+.2f}  {top}")
    json.dump(out, open(f"{HERE}/terms_table.json", "w"), indent=1)
    print(f"\nwrote terms_table.json ({len(out)} kernel-models)")


if __name__ == "__main__":
    main()
