#!/usr/bin/env python3
"""Derived quantities over the extracted regime structures.

Reads regimes/<kernel>.<model>.json (symbolic level structure, exact fits)
and data/<kernel>.<model>.json (raw analyzer output), and produces under
tables/:

  suite_regimes.md   per-kernel regime tables (displayed levels only)
  machine_map.md     miss ratio and active regime at real cache sizes
  signatures.md      co-scaling signatures and the suite taxonomy
  dmd_inversion.md   scalar DMD-per-access ranking vs miss ratio ranking
  tiling.md          naive vs tiled matmul staircases and pointwise gain
  summary.json       machine-readable form of all of the above

Two evaluation paths, used deliberately:
  * symbolic statements (boundary scales, portions, plateaus, signatures)
    come from the exact polynomial fits on the sampling residue class;
  * every concrete number at a specific n is evaluated from the raw
    piecewise quasi-polynomials with their guards, so branch selection is
    always correct.  Fraction arithmetic throughout; floats only in display.

Conventions: every program parameter is bound to n (except the per-kernel
bindings mirrored from regimes.BINDINGS, e.g. convolution's filter = n/4).
The PolyBench suite data is block-8: rd values and cache capacities are in
64-byte lines.  The matmul3* variants are block-1 (element granularity).
"""
import json
import math
import os
from fractions import Fraction

from qp import Piecewise, domain_satisfiable
from regimes import BINDINGS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables")


# ---------------------------------------------------------------- polynomials

def pev(coeffs, x):
    v = Fraction(0)
    for c in reversed(coeffs):
        v = v * x + c
    return v


def padd(a, b):
    out = [Fraction(0)] * max(len(a), len(b))
    for i, c in enumerate(a):
        out[i] += c
    for i, c in enumerate(b):
        out[i] += c
    return out


def pneg(a):
    return [-c for c in a]


def ptrim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def plead(a):
    a = ptrim(a)
    return len(a) - 1, a[-1]


def lead_str(a, var="n"):
    d, c = plead(a)
    if d == 0:
        return str(c)
    v = var if d == 1 else f"{var}^{d}"
    return v if c == 1 else f"{c} {v}"


def ratio_lead(num, den):
    """Leading behavior of num/den as (degree difference, coefficient)."""
    dn, cn = plead(num)
    dd, cd = plead(den)
    if cn == 0 or cd == 0:
        return None
    return dn - dd, Fraction(cn, cd)


def ratio_lead_str(num, den, var="n"):
    r = ratio_lead(num, den)
    if r is None:
        return "0"
    d, c = r
    if d == 0:
        return str(c)
    if d < 0:
        den_s = var if d == -1 else f"{var}^{-d}"
        if c.numerator == 1:
            return f"1/({c.denominator} {den_s})" if c.denominator != 1 else f"1/{den_s}"
        return f"{c}/{den_s}" if c.denominator == 1 else \
            f"{c.numerator}/({c.denominator} {den_s})"
    v = var if d == 1 else f"{var}^{d}"
    return f"{c} {v}"


# ---------------------------------------------------------------- kernel model

class Kernel:
    def __init__(self, name, rec, raw):
        self.name = name
        self.model = rec["model"]
        self.binding = BINDINGS.get(name.split(".")[0], {})
        self.total = [Fraction(c) for c in rec["total_coeffs"]]
        self.levels = []
        for L in rec["levels"]:
            members = [([Fraction(c) for c in m["v"]],
                        [Fraction(c) for c in m["w"]]) for m in L["members"]]
            mass = [Fraction(c) for c in L["mass_coeffs"]]
            self.levels.append({
                "k": L["k"], "scale": L["rd_scale"], "avg": L["rd_avg"],
                "members": members, "mass": ptrim(mass),
                "vmax": max((m[0] for m in members), key=lambda p: pev(p, 10**6)),
            })
        covered = [Fraction(0)]
        for L in self.levels:
            covered = padd(covered, L["mass"])
        if self.model == "inf":
            self.cold = [Fraction(0)]
        else:
            self.cold = ptrim(padd(self.total, pneg(covered)))
        # raw piecewise evaluators (guard-correct at any concrete n),
        # gated by their region domains like the extraction pass
        dslp = os.path.join(HERE, "dsl", name.split(".")[0] + ".dsl")
        header = open(dslp).read().split(";")[0]
        self.param_names = [p.strip()
                            for p in header.replace("params", "").split(",")]
        self.raw_total = Piecewise(raw["total_accesses_plain"])
        self.raw = [(Piecewise(e["value_plain"]), Piecewise(e["mass_plain"]),
                     [r["domain_plain"] for r in e.get("regions", [])])
                    for e in raw["rd_distribution"]]

    def env(self, nv):
        e = {}
        for p in self.param_names:
            m = self.binding.get(p, Fraction(1))
            val = Fraction(nv) * m
            if val.denominator != 1:
                raise ValueError(f"n={nv} incompatible with binding {p}={m}n")
            e[p] = val
        return e

    def raw_entries_env(self, env):
        """[(rd, mass)] with region-domain gating, plus the total, at env."""
        out = []
        for v, w, doms in self.raw:
            if doms and not any(domain_satisfiable(d, env) for d in doms):
                continue
            wv = w({p: env[p] for p in w.params})
            if wv == 0:
                continue
            out.append((v({p: env[p] for p in v.params}), wv))
        return out, self.raw_total({p: env[p] for p in self.raw_total.params})

    def raw_entries(self, nv):
        return self.raw_entries_env(self.env(nv))

    def mr_env(self, C, env):
        entries, total = self.raw_entries_env(env)
        covered = sum(wv for _, wv in entries)
        miss = sum(wv for vv, wv in entries if vv > C)
        if self.model != "inf":
            miss += total - covered
        return miss / total

    # ---- exact concrete-n staircase (raw QP path) ------------------------
    def mr(self, C, nv):
        entries, total = self.raw_entries(nv)
        covered = sum(wv for _, wv in entries)
        miss = sum(wv for vv, wv in entries if vv > C)
        if self.model != "inf":
            miss += total - covered      # compulsory
        return miss / total

    def coverage(self, nv):
        entries, total = self.raw_entries(nv)
        return sum(wv for _, wv in entries) / total

    def dmd_per_access(self, nv):
        entries, total = self.raw_entries(nv)
        s = 0.0
        for vv, wv in entries:
            s += float(wv) * math.sqrt(max(0.0, float(vv)))
        return s / float(total)

    def min_cache_for(self, tau, nv):
        """Smallest C with mr(C, n) <= tau, from the raw rd thresholds."""
        entries, total = self.raw_entries(nv)
        entries.sort()
        covered = sum(wv for _, wv in entries)
        base_miss = (total - covered) if self.model != "inf" else Fraction(0)
        above = covered
        if (base_miss + above) / total <= tau:
            return Fraction(0)
        for vv, wv in entries:
            above -= wv
            if (base_miss + above) / total <= tau:
                return vv
        return None

    # ---- symbolic level presentation (fit path) --------------------------
    def displayed_levels(self):
        """Levels that either carry asymptotic mass or change the plateau."""
        tail = padd(self.cold, [Fraction(0)])
        tails = []
        for L in reversed(self.levels):
            tails.append(list(tail))
            tail = padd(tail, L["mass"])
        tails.reverse()          # tails[i] = miss mass strictly above level i
        before = padd(tail, [Fraction(0)])   # everything (= covered + cold)
        shown = []
        m_before = ratio_lead(before, self.total)
        for L, after_mass in zip(self.levels, tails):
            p = ratio_lead(L["mass"], self.total)
            m_after = ratio_lead(after_mass, self.total)
            significant_mass = p is not None and p[0] == 0
            plateau_change = (m_before is None) != (m_after is None) or (
                m_before is not None and m_after is not None
                and m_before != m_after)
            if significant_mass or plateau_change:
                shown.append({
                    "level": L,
                    "portion": ratio_lead_str(L["mass"], self.total),
                    "portion_lead": p,
                    "m_after": ratio_lead_str(after_mass, self.total),
                    "m_after_lead": m_after,
                })
            m_before = m_after
        return shown

    def footprint(self):
        """Largest rd scale = data footprint (in blocks)."""
        return self.levels[-1]["vmax"] if self.levels else [Fraction(0)]

    def nstar(self, level, C):
        """Largest n such that every reuse at this level fits in C blocks
        (asymptotic boundary from the fitted polynomial)."""
        vmax = level["vmax"]
        if pev(vmax, 10**12) <= C:
            return None          # never a constraint in practice
        if pev(vmax, 1) > C:
            return 0
        lo, hi = 1, 2
        while pev(vmax, hi) <= C and hi < 10**12:
            hi *= 2
        lo = hi // 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pev(vmax, mid) <= C:
                lo = mid
            else:
                hi = mid
        return lo


def load(name, model=None):
    models = (model,) if model else ("inf", "single")
    for m in models:
        p = os.path.join(HERE, "regimes", f"{name}.{m}.json")
        dpath = os.path.join(HERE, "data", f"{name}.{m}.json")
        if os.path.exists(p) and os.path.exists(dpath):
            rec = json.load(open(p))
            if "error" not in rec:
                return Kernel(name, rec, json.load(open(dpath)))
    return None


# ---------------------------------------------------------------- reports

SUITE_CACHES = [(512, "32 KB"), (16384, "1 MB"), (524288, "32 MB")]  # lines
# On the residue class of every quasi-polynomial branch in the suite
# (multiples of 32 cover mod-8/16 moduli and convolution's n/4 binding).
NSUITE = 2016
TAU = Fraction(1, 25)


COVERAGE_TOL = 0.01


def suite_kernels():
    """Load suite kernels, preferring infinite-repeat, keeping only models
    whose rd mass matches the access count to within COVERAGE_TOL (the
    conservation self-check).  Returns (kept, excluded) where excluded is
    [(name, best_coverage)]."""
    names = sorted({f.split(".")[0] for f in os.listdir(f"{HERE}/regimes")
                    if f.startswith("sym_")})
    if "sym_convolution9" in names:
        names.remove("sym_convolution")   # superseded by the 9x9 variant
    ks, excluded = [], []
    for name in names:
        best = None
        for model in ("inf", "single"):
            k = load(name, model)
            if k is None:
                continue
            cov = float(k.coverage(NSUITE))
            if best is None or abs(cov - 1) < abs(best[1] - 1):
                best = (k, cov)
            if abs(cov - 1) <= COVERAGE_TOL:
                ks.append(k)
                break
        else:
            if best is not None:
                excluded.append((name, best[1]))
    return ks, excluded


def fmt_mr(x):
    if x == 0:
        return "0"
    f = float(x)
    return f"{f:.3g}" if f >= 1e-4 else f"{f:.1e}"


def suite_regimes_md(ks):
    lines = ["# Regime tables (displayed levels)", "",
             "Kernel data is block-8: rd and cache capacity are in "
             "64-byte lines; all program parameters bound to n."]
    for k in ks:
        lines.append(f"\n## {k.name}  [{k.model}]")
        lines.append(f"footprint ≈ {lead_str(k.footprint())} lines; "
                     f"cold portion ≈ {ratio_lead_str(k.cold, k.total)}")
        lines.append("")
        lines.append("| boundary (lines) | portion | miss ratio after |")
        lines.append("|---|---|---|")
        for s in k.displayed_levels():
            lines.append(f"| {lead_str(s['level']['vmax'])} | {s['portion']} "
                         f"| {s['m_after']} |")
    return "\n".join(lines)


def machine_map_md(ks):
    lines = ["# Machine mapping at n = %d" % NSUITE, "",
             "Caches in 64-byte lines: " +
             ", ".join(f"{lbl} = {C} lines" for C, lbl in SUITE_CACHES) +
             ". Compute-bound threshold tau = 1/25 misses per access.", "",
             "| kernel | model | " +
             " | ".join(f"mr @ {lbl}" for _, lbl in SUITE_CACHES) +
             " | min cache for mr<=1/25 | coverage |",
             "|---|---|" + "---|" * (len(SUITE_CACHES) + 2)]
    for k in ks:
        mrs = [fmt_mr(k.mr(C, NSUITE)) for C, _ in SUITE_CACHES]
        c_req = k.min_cache_for(TAU, NSUITE)
        req = "-" if c_req is None else f"{float(c_req)*64/1024:.0f} KB"
        cov = float(k.coverage(NSUITE))
        lines.append(f"| {k.name} | {k.model} | " + " | ".join(mrs) +
                     f" | {req} | {cov:.4f} |")
    return "\n".join(lines)


def signatures_md(ks):
    lines = ["# Co-scaling signatures", "",
             "Boundary scales as powers of the data footprint D; plateau "
             "after each boundary as leading behavior in n.", ""]
    taxonomy = {}
    for k in ks:
        fd, _ = plead(k.footprint())
        sig = []
        for s in k.displayed_levels():
            bd, _ = plead(s["level"]["vmax"])
            m = s["m_after_lead"]
            mdeg = None if m is None else m[0]
            sig.append((bd, fd, mdeg))
        key = tuple(sig)
        taxonomy.setdefault(key, []).append(k.name)
        parts = []
        for s in k.displayed_levels():
            bd, _ = plead(s["level"]["vmax"])
            expo = "1" if bd == 0 else (f"D^{{{bd}/{fd}}}" if bd != fd else "D")
            parts.append(f"{expo} → {s['m_after']}")
        lines.append(f"- **{k.name}** [{k.model}] (deg D = {fd}): " +
                     "; ".join(parts))
    # coarse signature: distinct D-exponents with the final plateau at each
    lines.append("\n## Coarse signatures (per D-exponent, final plateau)\n")
    coarse_tax = {}
    for k in ks:
        fd, _ = plead(k.footprint())
        by_expo = {}
        for s in k.displayed_levels():
            bd, _ = plead(s["level"]["vmax"])
            by_expo[Fraction(bd, fd)] = s   # keep last at this exponent
        parts, ckey = [], []
        for expo in sorted(by_expo):
            s = by_expo[expo]
            m = s["m_after_lead"]
            mdeg = None if m is None else m[0]
            ckey.append((expo, mdeg))
            e = "O(1)" if expo == 0 else \
                ("D" if expo == 1 else f"D^{{{expo}}}")
            parts.append(f"{e} → {s['m_after']}")
        coarse_tax.setdefault(tuple(ckey), []).append(k.name)
        lines.append(f"- **{k.name}**: " + "; ".join(parts))
    lines.append("\n## Coarse taxonomy\n")
    lines.append("Key: (boundary scale as exponent of D, decay order of the "
                 "plateau after it; 0 = constant plateau, -1 = Θ(1/n), "
                 "None = zero).\n")
    for key, names in sorted(coarse_tax.items(), key=lambda kv: -len(kv[1])):
        pretty = ", ".join(f"(D^{str(e)}, {d})" for e, d in key)
        lines.append(f"- [{pretty}]: {', '.join(names)}")
    return "\n".join(lines), coarse_tax


def dmd_inversion_md(ks):
    rows = []
    for k in ks:
        rows.append({
            "name": k.name,
            "dmd": k.dmd_per_access(NSUITE),
            "dmd_2x": k.dmd_per_access(2 * NSUITE),
            "mrL1": float(k.mr(SUITE_CACHES[0][0], NSUITE)),
            "mrL2": float(k.mr(SUITE_CACHES[1][0], NSUITE)),
        })
    for r in rows:
        r["dmd_exp"] = math.log(r["dmd_2x"] / r["dmd"], 2) if r["dmd"] > 0 else 0
    by_dmd = sorted(rows, key=lambda r: -r["dmd"])
    inv = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            hi, lo = (a, b) if a["dmd"] >= b["dmd"] else (b, a)
            # inversion: clearly higher DMD yet clearly lower miss ratio at
            # the realistic cache (32 KB), and not higher at 1 MB
            if hi["dmd"] >= 1.1 * lo["dmd"] and \
               lo["mrL1"] >= 1.5 * hi["mrL1"] and \
               hi["mrL2"] <= 1.05 * lo["mrL2"]:
                score = (hi["dmd"] / max(lo["dmd"], 1e-12)) * \
                        (lo["mrL1"] / max(hi["mrL1"], 1e-12))
                inv.append((score, hi, lo))
    inv.sort(key=lambda t: -t[0])
    near = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            hi, lo = (a, b) if a["dmd"] >= b["dmd"] else (b, a)
            if hi["dmd"] <= 1.05 * lo["dmd"] and (
                    hi["mrL1"] >= 2 * lo["mrL1"] or lo["mrL1"] >= 2 * hi["mrL1"]):
                r = max(hi["mrL1"], lo["mrL1"]) / max(min(hi["mrL1"], lo["mrL1"]), 1e-12)
                near.append((r, hi, lo))
    near.sort(key=lambda t: -t[0])
    lines = ["# Scalar DMD vs miss ratio, n = %d" % NSUITE, "",
             "| kernel | DMD/access | growth exp | mr @ 32KB | mr @ 1MB |",
             "|---|---|---|---|---|"]
    for r in by_dmd:
        lines.append(f"| {r['name']} | {r['dmd']:.3g} | {r['dmd_exp']:.2f} "
                     f"| {fmt_mr(r['mrL1'])} | {fmt_mr(r['mrL2'])} |")
    lines.append("\n## Order inversions (>=1.1x higher DMD, >=1.5x lower "
                 "miss ratio at 32 KB, no worse at 1 MB)\n")
    for score, hi, lo in inv[:12]:
        lines.append(f"- {hi['name']} (DMD {hi['dmd']:.3g}) vs {lo['name']} "
                     f"(DMD {lo['dmd']:.3g}): mr@32KB {fmt_mr(hi['mrL1'])} vs "
                     f"{fmt_mr(lo['mrL1'])}, mr@1MB {fmt_mr(hi['mrL2'])} vs "
                     f"{fmt_mr(lo['mrL2'])}")
    lines.append("\n## Near-equal DMD (within 5%), miss ratio apart >=2x "
                 "at 32 KB\n")
    for r, hi, lo in near[:12]:
        lines.append(f"- {hi['name']} (DMD {hi['dmd']:.3g}, mr@32KB "
                     f"{fmt_mr(hi['mrL1'])}) vs {lo['name']} (DMD "
                     f"{lo['dmd']:.3g}, mr@32KB {fmt_mr(lo['mrL1'])}): "
                     f"{r:.1f}x apart")
    return "\n".join(lines), rows, inv


def tiling_md():
    N = 2048
    variants = [("matmul3", N, "naive"),
                ("matmul3_tile8", N // 8, "tile 8"),
                ("matmul3_tile16", N // 16, "tile 16"),
                ("matmul3_tile32", N // 32, "tile 32")]
    CACHES = [(64, "4 KB"), (128, "8 KB"), (512, "32 KB"),
              (8192, "512 KB"), (262144, "16 MB"),
              (1024 * 1024, "64 MB")]  # 64-byte lines
    lines = [f"# Naive vs tiled matmul (block 8, infinite repeat), N = {N}", "",
             "Cache sizes in 64-byte lines: " +
             ", ".join(f"{l} = {C}" for C, l in CACHES) + ".", "",
             "| variant | " + " | ".join(l for _, l in CACHES) + " |",
             "|---|" + "---|" * len(CACHES)]
    ks = {}
    for name, nv, label in variants:
        k = load(name)
        ks[label] = (k, nv)
        mrs = [fmt_mr(k.mr(C, nv)) for C, _ in CACHES]
        lines.append(f"| {label} | " + " | ".join(mrs) + " |")
    knaive, nn = ks["naive"]
    lines.append("\n## Pointwise gain over naive (traffic ratio)\n")
    lines.append("| variant | " + " | ".join(l for _, l in CACHES) + " |")
    lines.append("|---|" + "---|" * len(CACHES))
    for label in ("tile 8", "tile 16", "tile 32"):
        k, nv = ks[label]
        row = []
        for C, _ in CACHES:
            a, b = knaive.mr(C, nn), k.mr(C, nv)
            row.append("-" if b == 0 else f"{float(a / b):.1f}x")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    kikj = load("matmul3_ikj")
    if kikj is not None:
        lines.append("\n## Loop order (same 3-access body, ijk vs ikj)\n")
        lines.append("| variant | " + " | ".join(l for _, l in CACHES) + " |")
        lines.append("|---|" + "---|" * len(CACHES))
        for label, kk in (("ijk (k inner)", knaive), ("ikj (j inner)", kikj)):
            lines.append(f"| {label} | " +
                         " | ".join(fmt_mr(kk.mr(C, N)) for C, _ in CACHES) +
                         " |")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    ks, excluded = suite_kernels()
    print(f"{len(ks)} suite kernels loaded; {len(excluded)} excluded")
    open(f"{OUT}/suite_regimes.md", "w").write(suite_regimes_md(ks))
    open(f"{OUT}/machine_map.md", "w").write(machine_map_md(ks))
    sig_md, taxonomy = signatures_md(ks)
    open(f"{OUT}/signatures.md", "w").write(sig_md)
    inv_md, rows, inv = dmd_inversion_md(ks)
    open(f"{OUT}/dmd_inversion.md", "w").write(inv_md)
    open(f"{OUT}/tiling.md", "w").write(tiling_md())
    summary = {
        "kernels": [{"name": k.name, "model": k.model,
                     "footprint_lead": lead_str(k.footprint()),
                     "cold": ratio_lead_str(k.cold, k.total)} for k in ks],
        "excluded_by_conservation": [
            {"name": n, "best_coverage": c} for n, c in excluded],
        "n_inversions": len(inv),
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
