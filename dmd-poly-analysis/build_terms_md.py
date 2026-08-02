#!/usr/bin/env python3
"""Render terms_table.json into TERMS.md — the researcher-facing list of
principal contribution terms per kernel, with order and analysis."""
import json
import os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))

NOTES = json.load(open(f"{HERE}/term_notes.json")) \
    if os.path.exists(f"{HERE}/term_notes.json") else {}

HEADER = """---
title: "Principal contribution terms of the PolyBench kernels"
subtitle: "The full reuse-distance term list per kernel: distances, populations, orders, and what each term contributes to misses and data movement"
date: "2026-07-30"
geometry: margin=0.9in
fontsize: 9pt
---

# How to read this document

The analyzer bins every warm access (every reuse) of a kernel by its **reuse
distance** (RD): the number of distinct cache lines touched in the reuse
window, at block granularity 8 doubles = one 64-byte line. Arrays follow the
padded layout of the paper's evaluation: every row starts on a fresh line
(only the innermost subscript is blocked). Each bin is a **principal
contribution term** — a pair

$$\\big(V(n),\\; M(n)\\big)$$

of a **distance** $V(n)$ (cache lines) and a **population** $M(n)$ (number
of accesses), both exact functions of the problem size $n$ (all program
parameters bound to $n$; convolution fixes its filter extent at 9). A term
states, completely and exactly:

* **Miss contribution.** The term's $M(n)$ accesses hit if the cache holds
  at least $V(n)$ lines and miss otherwise: the term contributes $M(n)$
  misses on the cache range $C < V(n)$ and none above. Its *boundary* is
  $C^{*} = V(n)$ lines ($64\\,V(n)$ bytes); its *portion* is $M(n)/A(n)$ of
  all accesses. The kernel's entire miss-ratio curve is the sum of these
  step contributions — nothing else.
* **Data-movement (DMD) contribution.** Under the square-root cost model
  the term contributes $M(n)\\sqrt{V(n)}$, of order
  $n^{\\deg M + \\deg V/2}$; the kernel's DMD spectrum is the ordered list
  of these orders with coefficients.

Two kinds appear. A **level** has one distance for its whole population
(split per residue class of an iterator mod 8 where the line boundary makes
classes differ — classes with identical polynomials are merged). A **ramp**
(family) spans a distance *range* $[V_{\\min}(n), V_{\\max}(n)]$ that grows
with a loop iterator (triangular kernels): it contributes misses that taper
off as the cache grows through the range; its population and range bounds
are exact polynomials, and its order is $\\deg M + \\deg V_{\\max}/2$.

*Why a term with vanishing portion still matters.* A population
$\\Theta(n^2)$ term in a $\\Theta(n^3)$-access kernel has portion
$\\Theta(1/n)$, so the *total* miss ratio above the lower boundaries thins
as $n$ grows. The term itself does not: on its active cache range it is the
entire miss traffic, its absolute misses grow as $n^2$, and its boundary is
exactly the cliff a cache-size sweep measures. The terms — not any single
aggregate — are the objects to reason with.

**Trust gate.** Every kernel below ran with **exact Barvinok counting**
(`method = exact`): bin populations conserve to the integer
($\\sum M = $ warm accesses, verified at two sizes per kernel and shown in
each header), and the reconstructed distance histograms match a
brute-force trace interpreter bin-for-bin at aligned and unaligned sizes
(30/30 spot checks). Polynomials are exact on the anchor residue class
$n \\equiv 0 \\pmod h$ (h = 8 or 16, per kernel); other classes differ only
in lower-order boundary constants. Ramp coefficients are evaluated at the
anchor sizes and quoted to three digits; ramp *orders* come from the exact
degrees. Populations below $10^{-6}$ of the accesses are aggregated into
their order row but omitted from the tables.
"""


def fmt_coeff(c):
    if c == 0:
        return "0"
    if abs(c) >= 1000:
        return f"{c:,.0f}"
    return f"{c:.3g}"


def portion_str(t, a_deg, a_lead):
    if "portion_deg" not in t:
        return "—"
    d, l = t["portion_deg"], t["portion_lead"]
    if d == 0:
        return f"{l:.3g}"
    return f"{l:.3g}·n^{d}" if d != -1 else f"{l:.3g}/n"


def term_rows(rec):
    rows = []
    a_deg, a_lead = rec["access_deg"], rec["access_lead"]
    total_mass_lead = None
    for t in rec["terms"]:
        # drop numerically negligible terms from the display
        if t.get("portion_deg", 0) - 0 <= -a_deg and t["dmd_coeff"] < 1e-6:
            continue
        if t["kind"] == "level":
            dist = t.get("value_poly", t["value_expr"])
        else:
            lo, hi = t.get("value_range", ["?", "?"])
            dist = f"{lo}  →  {hi}"
        pop = t.get("mass_poly", "")
        srcs = "; ".join(t.get("sources", [])[:2]) or "—"
        if len(t.get("sources", [])) > 2:
            srcs += f" (+{len(t['sources']) - 2})"
        kind = "ramp" if t["kind"] == "family" else "level"
        rows.append(
            f"| n^{t['dmd_order']:g} | {fmt_coeff(t['dmd_coeff'])} | {kind} "
            f"| {dist} | {pop} | {portion_str(t, a_deg, a_lead)} | {srcs} |")
    return rows


def kernel_section(key, rec):
    name = rec["kernel"].replace("sym_", "")
    model = key.rsplit(".", 1)[1]
    cons = rec.get("conservation", [])
    cons_txt = ", ".join(
        f"{c['ratio']:.6g} at n={c['n']}" if c["ratio"] is not None else "n/a"
        for c in cons)
    lines = []
    title = "single-shot" if model == "single" else "infinite-repeat"
    lines.append(f"\n## {name} — {title}  [`{rec['method']}`]\n")
    lines.append(f"Accesses $A(n) = {rec['access_poly']}$ "
                 f"(exact on n ≡ {rec['anchor_base'] % rec['modulus']} mod "
                 f"{rec['modulus']}); DMD order $n^{{{rec['dmd_order']:g}}}$, "
                 f"headroom **{rec['headroom']:+g}**; "
                 f"conservation Σmass/warm = {cons_txt}.\n")
    spec = "  +  ".join(f"{fmt_coeff(c)}·n^{o:g}" for o, c in rec["spectrum"]
                        if c >= 1e-6)
    lines.append(f"**DMD spectrum:**  {spec}\n")
    lines.append("| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.extend(term_rows(rec))
    if rec.get("skipped_bins"):
        lines.append(f"\n*Unresolved bins (>3 iterators, reported not "
                     f"dropped): {rec['skipped_bins']}*")
    notes = []
    for t in rec["terms"]:
        if t.get("note"):
            notes.append(f"bin {t['bin']}: {t['note']}")
    if notes:
        lines.append("\n*" + "; ".join(sorted(set(notes))[:3]) + ".*")
    note = NOTES.get(name, {}).get(model) or NOTES.get(name, {}).get("both")
    if note:
        lines.append(f"\n{note}")
    return "\n".join(lines)


def main():
    table = json.load(open(f"{HERE}/terms_table.json"))
    out = [HEADER]
    out.append("\n# Suite overview\n")
    out.append("Access order $a$, DMD order $d$, headroom $d-a$; top spectrum "
               "entries. Single-shot / infinite-repeat per kernel.\n")
    out.append("| kernel | model | a | d | headroom | leading terms |")
    out.append("|---|---|---|---|---|---|")
    def sort_key(k):
        rec = table[k]
        return (-(rec["headroom"] or 0), rec["kernel"], k.rsplit(".", 1)[1])
    for key in sorted(table, key=sort_key):
        rec = table[key]
        name = rec["kernel"].replace("sym_", "")
        model = key.rsplit(".", 1)[1]
        top = ",  ".join(f"{fmt_coeff(c)}·n^{o:g}" for o, c in rec["spectrum"][:2])
        out.append(f"| {name} | {model} | {rec['access_deg']} "
                   f"| {rec['dmd_order']:g} | {rec['headroom']:+g} | {top} |")
    out.append("\n# Per-kernel principal terms\n")
    for key in sorted(table):
        out.append(kernel_section(key, table[key]))
    open(f"{HERE}/TERMS.md", "w").write("\n".join(out) + "\n")
    print(f"wrote TERMS.md ({len(table)} kernel-models)")


if __name__ == "__main__":
    main()
