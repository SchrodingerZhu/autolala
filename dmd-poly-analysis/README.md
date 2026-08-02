# Principal contribution terms of PolyBench (reuse-distance study)

Symbolic **reuse-distance** analysis of the PolyBench kernels with the
AutoLALA `dmd` analyzer at block size **8 elements** (= one 64-byte line of
`f64`), under two execution models — single-shot and infinitely-repeating —
with **exact Barvinok counting** wherever it completes (the scale
approximation only as a marked fallback).

**The deliverable is `TERMS.md` / `TERMS.pdf`**: for every kernel, the full
list of *principal contribution terms* — each reuse-distance bin as a pair
(distance `V(n)`, population `M(n)`), both exact in the problem size — with
each term's order, the cache range where it contributes misses
(`C < V(n)` lines), its DMD contribution `M·√V`, and the source access it
belongs to. `REPORT.md` is the older narrative study; where its numbers
disagree with `TERMS.md`, `TERMS.md` (produced after the analyzer fixes
below) supersedes it.

## Analyzer fixes this iteration (all in `crates/dmd-core`)

The isl/Barvinok arithmetic was always right; the *rendered* formulas were
not. Six bugs fixed, each pinned by a unit test or suite self-check:

1. **Floors dropped**: qpoly div factors rendered as plain rational affines
   (`-floor((1+i)/8)+floor((7+i)/8)` printed `3/4`). New `Floor` node.
2. **Double division**: isl `*_val` getters already divide; dividing by the
   denominator again turned `floor((i+1)/8)` into `(i+1)/64`.
3. **Guard dropped on single-piece renders**: masses evaluated garbage
   outside their validity region; warm/compulsory now isl-side (pw add/sub).
4. **Zero-count cells counted as warm**: distribution pieces now restricted
   to the counted relation's domain.
5. **`-1/k` coefficients lost their magnitude** in sums (`-1/2·x` → `-x`).
6. **Set-dim monomials dropped**: isl terms expose exponents on param/set
   slots only; querying `in` silently erased every iterator monomial, so
   triangular values like `-7 + 2·i1` rendered as constants. This corrupted
   all previous triangular-kernel distances.

With the fixes and exact counting, **mass conservation holds to the
integer** suite-wide (`verify_conservation.py`), and the reconstructed
distance histograms match a brute-force trace interpreter bin-for-bin
(`validate_bins.py`).

## Layout

| path | contents |
|------|----------|
| `TERMS.md` / `TERMS.pdf` | **the deliverable**: per-kernel principal terms, orders, boundaries, attribution |
| `terms_table.json` | machine-readable term table (exact polynomials on the anchor residue class) |
| `REPORT.md`, `REPORT.pdf` | the older narrative study (see supersession note at top) |
| `run_analysis.py` | batch: tag → `mlir-extract` → `dmd-cli --json`, exact-first with scale fallback, both models |
| `terms_analysis.py` | RD bins → principal terms: levels, residue-class splits, ramp families; exact rational fits |
| `build_terms_md.py` | renders `terms_table.json` → `TERMS.md` |
| `qpeval.py` | exact evaluator for the analyzer's plain-formula syntax (piecewise, floor, mod) |
| `schedule_map.py` | maps analyzer `sources` signatures back to DSL accesses |
| `dsl_sim.py` | reference trace interpreter (ground truth) |
| `verify_conservation.py` | suite gate: Σ bin masses = warm accesses, exactly |
| `validate_bins.py` | bin-for-bin histogram check vs the trace interpreter |
| `results/<k>.json` | per-kernel `single`/`inf` records (RI/RD bins with exact masses + sources) |
| `dsl/<k>.dsl` | extracted AutoLALA DSL per kernel |
| `exact/`, `confirm/` | older exact-trace simulator and cachegrind confirmation |

## Reproduce

```sh
python3 run_analysis.py both            # analyze all kernels -> results/
python3 verify_conservation.py          # conservation gate (exact: to the integer)
python3 validate_bins.py                # bin-level check vs brute-force trace
python3 terms_analysis.py               # principal terms -> terms_table.json
python3 build_terms_md.py               # -> TERMS.md
pandoc TERMS.md -o TERMS.pdf --pdf-engine=xelatex \
   -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono"
```

Notes: block size is in elements (8 doubles = one 64-byte line);
`--infinite-repeat` wraps the kernel in a two-pass loop and keeps
second-pass reuse intervals (steady state); `--approximation-method exact`
is the default here — `scale` is only used where exact exceeds the operation
budget, and such records are marked `method: scale`. Non-affine kernels
(`adi`, `deriche`, `durbin`) remain out of scope.
