# How much data does PolyBench move? A reuse-distance study

Symbolic **data-movement (DMD)** analysis of the PolyBench kernels with the
AutoLALA `dmd` analyzer (scale approximation, block size **8 elements** = one
64-byte cache line for `f64`), under **two execution models** — single-shot and
infinitely-repeating — plus empirical confirmation on real cache hardware.

**Read `REPORT.md` / `REPORT.pdf` for the full study.** In one paragraph:

> Each kernel's data movement grows like `coeff · N^d`, and the growth exponent
> obeys one rule: `d = a + ½·ρ`, where `a` is how fast the access count grows and
> `ρ` is how fast the largest **reuse distances** grow. The quantity `d − a =
> ½ρ` — we call it the **headroom** — is how much faster data movement grows than
> arithmetic, i.e. how much a locality transformation can recover. Across the
> suite the headroom lands on exactly **0, ½, or 1**, splitting the kernels into
> three classes: **1.0** (gemm/2mm/3mm/doitgen/floyd/jacobi-2d/seidel-2d — tile
> for a factor of `N`), **0.5** (mvt/atax/bicg/gemver/gesummv/covariance/… —
> interchange/fuse for a factor of `√N`), and **0.0** (syrk/syr2k/cholesky/lu/
> trmm/trisolve — already local, no locality slack). Within a class, the leading
> **coefficient** ranks kernels (3mm moves 3× gemm's data at the same `N^4`).
> Switching to the infinite-repeat model **reveals hidden cross-invocation reuse**
> in the kernels that stream their matrix once per pass (atax/bicg/gesummv rise
> `N^2.5→N^3`, trisolve `N^2→N^3`). Confirmed on hardware: matmul's last-level
> misses jump 50× at the cache-crossing size (tiling holds them 22× lower), mvt's
> grow 3× (interchange caps them 4× lower), syrk's stay flat.

## Layout

| path | contents |
|------|----------|
| `REPORT.md`, `REPORT.pdf` | the full write-up (definitions, the law, results, three classes, infinite-repeat, empirical confirmation) |
| `run_analysis.py` | batch: tag → `mlir-extract` → `dmd-cli --json`, **both** single-shot and `--infinite-repeat`, over every kernel |
| `analyze_math.py` | growth rates + leading coefficients (`d = a + ½ρ`, `DMD ≈ coeff·N^d`) |
| `local_analysis.py` | finite-range analysis: local exponent `p(N)`, exact doubling cost, cache threshold `N* = √(C/c)` |
| `results/<k>.json` | per-kernel `single` and `inf` records: RI/RD distributions, DMD formula, access counts |
| `order_table.json` | computed growth rates, headroom, and coefficients for the 27 symbolic kernels, both models |
| `confirm/` | empirical confirmation: `k.c` kernels, `sweep.py` cachegrind sweep, `cg.json`, `runtime.json` |
| `dsl/<k>.dsl` | the extracted AutoLALA DSL fed to the analyzer |

## Reproduce

```sh
python3 run_analysis.py both --resume   # analyze all kernels, both models -> results/
python3 analyze_math.py                 # growth rates + coefficients -> order_table.json
python3 local_analysis.py               # local exponent / doubling cost / cache threshold N*
cd confirm && python3 sweep.py          # cachegrind miss-scaling -> cg.json
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex \
   -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono"
```

## Notes

- **Block size is in elements, not bytes.** A 64-byte line holds 8 doubles, so we
  use `--block-size 8` (16 would be single precision). This matches the legacy
  AutoLALA run and the line cachegrind simulates.
- **`--infinite-repeat`** is a new `dmd-cli` flag added on this branch (see the
  report's implementation notes); it wraps the kernel in a two-pass outer loop and
  keeps the steady-state (second-pass) reuse intervals.
- Coverage: 48/53 programs analyze single-shot (up from 41 after the reduction-loop
  extractor fix), 43 also under infinite-repeat. The rest are non-affine
  (`adi`/`deriche`/`durbin`), over budget (`heat-3d`), or excluded (`fdtd-apml`).
