# DMD / Reuse-Interval analysis of PolyBench

Symbolic RI/RD/DMD analysis (AutoLALA `dmd`, **scale approximation**, block size 64) of all
53 PolyBench programs in `../autolala/analyzer/misc/polybench`, a rigorous order analysis of
the resulting formulas, and empirical confirmation.

**Read `REPORT.md` / `REPORT.pdf` for the full analysis.** Headline:

> DMD obeys one law — `ord(DMD) = access_order + ½·ρ`, where `ρ` is the dominant
> reuse-distance exponent — and `ρ` is **quantised to {0,1,2}** across all kernels. The gap
> `g = ½ρ` is the asymptotic tiling headroom: Class A (`g=1`: gemm/2mm/3mm/doitgen/
> jacobi-2d/seidel-2d/floyd) wins `~N`, Class B (`g=½`: mvt/atax/bicg/gemver/gesummv/
> covariance/…) wins `~√N` by fixing the transposed/re-streamed access, Class C (`g=0`:
> syrk/syr2k/cholesky/lu/trmm/trisolve) has no asymptotic headroom. Confirmed by cachegrind
> miss-scaling (matmul's LL misses jump 50× exactly at the N²=cache threshold; syrk flat) and
> raw runtime.

## Layout

| path | contents |
|------|----------|
| `REPORT.md`, `REPORT.pdf` | the full rigorous analysis + empirical confirmation |
| `run_analysis.py` | batch: tag → `mlir-extract` → `dmd-cli --json` (scale) over all kernels |
| `analyze_math.py` | order analysis (the `ord(DMD)=a+½ρ` identity, per-kernel) |
| `results/<k>.json` | per-kernel RI distribution, RD distribution, DMD formula, access counts |
| `dsl/<k>.dsl` | the extracted AutoLALA DSL fed to the analyzer |
| `order_table.json` | computed orders/gaps for the 24 symbolic kernels |
| `summary.json` | per-kernel analysis status (41/53 ok; 12 non-affine/over-budget) |
| `confirm/` | empirical confirmation: `k.c` kernels, `sweep.py` cachegrind sweep, `cg.json`, `runtime.json` |

## Reproduce

```sh
python3 run_analysis.py both     # ~10 min: analyzes all 53 kernels -> results/
python3 analyze_math.py          # prints the order table -> order_table.json
cd confirm && python3 sweep.py   # cachegrind miss-scaling confirmation -> cg.json
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex \
   -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono"
```
