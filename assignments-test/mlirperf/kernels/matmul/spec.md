# Kernel `matmul` — optimize an MLIR affine loop nest for native performance

## What it computes
C += A*B   (all N x N).
Reference: `ref.mlir` in this directory (the naive version — your output must match it).

## You write MLIR, NOT C
Produce an **MLIR affine** kernel with this EXACT signature (do not change names/types/order):
```mlir
func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface}
```
Allowed dialects: `affine`, `arith`, `memref`, `func`. Use `affine.for` / `affine.load` /
`affine.store`. Express tiling with `affine.for ... step T` + `affine.min` upper bounds so
it is correct for ANY N (not just multiples of the tile). You may interchange, tile, fuse,
or unroll-and-jam the loops. Do NOT change the math (benign reassociation only).

## How it is compiled (single-core native; this is what the grader runs)
```sh
/usr/lib/llvm-22/bin/mlir-opt opt.mlir -lower-affine -convert-scf-to-cf \
  -convert-cf-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm \
  -convert-func-to-llvm -reconcile-unrealized-casts \
  | /usr/lib/llvm-22/bin/mlir-translate --mlir-to-llvmir \
  | /usr/lib/llvm-22/bin/llc -O3 -filetype=obj -o opt.o
clang -O3 -march=native driver.c opt.o -o bin     # then: taskset -c 0 ./bin N
```
`llc -O3` does the low-level vectorization for everyone — so the lever you control is the
**loop structure / locality**, not hand SIMD. `check.sh <file.mlir>` builds your kernel and
checks it is all-close to `ref.mlir`; use it to verify correctness (do NOT time/benchmark).

## Tested on THREE size regimes — you may submit one or three versions
  - **small**: N in [192, 384] (tested at an undisclosed N in this range)
  - **medium**: N in [512, 1024] (tested at an undisclosed N in this range)
  - **large**: N in [1152, 1536] (tested at an undisclosed N in this range)

Average speedup over the three regimes (vs `ref.mlir`) is your score. A loop structure that
wins at large N can lose at small N. You may submit a single `opt.mlir`, OR three tuned
versions `opt_small.mlir`, `opt_medium.mlir`, `opt_large.mlir` (same signature each).

## Correctness bar
Full output array(s) must match `ref.mlir` within `numpy.allclose(rtol=1e-6, atol=1e-9)`
at every regime, including N that are not multiples of your tile sizes.

## Deliverables (in THIS directory)
- `opt.mlir` (and optionally `opt_small.mlir` / `opt_medium.mlir` / `opt_large.mlir`)
- `rationale.md` — transformation(s) per regime and why they help locality.
