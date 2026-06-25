#!/usr/bin/env python3
"""Generate the MLIR-affine performance experiment (6 kernels).

Per kernel emits:
  kernels/<k>/ref.mlir   -- naive affine reference (fixed signature, C interface)
  kernels/<k>/driver.c   -- builds memref descriptors, inits, calls _mlir_ciface_kernel, dumps outputs
  kernels/<k>/spec.md     -- the contract handed to the agent (signature + math + 3 size regimes + rules)

Agents write opt.mlir (or opt_small/medium/large.mlir) with the SAME signature.
Ground truth: lower MLIR -> LLVM -> native (llvm-22), link the driver, compare full
output arrays with numpy.allclose, time with hyperfine. The ONLY hand-written code an
agent can produce is the affine loop structure -> isolates the locality dimension.
"""
import os, json

RANK_T = {1: "memref<?xf64>", 2: "memref<?x?xf64>"}

KERNELS = {
"matmul": dict(
  arrays=[("A",2,"rand"),("B",2,"rand"),("C",2,"zero")], outs=["C"],
  desc="C += A*B   (all N x N).",
  body="""affine.for %i = 0 to %N { affine.for %j = 0 to %N { affine.for %k = 0 to %N {
      %a = affine.load %A[%i,%k] : memref<?x?xf64>
      %b = affine.load %B[%k,%j] : memref<?x?xf64>
      %c = affine.load %C[%i,%j] : memref<?x?xf64>
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %c, %p : f64
      affine.store %s, %C[%i,%j] : memref<?x?xf64>
    }}}"""),

"gemm": dict(
  arrays=[("A",2,"rand"),("B",2,"rand"),("C",2,"rand")], outs=["C"],
  desc="GEMM: C = 0.9*C + A*B   (all N x N).",
  body="""%b9 = arith.constant 0.9 : f64
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %c = affine.load %C[%i,%j] : memref<?x?xf64>
      %cs = arith.mulf %c, %b9 : f64
      affine.store %cs, %C[%i,%j] : memref<?x?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N { affine.for %k = 0 to %N {
      %a = affine.load %A[%i,%k] : memref<?x?xf64>
      %b = affine.load %B[%k,%j] : memref<?x?xf64>
      %c = affine.load %C[%i,%j] : memref<?x?xf64>
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %c, %p : f64
      affine.store %s, %C[%i,%j] : memref<?x?xf64>
    }}}"""),

"2mm": dict(
  arrays=[("A",2,"rand"),("B",2,"rand"),("C",2,"rand"),("D",2,"rand"),("T",2,"zero")], outs=["D"],
  desc="2mm: T = A*B ; D = 0.9*D + T*C   (all N x N; T starts at 0).",
  body="""%b9 = arith.constant 0.9 : f64
    affine.for %i = 0 to %N { affine.for %j = 0 to %N { affine.for %k = 0 to %N {
      %a = affine.load %A[%i,%k] : memref<?x?xf64>
      %b = affine.load %B[%k,%j] : memref<?x?xf64>
      %t = affine.load %T[%i,%j] : memref<?x?xf64>
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %t, %p : f64
      affine.store %s, %T[%i,%j] : memref<?x?xf64>
    }}}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %d = affine.load %D[%i,%j] : memref<?x?xf64>
      %ds = arith.mulf %d, %b9 : f64
      affine.store %ds, %D[%i,%j] : memref<?x?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N { affine.for %k = 0 to %N {
      %t = affine.load %T[%i,%k] : memref<?x?xf64>
      %c = affine.load %C[%k,%j] : memref<?x?xf64>
      %d = affine.load %D[%i,%j] : memref<?x?xf64>
      %p = arith.mulf %t, %c : f64
      %s = arith.addf %d, %p : f64
      affine.store %s, %D[%i,%j] : memref<?x?xf64>
    }}}"""),

"mvt": dict(
  arrays=[("A",2,"rand"),("x1",1,"rand"),("x2",1,"rand"),("y1",1,"rand"),("y2",1,"rand")],
  outs=["x1","x2"],
  desc="mvt: x1 += A*y1 ; x2 += A^T*y2   (A is N x N; second pass reads A transposed).",
  body="""affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %x = affine.load %x1[%i] : memref<?xf64>
      %a = affine.load %A[%i,%j] : memref<?x?xf64>
      %y = affine.load %y1[%j] : memref<?xf64>
      %p = arith.mulf %a, %y : f64
      %s = arith.addf %x, %p : f64
      affine.store %s, %x1[%i] : memref<?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %x = affine.load %x2[%i] : memref<?xf64>
      %a = affine.load %A[%j,%i] : memref<?x?xf64>
      %y = affine.load %y2[%j] : memref<?xf64>
      %p = arith.mulf %a, %y : f64
      %s = arith.addf %x, %p : f64
      affine.store %s, %x2[%i] : memref<?xf64>
    }}"""),

"atax": dict(
  arrays=[("A",2,"rand"),("x",1,"rand"),("y",1,"zero"),("T",1,"zero")], outs=["y"],
  desc="atax: T = A*x ; y = A^T*T   (A is N x N; T,y start at 0).",
  body="""affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %tv = affine.load %T[%i] : memref<?xf64>
      %a = affine.load %A[%i,%j] : memref<?x?xf64>
      %xv = affine.load %x[%j] : memref<?xf64>
      %p = arith.mulf %a, %xv : f64
      %s = arith.addf %tv, %p : f64
      affine.store %s, %T[%i] : memref<?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %yv = affine.load %y[%j] : memref<?xf64>
      %a = affine.load %A[%i,%j] : memref<?x?xf64>
      %tv = affine.load %T[%i] : memref<?xf64>
      %p = arith.mulf %a, %tv : f64
      %s = arith.addf %yv, %p : f64
      affine.store %s, %y[%j] : memref<?xf64>
    }}"""),

"syrk": dict(
  arrays=[("A",2,"rand"),("C",2,"rand")], outs=["C"],
  desc="syrk: C = 0.9*C + A*A^T, LOWER triangle only (j<=i). A,C are N x N.",
  body="""%b9 = arith.constant 0.9 : f64
    affine.for %i = 0 to %N { affine.for %j = 0 to #tri(%i) {
      %c = affine.load %C[%i,%j] : memref<?x?xf64>
      %cs = arith.mulf %c, %b9 : f64
      affine.store %cs, %C[%i,%j] : memref<?x?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %k = 0 to %N { affine.for %j = 0 to #tri(%i) {
      %c = affine.load %C[%i,%j] : memref<?x?xf64>
      %aik = affine.load %A[%i,%k] : memref<?x?xf64>
      %ajk = affine.load %A[%j,%k] : memref<?x?xf64>
      %p = arith.mulf %aik, %ajk : f64
      %s = arith.addf %c, %p : f64
      affine.store %s, %C[%i,%j] : memref<?x?xf64>
    }}}"""),
}

# small / medium / large : (lo, hi) disclosed ranges + the hidden test N
SIZES = {
 "matmul": {"small":[192,384,256],"medium":[512,1024,768],"large":[1152,1536,1280]},
 "gemm":   {"small":[192,384,256],"medium":[512,1024,768],"large":[1152,1536,1280]},
 "2mm":    {"small":[160,320,224],"medium":[448,768,576],"large":[896,1152,1024]},
 "syrk":   {"small":[192,384,256],"medium":[512,1024,768],"large":[1152,1664,1408]},
 "mvt":    {"small":[1024,2048,1536],"medium":[3072,5120,4096],"large":[6144,8192,7168]},
 "atax":   {"small":[1024,2048,1536],"medium":[3072,5120,4096],"large":[6144,8192,7168]},
}

SIG = lambda arrs: ", ".join(f"%{n}: {RANK_T[r]}" for n,r,_ in arrs) + ", %N: index"
TRI = '#tri = affine_map<(d0) -> (d0 + 1)>\n'

REF_TMPL = """{maps}module {{
  func.func @kernel({sig}) attributes {{llvm.emit_c_interface}} {{
    {body}
    return
  }}
}}
"""

def c_driver(name, spec):
    arrs = spec["arrays"]
    decl_ptr = ",".join("void*" for _ in arrs) + ",long"
    # struct + mk for ranks used
    structs = """typedef struct{double*a,*b;long o,s[1],t[1];}MR1;
typedef struct{double*a,*b;long o,s[2],t[2];}MR2;
static MR1 mk1(double*p,long n){MR1 m={p,p,0,{n},{1}};return m;}
static MR2 mk2(double*p,long r,long c){MR2 m={p,p,0,{r,c},{c,1}};return m;}"""
    allocs, inits, mks, args = [], [], [], []
    for n,r,init in arrs:
        cnt = "N" if r==1 else "N*N"
        allocs.append(f"  double*{n}=malloc(8L*{cnt});")
        if init=="rand": inits.append(f"  for(long i=0;i<{cnt};i++){{R;{n}[i]=U;}}")
        else: inits.append(f"  for(long i=0;i<{cnt};i++){n}[i]=0.0;")
        mks.append(f"  MR{r} d{n}={'mk1('+n+',N)' if r==1 else 'mk2('+n+',N,N)'};")
        args.append(f"&d{n}")
    dumps = "".join(f'fwrite({o},8,{"N" if dict((x[0],x[1]) for x in arrs)[o]==1 else "N*N"},f);' for o in spec["outs"])
    sink = spec["outs"][0]
    return f"""#include <stdio.h>
#include <stdlib.h>
{structs}
extern void _mlir_ciface_kernel({decl_ptr});
int main(int argc,char**argv){{
  long N=argc>1?atol(argv[1]):512;
  unsigned long long s=88172645463325252ULL;
  #define R s^=s<<13;s^=s>>7;s^=s<<17
  #define U ((s>>11)&((1ULL<<53)-1))/(double)(1ULL<<53)
{chr(10).join(allocs)}
{chr(10).join(inits)}
{chr(10).join(mks)}
  _mlir_ciface_kernel({",".join(args)},N);
  volatile double sink={sink}[0];(void)sink;
  if(argc>2){{FILE*f=fopen(argv[2],"wb");{dumps}fclose(f);}}
  return 0;
}}
"""

def spec_md(name, spec):
    arrs = spec["arrays"]; sz = SIZES[name]
    sig = f"func.func @kernel({SIG(arrs)}) attributes {{llvm.emit_c_interface}}"
    regimes = "\n".join(f"  - **{r}**: N in [{sz[r][0]}, {sz[r][1]}] (tested at an undisclosed N in this range)"
                        for r in ("small","medium","large"))
    return f"""# Kernel `{name}` — optimize an MLIR affine loop nest for native performance

## What it computes
{spec['desc']}
Reference: `ref.mlir` in this directory (the naive version — your output must match it).

## You write MLIR, NOT C
Produce an **MLIR affine** kernel with this EXACT signature (do not change names/types/order):
```mlir
{sig}
```
Allowed dialects: `affine`, `arith`, `memref`, `func`. Use `affine.for` / `affine.load` /
`affine.store`. Express tiling with `affine.for ... step T` + `affine.min` upper bounds so
it is correct for ANY N (not just multiples of the tile). You may interchange, tile, fuse,
or unroll-and-jam the loops. Do NOT change the math (benign reassociation only).

## How it is compiled (single-core native; this is what the grader runs)
```sh
/usr/lib/llvm-22/bin/mlir-opt opt.mlir -lower-affine -convert-scf-to-cf \\
  -convert-cf-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm \\
  -convert-func-to-llvm -reconcile-unrealized-casts \\
  | /usr/lib/llvm-22/bin/mlir-translate --mlir-to-llvmir \\
  | /usr/lib/llvm-22/bin/llc -O3 -filetype=obj -o opt.o
clang -O3 -march=native driver.c opt.o -o bin     # then: taskset -c 0 ./bin N
```
`llc -O3` does the low-level vectorization for everyone — so the lever you control is the
**loop structure / locality**, not hand SIMD. `check.sh <file.mlir>` builds your kernel and
checks it is all-close to `ref.mlir`; use it to verify correctness (do NOT time/benchmark).

## Tested on THREE size regimes — you may submit one or three versions
{regimes}

Average speedup over the three regimes (vs `ref.mlir`) is your score. A loop structure that
wins at large N can lose at small N. You may submit a single `opt.mlir`, OR three tuned
versions `opt_small.mlir`, `opt_medium.mlir`, `opt_large.mlir` (same signature each).

## Correctness bar
Full output array(s) must match `ref.mlir` within `numpy.allclose(rtol=1e-6, atol=1e-9)`
at every regime, including N that are not multiples of your tile sizes.

## Deliverables (in THIS directory)
- `opt.mlir` (and optionally `opt_small.mlir` / `opt_medium.mlir` / `opt_large.mlir`)
- `rationale.md` — transformation(s) per regime and why they help locality.
"""

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for name, spec in KERNELS.items():
        d = f"kernels/{name}"; os.makedirs(d, exist_ok=True)
        maps = TRI if "#tri" in spec["body"] else ""
        open(f"{d}/ref.mlir","w").write(REF_TMPL.format(maps=maps, sig=SIG(spec["arrays"]), body=spec["body"]))
        open(f"{d}/driver.c","w").write(c_driver(name, spec))
        open(f"{d}/spec.md","w").write(spec_md(name, spec))
    json.dump(SIZES, open("sizes.json","w"), indent=2)
    print("generated", len(KERNELS), "kernels:", " ".join(KERNELS))
