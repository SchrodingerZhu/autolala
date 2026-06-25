import json, os, re
from generate import KERNELS
HERE=os.path.dirname(os.path.abspath(__file__))
SIZES=json.load(open("sizes.json"))
# disclosed range per kernel (eval perf size lies inside, exact value hidden from agent)
RANGE={"matmul":(768,2048),"gemm":(768,2048),"2mm":(512,1536),"3mm":(512,1536),
 "mvt":(2048,8192),"atax":(2048,8192),"bicg":(2048,8192),"gemver":(2048,8192),
 "gesummv":(2048,8192),"syrk":(768,2048),"doitgen":(128,256),"covariance":(768,2048)}
for k,spec in KERNELS.items():
    sig=open(f"kernels/{k}/ref.c").read().split("{")[0].strip()  # signature line
    lo,hi=RANGE[k]
    spec_md=f"""# Kernel `{k}` — optimize for single-core native performance

## What it computes
{spec['desc']}

## Exact function signature (your opt.c MUST match this verbatim)
```c
{sig};
```
Arrays are flat row-major `double*`. `N` is the problem size, passed at runtime.

## Test conditions
- Compiled with: `clang -O3 -march=native -funroll-loops` (single translation unit, no LTO).
- Run pinned to ONE core (`taskset -c 0`). Optimize for single-core performance only.
- Evaluated at a problem size somewhere in the range **N in [{lo}, {hi}]** — the exact
  size is NOT disclosed, so DO NOT hardcode or special-case a specific N. Your code must
  be correct and fast for any N in that range, including N that are NOT multiples of any
  tile size you pick (handle remainder/boundary iterations).

## Correctness bar
Your kernel's full output array(s) must match the reference within
`numpy.allclose(rtol=1e-6, atol=1e-9)`. You may reorder floating-point operations
(tiling, blocking, interchange) — small reassociation is fine — but the result must
stay all-close. Do NOT change what is computed.

## Deliverables (write into THIS directory)
- `opt.c` — contains ONLY the optimized `kernel(...)` (same signature). No `main`.
- `rationale.md` — the transformation(s) you applied and why they speed it up.
"""
    for g in ("tool","ctrl"):
        d=f"runs/{g}-{k}"; os.makedirs(d,exist_ok=True)
        open(f"{d}/spec.md","w").write(spec_md)
        # provide the reference implementation read-only for them to study
        open(f"{d}/ref.c","w").write(open(f"kernels/{k}/ref.c").read())
print("set up", len(KERNELS)*2, "run dirs")
