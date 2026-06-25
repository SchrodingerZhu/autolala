#!/usr/bin/env python3
"""Deterministic ground-truth scorer for affine-kernel variants.

Pipeline:  tagged .mlir  --(mlir-extract)-->  .dsl  --(dmd-cli --json)-->  DMD formula
We then evaluate the symbolic DMD formula numerically at a fixed problem size N
and cache block size B, yielding a single scalar = predicted data-movement cost.
Lower is better. The same evaluator is applied to every variant, so scores are
directly comparable across agents and against the untransformed baseline.
"""
import json, math, re, subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT = os.path.join(ROOT, "target/release/mlir-extract")
DMDCLI = os.path.join(ROOT, "target/release/dmd-cli")


def _env(expr, N):
    env = {"sqrt": math.sqrt, "floor": math.floor, "min": min, "max": max}
    for name in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)):
        if name not in env:
            env[name] = float(N)
    return env


def _ev(piece, N):
    piece = piece.replace("^", "**").strip().rstrip("+-").strip()
    if not piece:
        return 0.0
    return float(eval(piece, {"__builtins__": {}}, _env(piece, N)))


def eval_formula(expr: str, N: int) -> float:
    """Evaluate a dmd_formula_plain / access-count string at every param = N.

    Handles the piecewise tail of the form ``... + [domain] => value + [domain] => value``
    that some kernels emit for the compulsory-access term: only branches whose
    domain holds at p_i = N (large positive) contribute.
    """
    # split off the leading algebraic part from any guarded segments
    idx = expr.find("[")
    base = expr if idx < 0 else expr[:idx]
    total = _ev(base, N)
    if idx >= 0:
        # each segment: `domain] => value`  (value runs to the next `[`)
        for seg in expr[idx + 1:].split("["):
            dom, _, val = seg.partition("] =>")
            dom = dom.replace("^", "**")
            ok = bool(eval(dom, {"__builtins__": {}}, _env(dom, N)))
            if ok:
                total += _ev(val, N)
    return float(total)


def score_mlir(mlir_path: str, attr="dmd.extract", block_size=64, N=1024):
    dsl = subprocess.run([EXTRACT, mlir_path, "-a", attr],
                         capture_output=True, text=True)
    if dsl.returncode != 0:
        return {"ok": False, "stage": "extract", "error": dsl.stderr.strip()[:500]}
    p = subprocess.run([DMDCLI, "--block-size", str(block_size),
                        "--max-operations", "300000000", "--json"],
                       input=dsl.stdout, capture_output=True, text=True, timeout=400)
    if p.returncode != 0:
        return {"ok": False, "stage": "analyze", "error": p.stderr.strip()[:500],
                "dsl": dsl.stdout}
    d = json.loads(p.stdout)
    try:
        dmd = eval_formula(d["dmd_formula_plain"], N)
        total = eval_formula(d["total_accesses_plain"], N)
        warm = eval_formula(d["warm_accesses_plain"], N)
        comp = eval_formula(d["compulsory_accesses_plain"], N)
    except Exception as e:
        return {"ok": False, "stage": "eval", "error": str(e)}
    return {"ok": True, "dmd": dmd, "total": total, "warm": warm,
            "compulsory": comp, "miss_ratio": (total - warm) / total if total else None,
            "dmd_formula": d["dmd_formula_plain"], "dsl": dsl.stdout,
            "N": N, "block_size": block_size}


if __name__ == "__main__":
    res = score_mlir(sys.argv[1], N=int(sys.argv[2]) if len(sys.argv) > 2 else 1024)
    print(json.dumps({k: v for k, v in res.items() if k not in ("dsl", "dmd_formula")}, indent=2))
