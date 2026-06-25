#!/usr/bin/env python3
"""Evaluate an optimized kernel: correctness (numpy.allclose on full output arrays)
+ raw single-core performance (hyperfine wall-clock vs the reference).

Usage: eval.py <kernel> <opt.c>   (defaults to kernels/<kernel>/ref.c)
"""
import json, os, re, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = json.load(open(os.path.join(HERE, "sizes.json")))
CFLAGS = ["-O3", "-march=native", "-funroll-loops"]
RTOL, ATOL = 1e-6, 1e-9


def build(kernel, src, out):
    drv = os.path.join(HERE, "kernels", kernel, "driver.c")
    p = subprocess.run(["clang", *CFLAGS, drv, src, "-lm", "-o", out],
                       capture_output=True, text=True)
    return (p.returncode == 0), p.stderr


def run_dump(binp, N):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name
    subprocess.run([binp, str(N), f], check=True, timeout=600)
    a = np.fromfile(f, dtype=np.float64)
    os.unlink(f)
    return a


def hyperfine(binp, N, runs=8):
    cmd = f"taskset -c 0 {binp} {N}"
    p = subprocess.run(["hyperfine", "-N", "-w", "2", "-r", str(runs),
                        "--export-json", "/dev/stdout", cmd],
                       capture_output=True, text=True, timeout=900)
    m = re.search(r'"mean":\s*([0-9.eE+-]+)', p.stdout)
    return float(m.group(1)) if m else None


def evaluate(kernel, opt_src):
    ref_src = os.path.join(HERE, "kernels", kernel, "ref.c")
    cfg = SIZES[kernel]
    res = {"kernel": kernel, "build_ok": False, "correct": False, "speedup": None}

    rb, ob = f"/tmp/ev_ref_{kernel}", f"/tmp/ev_opt_{kernel}"
    ok, err = build(kernel, ref_src, rb)
    if not ok:
        res["error"] = "ref build failed: " + err[:300]; return res
    ok, err = build(kernel, opt_src, ob)
    res["build_ok"] = ok
    if not ok:
        res["error"] = "opt build failed: " + err[:400]; return res

    # correctness: full output arrays must be all-close at every test size
    res["corr_detail"] = []
    allgood = True
    for N in cfg["corr"]:
        ra, oa = run_dump(rb, N), run_dump(ob, N)
        if ra.shape != oa.shape:
            allgood = False; res["corr_detail"].append({"N": N, "ok": False, "why": "shape"}); continue
        close = np.allclose(ra, oa, rtol=RTOL, atol=ATOL)
        maxrel = float(np.max(np.abs(ra - oa) / (np.abs(ra) + ATOL)))
        res["corr_detail"].append({"N": N, "ok": bool(close), "max_rel_err": maxrel})
        allgood = allgood and close
    res["correct"] = allgood

    # performance (only meaningful if correct)
    P = cfg["perf"]
    tref = hyperfine(rb, P); topt = hyperfine(ob, P)
    res["perf_N"] = P; res["t_ref_s"] = tref; res["t_opt_s"] = topt
    if tref and topt:
        res["speedup"] = tref / topt
    return res


if __name__ == "__main__":
    kernel = sys.argv[1]
    src = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "kernels", kernel, "ref.c")
    print(json.dumps(evaluate(kernel, src), indent=2))
