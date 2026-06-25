#!/usr/bin/env python3
"""Grade all MLIR-affine submissions: per-regime correctness (allclose) + hyperfine
speedup vs ref.mlir, averaged over small/medium/large. Ground truth = real runtime."""
import json, os, re, subprocess, tempfile, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = json.load(open(f"{HERE}/sizes.json"))
LOWER = f"{HERE}/lower.sh"
KS = ["matmul", "gemm", "2mm", "mvt", "atax", "syrk"]
RTOL, ATOL = 1e-6, 1e-9
CORR_N = [96, 130]   # small sizes incl. a non-tile-multiple


def build(kernel, mlir, outbin):
    o = outbin + ".o"
    r = subprocess.run([LOWER, mlir, o], capture_output=True, text=True)
    err = open(o + ".err").read() if os.path.exists(o + ".err") else ""
    if r.returncode != 0 or not os.path.exists(o) or os.path.getsize(o) == 0:
        return False, "lower failed: " + err[:300]
    drv = f"{HERE}/kernels/{kernel}/driver.c"
    c = subprocess.run(["clang", "-O3", "-march=native", drv, o, "-o", outbin],
                       capture_output=True, text=True)
    return (c.returncode == 0), c.stderr[:300]


def dump(binp, N):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name
    subprocess.run([binp, str(N), f], check=True, timeout=900)
    a = np.fromfile(f, dtype=np.float64); os.unlink(f); return a


def htime(binp, N, runs=4):
    p = subprocess.run(["hyperfine", "-N", "-w", "1", "-r", str(runs),
                        "--export-json", "/dev/stdout", f"taskset -c 0 {binp} {N}"],
                       capture_output=True, text=True, timeout=1800)
    m = re.search(r'"mean":\s*([0-9.eE+-]+)', p.stdout)
    return float(m.group(1)) if m else None


def grade(kernel, rundir):
    res = {"kernel": kernel, "regimes": {}, "avg_speedup": None, "all_correct": True}
    refbin = f"/tmp/mp_ref_{kernel}"
    ok, err = build(kernel, f"{HERE}/kernels/{kernel}/ref.mlir", refbin)
    if not ok:
        res["error"] = "ref build: " + err; res["all_correct"] = False; return res
    ref_corr = {N: dump(refbin, N) for N in CORR_N}
    ref_t = {}  # ref time per regime size (cached)

    speedups = []
    built = {}  # mlir path -> binary (dedupe identical files)
    for regime in ("small", "medium", "large"):
        lo, hi, testN = SIZES[kernel][regime]
        cand = f"{rundir}/opt_{regime}.mlir"
        src = cand if os.path.exists(cand) else f"{rundir}/opt.mlir"
        rec = {"testN": testN, "src": os.path.basename(src) if os.path.exists(src) else None}
        if not os.path.exists(src):
            rec["status"] = "missing"; res["all_correct"] = False
            res["regimes"][regime] = rec; continue
        if src not in built:
            ob = f"/tmp/mp_opt_{kernel}_{abs(hash(src))%9999}"
            okb, errb = build(kernel, src, ob)
            built[src] = ob if okb else None
            if not okb: rec["build_err"] = errb
        ob = built[src]
        if ob is None:
            rec["status"] = "build_failed"; res["all_correct"] = False
            res["regimes"][regime] = rec; continue
        # correctness at small sizes
        cok = True
        for N in CORR_N:
            oa = dump(ob, N)
            close = (oa.shape == ref_corr[N].shape) and np.allclose(oa, ref_corr[N], rtol=RTOL, atol=ATOL)
            cok = cok and close
        rec["correct"] = bool(cok)
        if not cok:
            rec["status"] = "incorrect"; res["all_correct"] = False
            res["regimes"][regime] = rec; continue
        # timing
        if testN not in ref_t: ref_t[testN] = htime(refbin, testN)
        topt = htime(ob, testN)
        rec["t_ref"], rec["t_opt"] = ref_t[testN], topt
        rec["speedup"] = (ref_t[testN] / topt) if (topt and ref_t[testN]) else None
        if rec["speedup"]: speedups.append(rec["speedup"])
        rec["status"] = "ok"
        res["regimes"][regime] = rec
    if speedups:
        res["avg_speedup"] = sum(speedups) / len(speedups)             # arithmetic mean of 3 regimes
        res["geo_speedup"] = math.exp(sum(map(math.log, speedups)) / len(speedups))
        res["n_regimes_scored"] = len(speedups)
    return res


if __name__ == "__main__":
    out = {}
    for k in KS:
        for g in ("tool", "ctrl"):
            slot = f"{g}-{k}"; rd = f"{HERE}/runs/{slot}"
            r = grade(k, rd); r["group"] = g; out[slot] = r
            json.dump(out, open(f"{HERE}/results.json", "w"), indent=2)
            sp = r.get("avg_speedup")
            print(f"{slot:16s} correct={r['all_correct']} avg_speedup="
                  f"{sp:.3f}" if sp else f"{slot:16s} correct={r['all_correct']} avg_speedup=None",
                  flush=True)
    print("DONE")
