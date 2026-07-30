#!/usr/bin/env python3
"""Regenerate analyzer output for the symbolic PolyBench kernels, fresh.

For every dsl/sym_*.dsl (plus dsl/matmul3.dsl at block size 8), run
dmd-cli under both execution models:

  single : one execution; first-touch accesses are compulsory misses
  inf    : --infinite-repeat; first-touch accesses become imaginary reuses

Output: data/<kernel>.<model>.json  (raw dmd-cli JSON, untouched)
        data/<kernel>.<model>.err   (stderr, on failure)

Block size is 8 elements = one 64-byte line of f64, matching PolyBench
double precision. Everything downstream reads only these files.
"""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "target", "release", "dmd-cli")
BLOCK = "8"
MAXOPS = "300000000"
TIMEOUT = 360


def run_one(job):
    name, model = job
    out = os.path.join(HERE, "data", f"{name}.{model}.json")
    err = os.path.join(HERE, "data", f"{name}.{model}.err")
    if os.path.exists(out):
        return name, model, "cached"
    args = [CLI, "--block-size", BLOCK, "--max-operations", MAXOPS,
            "--approximation-method", "scale", "--json",
            "-i", os.path.join(HERE, "dsl", f"{name}.dsl")]
    if model == "inf":
        args.append("--infinite-repeat")
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        open(err, "w").write("timeout\n")
        return name, model, "timeout"
    if r.returncode != 0:
        open(err, "w").write(r.stderr)
        return name, model, "fail"
    open(out, "w").write(r.stdout)
    return name, model, "ok"


def main():
    kernels = sorted(f[:-4] for f in os.listdir(os.path.join(HERE, "dsl"))
                     if f.startswith("sym_") and f.endswith(".dsl"))
    kernels.append("matmul3")
    jobs = [(k, m) for k in kernels for m in ("single", "inf")]
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            name, model, status = f.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {name}.{model}: {status}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
