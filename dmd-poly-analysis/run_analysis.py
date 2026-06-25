#!/usr/bin/env python3
"""Batch DMD analysis (scale approximation) over all PolyBench programs in ../autolala.

For each kernel:  tag outer loop -> mlir-extract -> dmd-cli --json (scale).
Saves per-kernel JSON (RI distribution, RD distribution, DMD formula, access counts)
to results/<name>.json and the extracted DSL to dsl/<name>.dsl. Records status/errors.
"""
import json, os, re, subprocess, sys, glob, time

ROOT = "/home/schrodingerzy/Documents/autolala2"
EXTRACT = f"{ROOT}/target/release/mlir-extract"
DMDCLI = f"{ROOT}/target/release/dmd-cli"
HERE = os.path.dirname(os.path.abspath(__file__))
POLY = "/home/schrodingerzy/Documents/autolala/analyzer/misc/polybench"
BLOCK = 64
MAXOPS = "300000000"
TIMEOUT = 360

sys.path.insert(0, HERE)
from tag_outer import tag  # noqa


def analyze(mlir_path):
    name = os.path.basename(mlir_path)[:-5]
    src = open(mlir_path).read().replace("slap.extract", "dmd.extract")
    try:
        src = tag(src)
    except SystemExit as e:
        return name, {"status": "no_loop", "error": str(e)}
    tagged = f"/tmp/poly_{name}.mlir"
    open(tagged, "w").write(src)
    ex = subprocess.run([EXTRACT, tagged, "-a", "dmd.extract"], capture_output=True, text=True)
    if ex.returncode != 0:
        return name, {"status": "extract_fail", "error": ex.stderr.strip()[:400]}
    dsl = ex.stdout
    open(f"{HERE}/dsl/{name}.dsl", "w").write(dsl)
    try:
        an = subprocess.run([DMDCLI, "--block-size", str(BLOCK), "--max-operations", MAXOPS,
                             "--approximation-method", "scale", "--json"],
                            input=dsl, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return name, {"status": "timeout", "dsl_len": len(dsl)}
    if an.returncode != 0:
        return name, {"status": "analyze_fail", "error": an.stderr.strip()[:400], "dsl_len": len(dsl)}
    d = json.loads(an.stdout)
    rec = {"status": "ok", "dsl_len": len(dsl),
           "total": d["total_accesses_plain"], "warm": d["warm_accesses_plain"],
           "compulsory": d["compulsory_accesses_plain"],
           "dmd": d["dmd_formula_plain"],
           "ri": d["ri_distribution"], "rd": d["rd_distribution"],
           "n_ri": len(d["ri_distribution"]), "n_rd": len(d["rd_distribution"]),
           "n_dmd_terms": len(d.get("dmd_terms", []))}
    return name, rec


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    files = []
    if which in ("symbolic", "both"):
        files += sorted(glob.glob(f"{POLY}/symbolic/*.mlir"))
    if which in ("const", "both"):
        files += sorted(glob.glob(f"{POLY}/const/*.mlir"))
    summary = {}
    for f in files:
        t0 = time.time()
        name, rec = analyze(f)
        rec["secs"] = round(time.time() - t0, 1)
        rec["family"] = "symbolic" if "/symbolic/" in f else "const"
        json.dump(rec, open(f"{HERE}/results/{name}.json", "w"), indent=1)
        summary[name] = {k: rec[k] for k in ("status", "secs", "family") if k in rec}
        print(f"{name:28s} {rec['status']:14s} {rec['secs']:6.1f}s "
              f"{'ri='+str(rec.get('n_ri','')) if rec['status']=='ok' else rec.get('error','')[:50]}",
              flush=True)
        json.dump(summary, open(f"{HERE}/summary.json", "w"), indent=1)
    nok = sum(1 for v in summary.values() if v["status"] == "ok")
    print(f"DONE  ok={nok}/{len(files)}")
