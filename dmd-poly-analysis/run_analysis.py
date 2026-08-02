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
# Block size is in ELEMENTS, not bytes. A 64-byte cache line holds 8 f64 (double)
# or 16 f32 (float). PolyBench is double precision, so 8 elements == one 64-byte
# line. (This matches the legacy autolala run, which used --block-size=8.)
BLOCK = 8
MAXOPS = "2000000000"
TIMEOUT = 1500

sys.path.insert(0, HERE)
from tag_outer import tag  # noqa


def run_dmd_method(dsl, infinite_repeat, method):
    """Run dmd-cli on one DSL with the given counting method.
    Returns (record, error). record is None on failure."""
    args = [DMDCLI, "--block-size", str(BLOCK), "--max-operations", MAXOPS,
            "--approximation-method", method, "--json"]
    if infinite_repeat:
        args.append("--infinite-repeat")
    try:
        an = subprocess.run(args, input=dsl, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if an.returncode != 0:
        return None, an.stderr.strip()[:400]
    d = json.loads(an.stdout)
    return {
        "method": method,
        "total": d["total_accesses_plain"], "warm": d["warm_accesses_plain"],
        "compulsory": d["compulsory_accesses_plain"], "dmd": d["dmd_formula_plain"],
        "ri": d["ri_distribution"], "rd": d["rd_distribution"],
        "n_ri": len(d["ri_distribution"]), "n_rd": len(d["rd_distribution"]),
        "n_dmd_terms": len(d.get("dmd_terms", [])),
    }, None


def run_dmd(dsl, infinite_repeat):
    """Exact counting first (masses conserve to the integer); fall back to the
    scale approximation when exact exceeds the operation budget or times out.
    The record carries `method` so downstream tables can gate on it."""
    rec, err = run_dmd_method(dsl, infinite_repeat, "exact")
    if rec is not None:
        return rec, None
    rec2, err2 = run_dmd_method(dsl, infinite_repeat, "scale")
    if rec2 is not None:
        rec2["exact_error"] = err
        return rec2, None
    return None, f"exact: {err}; scale: {err2}"


def analyze(mlir_path):
    """Extract one kernel, then analyze it under BOTH the single-shot and the
    infinite-repeat models so the report can compare them directly."""
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

    single, err = run_dmd(dsl, infinite_repeat=False)
    if single is None:
        status = "timeout" if err == "timeout" else "analyze_fail"
        return name, {"status": status, "error": err, "dsl_len": len(dsl)}
    inf, inf_err = run_dmd(dsl, infinite_repeat=True)  # may fail independently

    rec = {"status": "ok", "dsl_len": len(dsl), "single": single, "inf": inf,
           "inf_error": inf_err,
           # top-level mirrors of the single-shot fields for backward compat
           **single}
    return name, rec


# Kernels that are known-pathological for the Barvinok counter (deeply nested,
# heavily parametric) and not worth the analysis budget. fdtd-apml times out.
SKIP = {"const_fdtd-apml"}

def rebuild_summary(files):
    """Rebuild summary.json from the per-kernel result files."""
    summary = {}
    for f in files:
        name = os.path.basename(f)[:-5]
        path = f"{HERE}/results/{name}.json"
        if not os.path.exists(path):
            continue
        rec = json.load(open(path))
        summary[name] = {k: rec[k] for k in ("status", "secs", "family") if k in rec}
        if rec.get("status") == "ok":
            summary[name]["inf"] = bool(rec.get("inf"))
            summary[name]["method"] = rec.get("single", {}).get("method")
    json.dump(summary, open(f"{HERE}/summary.json", "w"), indent=1)
    return summary


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    resume = "--resume" in sys.argv
    only = None
    for arg in sys.argv[2:]:
        if arg.startswith("--only="):
            only = set(arg[len("--only="):].split(","))
    no_summary = "--no-summary" in sys.argv
    files = []
    if which in ("symbolic", "both"):
        files += sorted(glob.glob(f"{POLY}/symbolic/*.mlir"))
    if which in ("const", "both"):
        files += sorted(glob.glob(f"{POLY}/const/*.mlir"))
    for f in files:
        name0 = os.path.basename(f)[:-5]
        if name0 in SKIP:
            if only is None or name0 in only:
                print(f"{name0:28s} skipped (known bad case)", flush=True)
            continue
        if only is not None and name0 not in only:
            continue
        if resume and os.path.exists(f"{HERE}/results/{name0}.json"):
            existing = json.load(open(f"{HERE}/results/{name0}.json"))
            if existing.get("status") == "ok":
                print(f"{name0:28s} resume: already ok", flush=True)
                continue
        t0 = time.time()
        name, rec = analyze(f)
        rec["secs"] = round(time.time() - t0, 1)
        rec["family"] = "symbolic" if "/symbolic/" in f else "const"
        json.dump(rec, open(f"{HERE}/results/{name}.json", "w"), indent=1)
        if rec["status"] == "ok":
            m = rec["single"].get("method", "?")
            infn = "inf_ok" if rec.get("inf") else f"inf_fail({(rec.get('inf_error') or '')[:20]})"
            note = f"[{m}] single n_rd={rec['single']['n_rd']} / {infn}"
        else:
            note = rec.get("error", "")[:50]
        print(f"{name:28s} {rec['status']:14s} {rec['secs']:6.1f}s {note}", flush=True)
    if not no_summary:
        summary = rebuild_summary(files)
        nok = sum(1 for v in summary.values() if v["status"] == "ok")
        print(f"DONE  ok={nok}/{len(files)}")
