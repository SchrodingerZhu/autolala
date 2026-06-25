#!/usr/bin/env python3
"""Cachegrind cache-miss scaling sweep. Confirms the DMD reuse-distance prediction:
L1 (D1) and last-level (LLd) miss RATE should grow with N for high-gap kernels
(reuse distance grows), and stay flat for gap~0 kernels (reuse distance bounded)."""
import subprocess, re, json, sys

def cg(N, kern):
    p = subprocess.run(["valgrind", "--tool=cachegrind", "--cache-sim=yes",
                        "--D1=32768,8,64", "--LL=2097152,16,64",
                        "--cachegrind-out-file=/dev/null", "./k", str(N), kern],
                       capture_output=True, text=True)
    out = p.stderr
    def num(label):
        m = re.search(rf"{label}:\s*([0-9,]+)", out)
        return int(m.group(1).replace(",", "")) if m else None
    return {"Dref": num(r"D\s+refs"), "D1m": num(r"D1\s+misses"), "LLd": num(r"LLd misses")}

PLAN = {
 "mm_naive":  [128,256,384,512], "mm_tiled": [128,256,384,512],
 "syrk_naive":[128,256,384,512],
 "mvt_naive": [512,1024,2048],   "mvt_interch":[512,1024,2048],
}
res = {}
print(f"{'kernel':12s} {'N':>5s} {'Drefs':>13s} {'D1miss%':>8s} {'LLdmiss%':>9s}")
for k, ns in PLAN.items():
    res[k] = []
    for N in ns:
        c = cg(N, k)
        d1 = 100*c["D1m"]/c["Dref"]; ll = 100*c["LLd"]/c["Dref"]
        res[k].append({"N": N, **c, "D1_rate": d1, "LL_rate": ll})
        print(f"{k:12s} {N:5d} {c['Dref']:13,d} {d1:8.2f} {ll:9.3f}", flush=True)
json.dump(res, open("cg.json","w"), indent=1)
print("CGDONE")
