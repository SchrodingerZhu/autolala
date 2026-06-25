import json, os, time
from eval import evaluate
KS=["matmul","gemm","2mm","3mm","mvt","atax","bicg","gemver","gesummv","syrk","doitgen","covariance"]
out={}
for k in KS:
    for g in ("tool","ctrl"):
        slot=f"{g}-{k}"
        src=f"runs/{slot}/opt.c"
        t0=time.time()
        if not os.path.exists(src):
            out[slot]={"kernel":k,"group":g,"build_ok":False,"error":"no opt.c"}
        else:
            r=evaluate(k,src); r["group"]=g; out[slot]=r
        out[slot]["eval_secs"]=round(time.time()-t0,1)
        json.dump(out,open("perf_results.json","w"),indent=2)
        rr=out[slot]
        print(f"{slot:20s} build={rr.get('build_ok')} correct={rr.get('correct')} speedup={rr.get('speedup')}", flush=True)
print("DONE")
