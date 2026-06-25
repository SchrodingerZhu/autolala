# atax optimization rationale

Single `opt.mlir` for all three regimes (small/medium/large).

**Key facts.** `llc -O3` does NOT auto-vectorize either pass (strict-fp f64 reductions, no
fast-math) — the ref and any naive variant compile to scalar code. So the levers are
**memory traffic** and **scalar ILP**, not SIMD.

**Transformation (same for every regime): fuse + i-block(8) + unroll-and-jam.**
- *Fusion of the two A-sweeps.* atax reads A twice (T=Ax, then y=AᵀT). I process an
  8-row block: compute T for the 8 rows, then immediately use them for y. Each A row is
  touched by both passes while still hot in L1/L2, so A is effectively streamed from DRAM
  **once** instead of twice (A is N²·8 B — the dominant cost at every size; halving it is
  the main win at medium/large where A ≫ L2/L3).
- *Unroll-and-jam by 8 (TI=8).* Pass1 loads `x[j]` once and feeds 8 independent
  accumulators (x reused 8×; 8 parallel add-chains hide FP latency on the wide X925/A725
  core). Pass2 loads/stores `y[j]` once per 8-row block — y streaming traffic drops 8× vs
  the ref — and issues 8 independent multiplies summed by a **balanced tree** (benign
  reassociation, still within rtol=1e-6) for ILP.
- *Correct for any N.* full 8-row blocks via `(N floordiv 8)*8`; a scalar remainder loop
  (`affine.min`-free tail) handles the last 0–7 rows. Verified all-close for N =
  7,96,130,1024,1025,2047.

**Why faster per regime.** small (A ~8–32 MB ≈ L3): win mostly from ILP + ~2× fewer y/A
passes through L2. medium/large (A ≫ caches): win dominated by reading A once not twice,
i.e. ~2× less DRAM traffic, plus reduced y traffic.

**Predicted average speedup vs ref:** ~1.7–2.1× (closer to 2× at medium/large where the
halved A-traffic dominates, ~1.5× at small where ref already fits more in L3).
