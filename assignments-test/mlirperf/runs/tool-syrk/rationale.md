# syrk (C = 0.9*C + A*A^T, lower triangle) — optimization rationale

Single `opt.mlir` for all three regimes (small/medium/large).

- **Loop order kept i-k-j.** dmd's TREND analysis showed i-k-j is already the data-movement
  optimum (N^3 order): inner `j` keeps `A[i,k]` loop-invariant and streams `C[i,j]`
  unit-stride; the alternatives i-j-k (~N^3.5) and k-i-j (~N^4 under line modeling) are
  asymptotically worse. So no interchange.
- **Unroll-and-jam on i by 4 (register-tiling).** This was dmd's highest-leverage,
  model-backed transform: jamming 4 rows lets one column-strided `A[j,k]` load (the
  expensive, stride-N access) serve all 4 rows, dropping total inner traffic
  3/2 -> 9/8 ·N^3 toward the N^3 floor and cutting A reloads ~4x. The triangular j<=i bound
  is handled by running the jammed inner loop only over the rectangular range [0, ii]
  (valid for every row in the block) plus a tiny per-row diagonal cleanup.
- **Correctness for any N:** partial last blocks (N not a multiple of 4) and the per-row
  diagonal band are guarded by `affine.if` sets on the symbol N; verified all-close vs ref
  for N=1,2,3,5,7,96,130,193,257,383,511,1153,1663 (covers all regime boundaries + edges).
- **Why no tiling:** dmd reported k-/i-tiling give no asymptotic benefit on i-k-j, and the
  residual N^3 is compulsory A traffic that tiling can't remove; square i×j blocking is the
  only theoretical help at large N but its min(jj+T,i+1) bound is non-affine — not worth the
  risk over the solid UJ=4 win.
- **Predicted avg speedup vs ref:** ~1.3–1.7x (largest at medium/large where A-reload
  amortization matters most; small N already mostly cache-resident).
