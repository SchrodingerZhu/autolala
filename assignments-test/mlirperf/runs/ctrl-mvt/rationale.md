# mvt optimization rationale

Single `opt.mlir` serves all three regimes (small/medium/large); the dominant win is
regime-independent, so one tuned structure suffices.

- **Pass 1** `x1[i] += A[i,j]*y1[j]`: already row-major in A. Tiled the `j` axis (step 512,
  `jj`-outer / `i` / inner `j`) so a 4 KB block of `y1` stays L1-resident across the full
  `i` sweep instead of re-streaming all N of `y1` per row; `x1[i]` accumulates partial sums
  per tile. A is still read once, contiguously.
- **Pass 2** `x2[i] += A[j,i]*y2[j]`: the reference reads A **transposed** (`A[j,i]`) with
  stride N — one useful double per 64 B line, defeating the prefetcher. **Interchanged** so
  `i` is the inner loop, making `A[j,i]` contiguous row-major (8 useful doubles/line, ~8x
  fewer line fetches + clean prefetch). Tiled `i` (step 512, `ii`-outer / `j` / inner `i`)
  so a 4 KB `x2` block stays resident across the whole `j` reduction (its reuse axis);
  `y2[j]` is invariant in the inner loop (register).
- Tile size 512 keeps the secondary vectors (`y1`/`x2` blocks) in L1 at every N; A is the
  irreducible N² memory stream in both passes, so tiling only removes secondary traffic.
  Correctness via `affine.min` bounds holds for any N (verified N=96,130 allclose).
- **Predicted average speedup ≈ 3–5x**, driven almost entirely by the pass-2 interchange
  eliminating strided column reads of A; pass-1 tiling adds a smaller L1-residency gain.
