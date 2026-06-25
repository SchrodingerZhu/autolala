# GEMM optimization rationale

Math: C = 0.9*C + A*B. The 0.9 scaling is kept as a separate cache-friendly
prologue (unit-stride sweep over C); the multiply-accumulate is the hot nest.
All variants interchange to inner-j so B[k,j] and C[i,j] are unit-stride
(llc -O3 vectorizes), while A[i,k] is loop-invariant in j and reused N times.

- **small (192-384):** plain i,k,j interchange, no tiling. Working set fits L2,
  so tiling only adds loop/min overhead; the interchange alone gives unit-stride
  inner-j and full A reuse. ~3-4x over ref (ref's k-inner reduces C in a scalar
  serial chain and strides B by N).
- **medium (512-1024):** tile i_t=32, j_t=256, k_t=256, order i_t,j_t,k_t,i,k,j.
  The k_t x j_t B-panel and the C strip stay resident in L2 across the i sweep,
  cutting B/C memory traffic by the i-tile factor. ~5-7x.
- **large (1152-1536):** same scheme, j_t shrunk to 128 so the 256x128 B-panel
  (256KB) plus C strip fit L2 even as N grows, avoiding L2 capacity misses that
  hurt ref badly here. ~6-8x.

All bounds use affine.min upper bounds, so correctness holds for any N (verified
at N=96 and N=130, non-multiples of the tiles). Predicted average speedup ~5-6x.
