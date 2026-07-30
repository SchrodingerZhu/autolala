# Anchors against the paper

## Table 1 (naive matmul, element granularity, infinite repeat)

| level | rd scale | c(ri) (avg) | portion | miss after | paper row |
|---|---|---|---|---|---|
| 1 | 3 | 3 | 1/3 - 1/(3*n) | 2/3 + 1/(3*n) | c=3, P=1/3-1/(3n), m=2/3+1/(3n) |
| 2 | 2 n^1 | 2*n + 2 - 1/n | 1/3 - 1/(3*n) | 1/3 + 2/(3*n) | c=2n+2-1/n, P=1/3-1/(3n), m=1/3+2/(3n) |
| 3 | 1 n^2 | n^2 + 3*n - 1/n | 1/3 | 2/(3*n) | c=n^2+3n-1/n, P=1/3, m=2/(3n) |
| 4 | 3 n^2 | 3*n^2 - n/2 + 1/2 + 1/(2*n) | 2/(3*n) | 0 | c=3n^2-ish (rows 4*,5* merged), P=2/(3n), m=0 |

## Table 6 (block 8, min-max co-scaling)

| level | rd scale | c (avg, lines) | portion | miss after |
|---|---|---|---|---|
| 1 | 21/8 | 21/8 | 7/24 | 17/24 - 61/(64*n) |
| 2 | 175/64 | 175/64 | 49/(192*n) | 17/24 - 29/(24*n) |
| 3 | 3 | 3 | 1/3 - 1/(3*n) | 3/8 - 7/(8*n) |
| 4 | 27/4 | 27/4 | 7/(24*n) | 3/8 - 7/(6*n) |
| 5 | 31/4 | 31/4 | 1/(24*n) | 3/8 - 29/(24*n) |
| 6 | 63/64 n^1 | 63*n/64 + 7/8 + 49/(512*n) + 735/(4096*n^2) | 7/24 - 161/(192*n) | 1/12 - 71/(192*n) |
| 7 | 9/8 n^1 | 9*n/8 - 13/4 - 1457/(64*n) - 97567/(512*n^2) | 1/24 - 79/(192*n) | 1/24 + 1/(24*n) |
| 8 | 143/64 n^1 | 143*n/64 + 185/64 | 0 | 1/24 + 1/(24*n) |
| 9 | 151/64 n^1 | 151*n/64 + 29/8 - 399/(512*n) - 5985/(4096*n | 1/(3*n) | 1/24 - 7/(24*n) |
| 10 | 1/8 n^2 | n^2/8 + 87*n/64 - 7 - 609/(8*n) - 4831/(8*n^ | 1/24 - 3/(8*n) | 1/(12*n) |
| 11 | 3/8 n^2 | 3*n^2/8 - n/2 - 3/8 - 16/n | 1/(12*n) | 0 |

Paper Table 6 boundaries: 1, 3, 4, 9n/8-9, 9n/8+2, n^2/8+3n/8-2 lines; miss plateaus 3/4, 1/2+1/(32n), 9/32+1/(32n), 1/4+1/(16n), 1/32+1/(16n), 3/(32n). The boundary structure (constant, 9n/8, n^2/8; plateaus constant, constant, Θ(1/n), 0) reproduces; the small constant offsets are consistent with the paper's array padding (Sec. 4.1), which this run does not apply.

## RI Sum Invariance: sum(ri x P(ri)) vs data size

| kernel | n | sum ri*P(ri) | data size D | relative gap |
|---|---|---|---|---|
| matmul3.b1 | 8400 | 2.11672e+08 | 211680000 | 3.97e-05 |
| matmul3 | 8400 | 2.64348e+07 | 26460000 | 9.52e-04 |
| sym_gemm | 8400 | 1.76326e+07 | 26460000 | 3.34e-01 |

The identity holds exactly on the unfiltered distribution (paper, Sec. 2.5); the residual here is the mass removed by the analyzer's degenerate-region filtering.