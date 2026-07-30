# gemm under p-way slicing, n = 2016 (block 8, infinite repeat)

Per-worker miss ratio from parameter substitution; aggregate traffic = p x worker accesses x worker miss ratio, private caches of 512 lines (32 KB) and 16384 lines (1 MB).


## i-slice (rows of the output)

Boundary reading: the re-swept matrix B (k x j) stays whole in every worker.

| p | worker accesses | worker mr @ 32 KB | aggregate traffic @ 32 KB (lines) | worker mr @ 1 MB | aggregate @ 1 MB |
|---|---|---|---|---|---|
| 1 | 3.28e+10 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 2 | 1.64e+10 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 4 | 8.2e+09 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 8 | 4.1e+09 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 16 | 2.05e+09 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 32 | 1.02e+09 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 63 | 5.2e+08 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 126 | 2.6e+08 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 252 | 1.3e+08 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 504 | 6.5e+07 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 1008 | 3.25e+07 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |

## j-slice (columns of the output)

Boundary reading: each worker re-sweeps only an n x n/p slice of B.

| p | worker accesses | worker mr @ 32 KB | aggregate traffic @ 32 KB (lines) | worker mr @ 1 MB | aggregate @ 1 MB |
|---|---|---|---|---|---|
| 1 | 3.28e+10 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 2 | 1.64e+10 | 0.0311 | 1.02e+09 | 0.0311 | 1.02e+09 |
| 4 | 8.2e+09 | 0.031 | 1.02e+09 | 0.031 | 1.02e+09 |
| 8 | 4.1e+09 | 0.0309 | 1.01e+09 | 0.0309 | 1.01e+09 |
| 16 | 2.05e+09 | 0.0305 | 9.99e+08 | 0.0305 | 9.99e+08 |
| 32 | 1.02e+09 | 0.0297 | 9.75e+08 | 0.000511 | 1.68e+07 |
| 63 | 5.2e+08 | 0.0283 | 9.28e+08 | 0.000992 | 3.25e+07 |
| 126 | 2.6e+08 | 0.0195 | 6.4e+08 | 0.00197 | 6.45e+07 |
| 252 | 1.3e+08 | 0.0351 | 1.15e+09 | 0.00392 | 1.29e+08 |
| 504 | 6.5e+07 | 0.0703 | 2.3e+09 | 0.00784 | 2.57e+08 |
| 1008 | 3.25e+07 | 0.141 | 4.61e+09 | 0.0157 | 5.14e+08 |

## k-slice (reduction; needs partial-sum combination)

Boundary reading: each worker re-sweeps an n/p x n slice of B.

| p | worker accesses | worker mr @ 32 KB | aggregate traffic @ 32 KB (lines) | worker mr @ 1 MB | aggregate @ 1 MB |
|---|---|---|---|---|---|
| 1 | 3.28e+10 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 2 | 1.64e+10 | 0.0312 | 1.02e+09 | 0.0312 | 1.02e+09 |
| 4 | 8.2e+09 | 0.0311 | 1.02e+09 | 0.0311 | 1.02e+09 |
| 8 | 4.1e+09 | 0.031 | 1.02e+09 | 0.031 | 1.02e+09 |
| 16 | 2.06e+09 | 0.0309 | 1.02e+09 | 0.0309 | 1.02e+09 |
| 32 | 1.03e+09 | 0.0306 | 1.01e+09 | 0.0305 | 1.01e+09 |
| 63 | 5.28e+08 | 0.03 | 9.99e+08 | 0.000977 | 3.25e+07 |
| 126 | 2.68e+08 | 0.0288 | 9.75e+08 | 0.00191 | 6.45e+07 |
| 252 | 1.38e+08 | 0.0271 | 9.45e+08 | 0.00369 | 1.29e+08 |
| 504 | 7.32e+07 | 0.0304 | 1.12e+09 | 0.00697 | 2.57e+08 |
| 1008 | 4.06e+07 | 0.0359 | 1.47e+09 | 0.0125 | 5.14e+08 |