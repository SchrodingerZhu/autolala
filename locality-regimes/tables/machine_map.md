# Machine mapping at n = 2016

Caches in 64-byte lines: 32 KB = 512 lines, 1 MB = 16384 lines, 32 MB = 524288 lines. Compute-bound threshold tau = 1/25 misses per access.

| kernel | model | mr @ 32 KB | mr @ 1 MB | mr @ 32 MB | min cache for mr<=1/25 | coverage |
|---|---|---|---|---|---|---|
| sym_2mm | inf | 0.28 | 0.0308 | 4.6e-05 | 142 KB | 0.9991 |
| sym_3mm | inf | 0.28 | 0.0308 | 4.6e-05 | 142 KB | 0.9991 |
| sym_atax | inf | 0.0466 | 0.0156 | 0 | 48 KB | 0.9995 |
| sym_bicg | inf | 0.0533 | 0.0179 | 0 | 47 KB | 0.9996 |
| sym_cholesky | inf | 4.1e-05 | 4.1e-05 | 0 | 0 KB | 0.9984 |
| sym_convolution9 | inf | 0.00767 | 0.00154 | 0.00154 | 2 KB | 0.9999 |
| sym_correlation | inf | 0.497 | 0.0306 | 5.5e-05 | 414 KB | 0.9995 |
| sym_covariance | inf | 0.497 | 0.0308 | 6.2e-05 | 390 KB | 0.9995 |
| sym_doitgen | inf | 0.281 | 0.0311 | 1.5e-05 | 142 KB | 0.9996 |
| sym_floyd_warshall | inf | 0.0325 | 0.0312 | 1.6e-05 | 32 KB | 0.9995 |
| sym_gemm | inf | 0.0312 | 0.0312 | 3.1e-05 | 32 KB | 0.9994 |
| sym_gemver | inf | 0.116 | 0.0268 | 0 | 124 KB | 0.9996 |
| sym_gesummv | inf | 0.0468 | 0.0313 | 0.0313 | 48 KB | 0.9998 |
| sym_gramschmidt | inf | 0.666 | 0.0203 | 6.6e-05 | 473 KB | 0.9984 |
| sym_lu | inf | 4.1e-05 | 4.1e-05 | 0 | 0 KB | 0.9990 |
| sym_lu_decomp | inf | 0.0314 | 0.0314 | 2.3e-05 | 32 KB | 0.9983 |
| sym_mvt | inf | 0.156 | 0.0312 | 0 | 142 KB | 0.9997 |
| sym_symm | inf | 5.4e-05 | 5.4e-05 | 5.4e-05 | 32 KB | 0.9985 |
| sym_syr2k | inf | 7.2e-05 | 5.1e-05 | 5.1e-05 | 32 KB | 0.9987 |
| sym_syrk | inf | 4.6e-05 | 4.6e-05 | 1.6e-05 | 0 KB | 0.9983 |
| sym_trisolve | inf | 0.0312 | 0.0312 | 0 | 0 KB | 0.9994 |
| sym_trmm | inf | 0.499 | 0.0309 | 4.6e-05 | 472 KB | 0.9986 |