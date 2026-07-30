# Scalar DMD vs miss ratio, n = 2016

| kernel | DMD/access | growth exp | mr @ 32KB | mr @ 1MB |
|---|---|---|---|---|
| sym_gramschmidt | 55 | 0.65 | 0.666 | 0.0203 |
| sym_trmm | 51.5 | 0.73 | 0.499 | 0.0309 |
| sym_covariance | 51.4 | 0.74 | 0.497 | 0.0308 |
| sym_correlation | 51.2 | 0.74 | 0.497 | 0.0306 |
| sym_doitgen | 35 | 0.83 | 0.281 | 0.0311 |
| sym_3mm | 34.3 | 0.84 | 0.28 | 0.0308 |
| sym_2mm | 34.3 | 0.84 | 0.28 | 0.0308 |
| sym_gesummv | 33.8 | 0.95 | 0.0468 | 0.0313 |
| sym_mvt | 29.4 | 0.88 | 0.156 | 0.0312 |
| sym_gemver | 24.5 | 0.88 | 0.116 | 0.0268 |
| sym_lu_decomp | 24.5 | 0.94 | 0.0314 | 0.0314 |
| sym_floyd_warshall | 24.4 | 0.94 | 0.0325 | 0.0312 |
| sym_gemm | 24.4 | 0.95 | 0.0312 | 0.0312 |
| sym_trisolve | 17.2 | 0.94 | 0.0312 | 0.0312 |
| sym_bicg | 15.5 | 0.89 | 0.0533 | 0.0179 |
| sym_atax | 13.8 | 0.88 | 0.0466 | 0.0156 |
| sym_convolution9 | 4.24 | 0.48 | 0.00767 | 0.00154 |
| sym_symm | 2.99 | 0.21 | 5.4e-05 | 5.4e-05 |
| sym_syr2k | 2.81 | 0.19 | 7.2e-05 | 5.1e-05 |
| sym_syrk | 1.9 | 0.15 | 4.6e-05 | 4.6e-05 |
| sym_lu | 1.8 | 0.16 | 4.1e-05 | 4.1e-05 |
| sym_cholesky | 1.49 | 0.00 | 4.1e-05 | 4.1e-05 |

## Order inversions (>=1.1x higher DMD, >=1.5x lower miss ratio at 32 KB, no worse at 1 MB)

- sym_gesummv (DMD 33.8) vs sym_mvt (DMD 29.4): mr@32KB 0.0468 vs 0.156, mr@1MB 0.0313 vs 0.0312

## Near-equal DMD (within 5%), miss ratio apart >=2x at 32 KB

- sym_doitgen (DMD 35, mr@32KB 0.281) vs sym_gesummv (DMD 33.8, mr@32KB 0.0468): 6.0x apart
- sym_3mm (DMD 34.3, mr@32KB 0.28) vs sym_gesummv (DMD 33.8, mr@32KB 0.0468): 6.0x apart
- sym_2mm (DMD 34.3, mr@32KB 0.28) vs sym_gesummv (DMD 33.8, mr@32KB 0.0468): 6.0x apart
- sym_gemver (DMD 24.5, mr@32KB 0.116) vs sym_gemm (DMD 24.4, mr@32KB 0.0312): 3.7x apart
- sym_gemver (DMD 24.5, mr@32KB 0.116) vs sym_lu_decomp (DMD 24.5, mr@32KB 0.0314): 3.7x apart
- sym_gemver (DMD 24.5, mr@32KB 0.116) vs sym_floyd_warshall (DMD 24.4, mr@32KB 0.0325): 3.6x apart