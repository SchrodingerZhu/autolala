# Naive vs tiled matmul (block 8, infinite repeat), N = 2048

Cache sizes in 64-byte lines: 4 KB = 64, 8 KB = 128, 32 KB = 512, 512 KB = 8192, 16 MB = 262144, 64 MB = 1048576.

| variant | 4 KB | 8 KB | 32 KB | 512 KB | 16 MB | 64 MB |
|---|---|---|---|---|---|---|
| naive | 0.374 | 0.374 | 0.374 | 0.0415 | 0.0415 | 4.1e-05 |
| tile 8 | 0.0104 | 0.0104 | 0.0104 | 0.00525 | 0.00525 | 4.1e-05 |
| tile 16 | 0.00781 | 0.00749 | 0.00523 | 0.00523 | 0.00264 | 4.1e-05 |
| tile 32 | 0.0443 | 0.0443 | 0.00353 | 0.00262 | 0.00134 | 4.1e-05 |

## Pointwise gain over naive (traffic ratio)

| variant | 4 KB | 8 KB | 32 KB | 512 KB | 16 MB | 64 MB |
|---|---|---|---|---|---|---|
| tile 8 | 35.9x | 35.9x | 35.9x | 7.9x | 7.9x | 1.0x |
| tile 16 | 47.9x | 50.0x | 71.6x | 7.9x | 15.7x | 1.0x |
| tile 32 | 8.5x | 8.5x | 106.2x | 15.8x | 30.9x | 1.0x |

## Loop order (same 3-access body, ijk vs ikj)

| variant | 4 KB | 8 KB | 32 KB | 512 KB | 16 MB | 64 MB |
|---|---|---|---|---|---|---|
| ijk (k inner) | 0.374 | 0.374 | 0.374 | 0.0415 | 0.0415 | 4.1e-05 |
| ikj (j inner) | 0.0831 | 0.0831 | 0.0831 | 0.0417 | 0.0417 | 4.1e-05 |