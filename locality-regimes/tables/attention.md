# Cache laws for the attention family

All numbers from the symbolic tables (infinite repeat, scale approximation, 64-byte lines of 8 f64); traffic = misses x 64 B. n = sequence length, d = head dimension.

## 0. Conservation self-check (trust gate)

| kernel | coverage @ (n=8192,d=64) | @ (n=8192,d=128) |
|---|---|---|
| att_dense | 0.9897 | 0.9948 |
| att_dense_causal | 0.9817 | 0.9908 |
| att_linear | 0.9908 | 0.9954 |
| att_linear_chunk16 | 0.9908 | 0.9954 |
| att_linear_chunk64 | 0.9908 | 0.9954 |
| att_linear_chunk256 | 0.9909 | 0.9954 |

## 1. Linear attention: context-length-free cache behavior, and the head-dimension cliff

At (n=8192, d=64): reuse distances whose formulas do not mention n carry 98.97% of accesses; n-dependent (whole-footprint / imaginary) distances carry 0.11%. Largest n-free distance: 540 lines (33.8 KB).

The n-free distances are the state reuse: the knee sits at the d x d state plus one row set. Miss ratio vs d (same at every n; verified n = 2048 ... 65536):

| d | mr @ 32 KB | mr @ 1 MB | per-token traffic @ 32 KB | @ 1 MB |
|---|---|---|---|---|
| 32 | 0.0022 | 0.0022 | 1 KB | 1 KB |
| 48 | 0.0015 | 0.0015 | 1.5 KB | 1.5 KB |
| 64 | 0.0359 | 0.0011 | 64.3 KB | 2 KB |
| 90 | 0.0358 | 0.0008 | 127 KB | 2.81 KB |
| 128 | 0.0358 | 0.0006 | 257 KB | 4 KB |
| 181 | 0.0358 | 0.0004 | 513 KB | 5.66 KB |
| 256 | 0.0358 | 0.0003 | 1 MB | 8 KB |

n-independence check (d=128, 32 KB): mr = 0.03580, 0.03580, 0.03580 at n = 2048, 8192, 65536.

## 2. Dense (softmax) attention: sequence-length cliffs

Per-token DRAM traffic vs n (d = 64):

| n | @ 32 KB | @ 1 MB | @ 32 MB |
|---|---|---|---|
| 512 | 531 KB | 21.9 KB | 0 B |
| 1024 | 1.04 MB | 41.9 KB | 0 B |
| 2048 | 2.08 MB | 2.08 MB | 81.9 KB |
| 4096 | 4.16 MB | 4.16 MB | 162 KB |
| 8192 | 8.31 MB | 8.31 MB | 322 KB |
| 16384 | 16.6 MB | 16.6 MB | 642 KB |
| 32768 | 33.2 MB | 33.2 MB | 1.25 MB |

Same at d = 128:

| n | @ 32 KB | @ 1 MB | @ 32 MB |
|---|---|---|---|
| 512 | 1.02 MB | 23.9 KB | 0 B |
| 1024 | 2.04 MB | 2.04 MB | 0 B |
| 2048 | 4.08 MB | 4.08 MB | 83.9 KB |
| 4096 | 8.15 MB | 8.15 MB | 164 KB |
| 8192 | 16.3 MB | 16.3 MB | 324 KB |
| 16384 | 32.6 MB | 32.6 MB | 644 KB |
| 32768 | 65.2 MB | 65.2 MB | 65.2 MB |

Causal variant conservation at (8192, 64): 0.9817 (see trust gate; triangular kernels are where the scale approximation is weakest).

## 3. Per-token traffic, dense vs linear

| (n, d) | dense @ 1 MB | linear @ 1 MB | ratio |
|---|---|---|---|
| (1024, 64) | 41.9 KB | 2 KB | 21x |
| (4096, 64) | 4.16 MB | 2 KB | 2128x |
| (16384, 64) | 16.6 MB | 2 KB | 8512x |
| (65536, 64) | 66.5 MB | 2 KB | 34048x |
| (1024, 128) | 2.04 MB | 4 KB | 522x |
| (4096, 128) | 8.15 MB | 4 KB | 2088x |
| (16384, 128) | 32.6 MB | 4 KB | 8352x |
| (65536, 128) | 130 MB | 4 KB | 33408x |

## 4. Chunked linear attention: the chunk-length law

Per-token traffic (n = 8192):

| variant | d=64 @32KB | d=64 @1MB | d=128 @32KB | d=128 @1MB | d=256 @32KB | d=256 @1MB |
|---|---|---|---|---|---|---|
| recurrent (L=1) | 64.3 KB | 2 KB | 257 KB | 4 KB | 1 MB | 8 KB |
| chunk 16 | 66.4 KB | 2 KB | 261 KB | 4 KB | 1.07 MB | 8 KB |
| chunk 64 | 130 KB | 2 KB | 388 KB | 4 KB | 1.26 MB | 20.4 KB |
| chunk 256 | 325 KB | 5.12 KB | 775 KB | 11.5 KB | 2.01 MB | 22 KB |