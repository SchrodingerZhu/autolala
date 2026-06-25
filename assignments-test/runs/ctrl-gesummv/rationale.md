# gesummv locality optimization

## Kernel
`gesummv` computes, over an N×N domain:
- `tmp = A·x`
- `y   = B·x`
- `y   = alpha·tmp + beta·y`

Source structure: outer `i`, inner reduction `j`, with `tmp[i]`/`y[i]`
initialized before the `j` loop and finalized after it.

## Transformation applied
1. **Loop fission (distribution).** Split the per-`i` body into three
   independent phases:
   - Phase 1 — initialize `tmp[i]=0`, `y[i]=0` for all `i`.
   - Phase 2 — the two matrix-vector accumulations.
   - Phase 3 — finalize `y[i] = alpha·tmp[i] + beta·y[i]`.

   This is legal because the init writes every `tmp[i]`/`y[i]` before any
   accumulation reads it, and the finalize reads each `tmp[i]`/`y[i]` only
   after all accumulations into it complete. No cross-`i` dependence exists.

2. **Reduction-loop tiling + interchange on Phase 2.** Strip-mine `j` by 32
   and hoist the tile loop `jj` outside `i`, giving order `jj → i → j`.

## Why this cuts data movement
The reused operand is the input vector `x`. In the original `i→j` order,
`x[0..N)` is re-streamed from memory once per `i` row → **~N² loads of `x`**
(its reuse distance is O(N), larger than cache for big N).

After tiling, the working set of an inner `jj`-tile is just `x[jj..jj+32)`
(plus the corresponding A/B columns). That 32-element `x` block is loaded
**once** and reused across the entire `i` sweep, so total memory traffic for
`x` drops from O(N²) to O(N) — its reuse distance becomes O(32), comfortably
cache-resident.

Cost paid: `tmp[i]` and `y[i]` are now revisited once per `j`-tile instead of
held in registers across the whole reduction, adding O(N²/32) traffic each —
negligible next to the O(N²) reads of A, B that are intrinsically once-touched
(streamed) and cannot be reduced.

## Predicted improvement
- Original capacity traffic ≈ `A(N²) + B(N²) + x(N²)` ≈ **3N²**.
- Tiled traffic ≈ `A(N²) + B(N²) + x(N) + tmp,y(2N²/32)` ≈ **~2.06N²**.

Predicted data-movement reduction factor ≈ **1.4–1.5×** (the `x`-streaming
term is essentially eliminated; A and B remain the irreducible floor).
A larger tile would shrink the tmp/y term further but risks evicting the
x/A/B-column working set; 32 is a safe cache-line-friendly choice.
