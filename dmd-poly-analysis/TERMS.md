---
title: "Principal contribution terms of the PolyBench kernels"
subtitle: "The full reuse-distance term list per kernel: distances, populations, orders, and what each term contributes to misses and data movement"
date: "2026-07-30"
geometry: margin=0.9in
fontsize: 9pt
---

# How to read this document

The analyzer bins every warm access (every reuse) of a kernel by its **reuse
distance** (RD): the number of distinct cache lines touched in the reuse
window, at block granularity 8 doubles = one 64-byte line. Arrays follow the
padded layout of the paper's evaluation: every row starts on a fresh line
(only the innermost subscript is blocked). Each bin is a **principal
contribution term** — a pair

$$\big(V(n),\; M(n)\big)$$

of a **distance** $V(n)$ (cache lines) and a **population** $M(n)$ (number
of accesses), both exact functions of the problem size $n$ (all program
parameters bound to $n$; convolution fixes its filter extent at 9). A term
states, completely and exactly:

* **Miss contribution.** The term's $M(n)$ accesses hit if the cache holds
  at least $V(n)$ lines and miss otherwise: the term contributes $M(n)$
  misses on the cache range $C < V(n)$ and none above. Its *boundary* is
  $C^{*} = V(n)$ lines ($64\,V(n)$ bytes); its *portion* is $M(n)/A(n)$ of
  all accesses. The kernel's entire miss-ratio curve is the sum of these
  step contributions — nothing else.
* **Data-movement (DMD) contribution.** Under the square-root cost model
  the term contributes $M(n)\sqrt{V(n)}$, of order
  $n^{\deg M + \deg V/2}$; the kernel's DMD spectrum is the ordered list
  of these orders with coefficients.

Two kinds appear. A **level** has one distance for its whole population
(split per residue class of an iterator mod 8 where the line boundary makes
classes differ — classes with identical polynomials are merged). A **ramp**
(family) spans a distance *range* $[V_{\min}(n), V_{\max}(n)]$ that grows
with a loop iterator (triangular kernels): it contributes misses that taper
off as the cache grows through the range; its population and range bounds
are exact polynomials, and its order is $\deg M + \deg V_{\max}/2$.

*Why a term with vanishing portion still matters.* A population
$\Theta(n^2)$ term in a $\Theta(n^3)$-access kernel has portion
$\Theta(1/n)$, so the *total* miss ratio above the lower boundaries thins
as $n$ grows. The term itself does not: on its active cache range it is the
entire miss traffic, its absolute misses grow as $n^2$, and its boundary is
exactly the cliff a cache-size sweep measures. The terms — not any single
aggregate — are the objects to reason with.

**Trust gate.** Every kernel below ran with **exact Barvinok counting**
(`method = exact`): bin populations conserve to the integer
($\sum M = $ warm accesses, verified at two sizes per kernel and shown in
each header), and the reconstructed distance histograms match a
brute-force trace interpreter bin-for-bin at aligned and unaligned sizes
(30/30 spot checks). Polynomials are exact on the anchor residue class
$n \equiv 0 \pmod h$ (h = 8 or 16, per kernel); other classes differ only
in lower-order boundary constants. Ramp coefficients are evaluated at the
anchor sizes and quoted to three digits; ramp *orders* come from the exact
degrees. Populations below $10^{-6}$ of the accesses are aggregated into
their order row but omitted from the tables.


# Suite overview

Access order $a$, DMD order $d$, headroom $d-a$; top spectrum entries. Single-shot / infinite-repeat per kernel.

| kernel | model | a | d | headroom | leading terms |
|---|---|---|---|---|---|
| heat-3d | single | 4 | 5.5 | +1.5 | 0.75·n^5.5,  1.06·n^5 |
| 2mm | inf | 3 | 4 | +1 | 0.0884·n^4,  2.12·n^3.5 |
| 2mm | single | 3 | 4 | +1 | 0.0884·n^4,  2.12·n^3.5 |
| 3mm | inf | 3 | 4 | +1 | 0.133·n^4,  3.18·n^3.5 |
| 3mm | single | 3 | 4 | +1 | 0.133·n^4,  3.18·n^3.5 |
| atax | inf | 2 | 3 | +1 | 0.0442·n^3,  0.22·n^2.5 |
| bicg | inf | 2 | 3 | +1 | 0.0442·n^3,  0.153·n^2.5 |
| cholesky | inf | 3 | 4 | +1 | 0.00347·n^4,  0.00607·n^3.5 |
| cholesky | single | 3 | 4 | +1 | 0.00347·n^4,  0.00607·n^3.5 |
| convolution | inf | 2 | 3 | +1 | 0.125·n^3,  1.12·n^2.5 |
| correlation | inf | 3 | 4 | +1 | 0.0153·n^4,  1.33·n^3.5 |
| correlation | single | 3 | 4 | +1 | 0.0161·n^4,  1.33·n^3.5 |
| covariance | inf | 3 | 4 | +1 | 0.0161·n^4,  1.33·n^3.5 |
| covariance | single | 3 | 4 | +1 | 0.0161·n^4,  1.33·n^3.5 |
| doitgen | inf | 4 | 5 | +1 | 0.0442·n^5,  1.1·n^4.5 |
| doitgen | single | 4 | 5 | +1 | 0.0442·n^5,  1.06·n^4.5 |
| fdtd | inf | 3 | 4 | +1 | 0.617·n^4,  0.253·n^3.5 |
| fdtd | single | 3 | 4 | +1 | 0.616·n^4,  0.253·n^3.5 |
| floyd_warshall | inf | 3 | 4 | +1 | 0.0442·n^4,  0.0625·n^3.5 |
| floyd_warshall | single | 3 | 4 | +1 | 0.0442·n^4,  0.0625·n^3.5 |
| gemm | inf | 3 | 4 | +1 | 0.0442·n^4,  0.0625·n^3.5 |
| gemm | single | 3 | 4 | +1 | 0.0442·n^4,  0.0625·n^3.5 |
| gemver | inf | 2 | 3 | +1 | 0.114·n^3,  1.28·n^2.5 |
| gemver | single | 2 | 3 | +1 | 0.0722·n^3,  1.28·n^2.5 |
| gesummv | inf | 2 | 3 | +1 | 0.125·n^3,  0.0765·n^2.5 |
| gramschmidt | inf | 3 | 4 | +1 | 0.016·n^4,  2.74·n^3.5 |
| gramschmidt | single | 3 | 4 | +1 | 0.0161·n^4,  2.74·n^3.5 |
| jacobi-2d | inf | 3 | 4 | +1 | 0.5·n^4,  0.707·n^3.5 |
| jacobi-2d | single | 3 | 4 | +1 | 0.5·n^4,  0.707·n^3.5 |
| lu | inf | 3 | 4 | +1 | 0.00643·n^4,  0.22·n^3.5 |
| lu | single | 3 | 4 | +1 | 0.00643·n^4,  0.221·n^3.5 |
| lu_decomp | inf | 3 | 4 | +1 | 0.0101·n^4,  0.0166·n^3.5 |
| lu_decomp | single | 3 | 4 | +1 | 0.01·n^4,  0.0166·n^3.5 |
| mvt | inf | 2 | 3 | +1 | 0.0721·n^3,  1.12·n^2.5 |
| mvt | single | 2 | 3 | +1 | 0.036·n^3,  1.12·n^2.5 |
| seidel-2d | inf | 3 | 4 | +1 | 0.0884·n^4,  0.306·n^3.5 |
| seidel-2d | single | 3 | 4 | +1 | 0.0884·n^4,  0.306·n^3.5 |
| symm | inf | 3 | 4 | +1 | 0.0466·n^4,  1.08·n^3.5 |
| symm | single | 3 | 4 | +1 | 0.0466·n^4,  1.08·n^3.5 |
| syr2k | inf | 3 | 4 | +1 | 0.0466·n^4,  1.08·n^3.5 |
| syr2k | single | 3 | 4 | +1 | 0.0466·n^4,  1.08·n^3.5 |
| syrk | inf | 3 | 4 | +1 | 0.0166·n^4,  0.416·n^3.5 |
| syrk | single | 3 | 4 | +1 | 0.0166·n^4,  0.416·n^3.5 |
| trisolve | inf | 2 | 3 | +1 | 0.0156·n^3,  0.0226·n^2.5 |
| trmm | inf | 3 | 4 | +1 | 0.0169·n^4,  1.05·n^3.5 |
| trmm | single | 3 | 4 | +1 | 0.0169·n^4,  1.05·n^3.5 |
| atax | single | 2 | 2.5 | +0.5 | 0.219·n^2.5,  11.7·n^2 |
| bicg | single | 2 | 2.5 | +0.5 | 0.153·n^2.5,  12.8·n^2 |
| convolution | single | 2 | 2.5 | +0.5 | 1.12·n^2.5,  394·n^2 |
| gesummv | single | 2 | 2.5 | +0.5 | 0.0765·n^2.5,  15.1·n^2 |
| jacobi-1d | inf | 2 | 2.5 | +0.5 | 0.25·n^2.5,  9.4·n^2 |
| jacobi-1d | single | 2 | 2.5 | +0.5 | 0.25·n^2.5,  9.4·n^2 |
| trisolve | single | 2 | 2.5 | +0.5 | 0.0226·n^2.5,  2.88·n^2 |

# Per-kernel principal terms


## 2mm — infinite-repeat  [`exact`]

Accesses $A(n) = 8·n^3 + 3·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0884·n^4  +  2.12·n^3.5  +  10.7·n^3  +  7.95·n^2.5  +  13.2·n^2  +  9.33·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.0137 | read E[i6, i5] (i0=0, i4=0); read E[i6, i5] (i0=0) |
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.0137 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.00195 | read E[i6, i5] (i0=0, i4=0); read E[i6, i5] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.00195 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0957 | read E[i6, i5] (i0=0, i4=0); read E[i6, i5] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0957 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.0137 | read E[i6, i5] (i0=0, i4=0); read E[i6, i5] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.0137 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.0137 | read A[i4, i6] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.0137 | read B[i1, i3] (i0=0, i1=0, i2=0); read B[i1, i3] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.00195 | read A[i4, i6] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.00195 | read B[i1, i3] (i0=0, i1=0, i2=0); read B[i1, i3] (i0=0) |
| n^3 | 3.03 | level | 3 | (7/4)·n^3 | 0.219 | read B[i1, i3] (i0=0); read A[i4, i6] (i0=0) |
| n^3 | 3.03 | level | 3 | (7/4)·n^3 | 0.219 | write A[i1, i2] (i0=0); write D[i4, i5] (i0=0) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (21/8)·n^2 | 0.219 | write A[i1, i2] (i0=0); read A[i1, i2] (i0=0) (+3) |
| n^3 | 0.433 | level | 3 | (1/4)·n^3 | 0.0312 | write A[i1, i2] (i0=0); write D[i4, i5] (i0=0, i6=0) (+1) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read E[i6, i5] (i0=0, i5=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read E[i6, i5] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.25 | level | 1 | (1/4)·n^3 + (1/8)·n^2 | 0.0312 | read A[i1, i2] (i0=0); write D[i4, i5] (i0=0) (+2) |
| n^3 | 0.0988 | level | (5/8)·n^2 | (1/8)·n^2 - 2·n | 0.0156/n | read B[i1, i3] (i0=0, i1=0, i2=0); read B[i1, i3] (i0=0, i2=0) |
| n^3 | 0.0988 | level | (5/8)·n^2 | (1/8)·n^2 - 2·n | 0.0156/n | read D[i4, i5] (i0=0, i4=0); read D[i4, i5] (i0=0) |
| n^3 | 0.0884 | level | (1/2)·n^2 + 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0156/n | read A[i4, i6] (i0=0, i5=0) |
| n^3 | 0.0884 | level | (1/2)·n^2 + 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0156/n | write A[i1, i2] (i0=0) |
| n^3 | 0.0773 | level | (1/2)·n^2 + (1/8)·n + 1 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0137/n | read E[i6, i5] (i0=0, i4=0) |
| n^3 | 0.0773 | level | (1/2)·n^2 + (1/4)·n | (7/64)·n^2 + (-15/8)·n + 2 | 0.0137/n | read C[i3, i2] (i0=0, i1=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0156/n | read E[i6, i5] (i0=0, i4=0); read E[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read E[i6, i5] (i0=0, i4=0); read E[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0156/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read E[i6, i5] (i0=0, i4=0, i6=0); read E[i6, i5] (i0=0, i6=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read C[i3, i2] (i0=0, i1=0, i3=0); read C[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read E[i6, i5] (i0=0, i5=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read E[i6, i5] (i0=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.011 | level | (1/2)·n^2 + (1/8)·n + 1 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195/n | read E[i6, i5] (i0=0, i4=0) |
| n^3 | 0.011 | level | (1/2)·n^2 + (1/4)·n | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195/n | read C[i3, i2] (i0=0, i1=0) |
| n^2.5 | 1.86 | level | (9/8)·n + 1 | (7/4)·n^2 | 0.219/n | read C[i3, i2] (i0=0); read E[i6, i5] (i0=0, i4=0) (+1) |
| n^2.5 | 1.86 | level | (9/8)·n - 6 | (7/4)·n^2 | 0.219/n | read B[i1, i3] (i0=0); read A[i4, i6] (i0=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.109/n | read E[i6, i5] (i0=0, i4=0, i6=0); read E[i6, i5] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.109/n | read C[i3, i2] (i0=0, i1=0, i3=0); read C[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.109/n | write A[i1, i2] (i0=0); read A[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.109/n | read B[i1, i3] (i0=0, i1=0, i2=0, i3=0); read B[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.265 | level | (9/8)·n - 5 | (1/4)·n^2 - 2·n | 0.0312/n | read B[i1, i3] (i0=0); read A[i4, i6] (i0=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0156/n | read A[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0156/n | read B[i1, i3] (i0=0, i1=0, i2=0, i3=0); read B[i1, i3] (i0=0, i3=0) |
| n^2 | 1.58 | level | (5/8)·n^2 | 2·n | 0.25·n^-2 | read B[i1, i3] (i0=0); read D[i4, i5] (i0=0, i4=0, i5=0) (+1) |
| n^2 | 0.791 | level | (5/8)·n^2 | n | 0.125·n^-2 | read B[i1, i3] (i0=0, i1=0, i2=0); read B[i1, i3] (i0=0, i2=0) |
| n^2 | 0.791 | level | (5/8)·n^2 | n | 0.125·n^-2 | read D[i4, i5] (i0=0, i4=0); read D[i4, i5] (i0=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + (-1/8)·n + 1 | n - 2 | 0.125·n^-2 | read A[i4, i6] (i0=0, i5=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + (-1/8)·n + 1 | n - 2 | 0.125·n^-2 | write A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + 1 | n - 2 | 0.125·n^-2 | write A[i1, i2] (i0=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + (1/8)·n + 1 | n | 0.125·n^-2 | read E[i6, i5] (i0=0, i4=0, i5=0, i6=0); read E[i6, i5] (i0=0, i4=0, i5=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + 1 | n - 2 | 0.125·n^-2 | read A[i4, i6] (i0=0, i5=0, i6=0) |
| n^2 | 0.619 | level | (1/2)·n^2 + (1/8)·n + 1 | (7/8)·n - 1 | 0.109·n^-2 | read E[i6, i5] (i0=0, i4=0) |
| n^2 | 0.617 | ramp | (1/2)·n^2 + (1/8)·n + 1  →  (1/2)·n^2 + (1/4)·n | (7/8)·n - 1 | 0.109·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0) |
| n^2 | 0.617 | ramp | (1/2)·n^2 + (1/8)·n + 1  →  (1/2)·n^2 + (1/4)·n | (7/8)·n - 1 | 0.109·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i1=0, i3=0); read C[i3, i2] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0, i4=0, i5=0); read E[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0, i4=0, i5=0); read E[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.125·n^-2 | read B[i1, i3] (i0=0); read E[i6, i5] (i0=0, i5=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0, i3=0); read C[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (29/8)·n + (15/8) | (1/8)·n + (-9/8) | 0.0156·n^-2 | read E[i6, i5] (i0=0, i4=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (1/8)·n + 1 | (1/8)·n - 1 | 0.0156·n^-2 | read E[i6, i5] (i0=0, i4=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (1/8)·n + 1 | (1/8)·n - 2 | 0.0156·n^-2 | read E[i6, i5] (i0=0, i4=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (15/4)·n + (7/4) | (1/8)·n + (-9/8) | 0.0156·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0156·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (1/8)·n + 1 | (1/8)·n - 2 | 0.0156·n^-2 | read E[i6, i5] (i0=0, i4=0, i6=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0156·n^-2 | read C[i3, i2] (i0=0, i1=0, i3=0) |
| n^2 | 0.0862 | ramp | (1/2)·n^2 + (1/8)·n + 2  →  (1/2)·n^2 + (1/4)·n | (1/8)·n - 1 | 0.0156·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0) |
| n^2 | 0.0862 | ramp | (1/2)·n^2 + (1/8)·n + 1  →  (1/2)·n^2 + (1/4)·n - 1 | (1/8)·n - 1 | 0.0156·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.0782 | ramp | (3/8)·n^2 + n + 1  →  (1/2)·n^2 - 2·n + 1 | (1/8)·n - 2 | 0.0156·n^-2 | read A[i4, i6] (i0=0, i5=0) |
| n^2 | 0.0782 | ramp | (3/8)·n^2 + n + 1  →  (1/2)·n^2 - 2·n + 1 | (1/8)·n - 2 | 0.0156·n^-2 | write A[i1, i2] (i0=0, i1=0) |
| n^2 | 0.0728 | ramp | (3/8)·n^2 + 10  →  (3/8)·n^2 + n - 14 | (1/8)·n - 2 | 0.0156·n^-2 | write A[i1, i2] (i0=0) |
| n^2 | 0.0728 | ramp | (3/8)·n^2 + 9  →  (3/8)·n^2 + n - 15 | (1/8)·n - 2 | 0.0156·n^-2 | read A[i4, i6] (i0=0, i4=0, i5=0) |
| n^1 | 0.707 | level | (1/2)·n^2 - n + 1 | 1 | 0.125·n^-3 | write A[i1, i2] (i0=0, i1=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/8)·n + 1 | 1 | 0.125·n^-3 | read E[i6, i5] (i0=0, i4=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/8)·n | 1 | 0.125·n^-3 | read C[i3, i2] (i0=0, i1=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/8)·n + 1 | 1 | 0.125·n^-3 | read E[i6, i5] (i0=0, i4=0, i6=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/4)·n | 1 | 0.125·n^-3 | read C[i3, i2] (i0=0, i1=0, i3=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/4)·n | 1 | 0.125·n^-3 | read C[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/8)·n + 1 | 1 | 0.125·n^-3 | read C[i3, i2] (i0=0, i1=0, i2=0, i3=0) |
| n^1 | 0.707 | level | (1/2)·n^2 - n + 1 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i5=0, i6=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (7/8)·n - 7 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i4=0, i5=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i5=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | 1 | 0.125·n^-3 | write A[i1, i2] (i0=0, i1=0, i2=0); read A[i4, i6] (i0=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + 1 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i4=0, i5=0, i6=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (7/8)·n - 6 | 1 | 0.125·n^-3 | write A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + 2 | 1 | 0.125·n^-3 | write A[i1, i2] (i0=0) |

Two chained matmuls: the intermediate C and the result E each contribute a verbatim copy of gemm's n^4 term (0.0387 + 0.0055 each, i.e. 0.0442 per matmul), so the total leading coefficient 0.0884 is exactly twice gemm's at the same boundaries. Everything else mirrors gemm per stage.

## 2mm — single-shot  [`exact`]

Accesses $A(n) = 8·n^3 + 3·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0884·n^4  +  2.12·n^3.5  +  10.3·n^3  +  7.95·n^2.5  +  6.85·n^2  +  2.54·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.0137 | read E[i6, i5] (i0=0) |
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.0137 | read C[i3, i2] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.00195 | read E[i6, i5] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.00195 | read C[i3, i2] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0957 | read E[i6, i5] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0957 | read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.0137 | read E[i6, i5] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.0137 | read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.0137 | read A[i4, i6] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.0137 | read B[i1, i3] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.00195 | read A[i4, i6] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.00195 | read B[i1, i3] (i0=0) |
| n^3 | 3.46 | level | 3 | 2·n^3 | 0.25 | read B[i1, i3] (i0=0); write A[i1, i2] (i0=0) (+2) |
| n^3 | 2 | level | 1 | 2·n^3 + n^2 | 0.25 | read A[i1, i2] (i0=0); write D[i4, i5] (i0=0) (+2) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 | 0.109 | read A[i4, i6] (i0=0) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 | 0.109 | read B[i1, i3] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read E[i6, i5] (i0=0, i5=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read E[i6, i5] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.109/n | read C[i3, i2] (i0=0) |
| n^3 | 0.0884 | level | (1/2)·n^2 + 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0156/n | read A[i4, i6] (i0=0, i5=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0156/n | read E[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read E[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0156/n | read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read E[i6, i5] (i0=0, i6=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0156/n | read C[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read E[i6, i5] (i0=0, i5=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read E[i6, i5] (i0=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0156/n | read C[i3, i2] (i0=0) |
| n^2.5 | 1.86 | level | (9/8)·n + 1 | (7/4)·n^2 | 0.219/n | read C[i3, i2] (i0=0); read E[i6, i5] (i0=0) |
| n^2.5 | 1.86 | level | (9/8)·n - 6 | (7/4)·n^2 | 0.219/n | read B[i1, i3] (i0=0); read A[i4, i6] (i0=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.109/n | read E[i6, i5] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.109/n | read C[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.109/n | read A[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.109/n | read B[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.265 | level | (9/8)·n - 5 | (1/4)·n^2 - 2·n | 0.0312/n | read B[i1, i3] (i0=0); read A[i4, i6] (i0=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0156/n | read A[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0156/n | read B[i1, i3] (i0=0, i3=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 | 0.219/n | write A[i1, i2] (i0=0); read D[i4, i5] (i0=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + (-1/8)·n + 1 | n - 2 | 0.125·n^-2 | read A[i4, i6] (i0=0, i5=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + 1 | n - 2 | 0.125·n^-2 | read A[i4, i6] (i0=0, i5=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.125·n^-2 | read E[i6, i5] (i0=0, i5=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.125·n^-2 | read C[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.0782 | ramp | (3/8)·n^2 + n + 1  →  (1/2)·n^2 - 2·n + 1 | (1/8)·n - 2 | 0.0156·n^-2 | read A[i4, i6] (i0=0, i5=0) |
| n^2 | 0.0728 | ramp | (3/8)·n^2 + 9  →  (3/8)·n^2 + n - 15 | (1/8)·n - 2 | 0.0156·n^-2 | read A[i4, i6] (i0=0, i4=0, i5=0) |
| n^1 | 0.707 | level | (1/2)·n^2 - n + 1 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i5=0, i6=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (7/8)·n - 7 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i4=0, i5=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i5=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + 1 | 1 | 0.125·n^-3 | read A[i4, i6] (i0=0, i4=0, i5=0, i6=0) |

Two chained matmuls: the intermediate C and the result E each contribute a verbatim copy of gemm's n^4 term (0.0387 + 0.0055 each, i.e. 0.0442 per matmul), so the total leading coefficient 0.0884 is exactly twice gemm's at the same boundaries. Everything else mirrors gemm per stage.

## 3mm — infinite-repeat  [`exact`]

Accesses $A(n) = 12·n^3 + 3·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.133·n^4  +  3.18·n^3.5  +  16.2·n^3  +  11.9·n^2.5  +  21.5·n^2  +  18.3·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.00911 | read D[i9, i8] (i0=0) |
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.00911 | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.00911 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.0013 | read D[i9, i8] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.0013 | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.0013 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0638 | read D[i9, i8] (i0=0, i7=0); read D[i9, i8] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0638 | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0638 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.00911 | read D[i9, i8] (i0=0, i7=0); read D[i9, i8] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.00911 | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.00911 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.00911 | read A[i7, i9] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.00911 | read E[i4, i6] (i0=0, i4=0, i5=0); read E[i4, i6] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.00911 | read B[i1, i3] (i0=0, i1=0, i2=0); read B[i1, i3] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.0013 | read A[i7, i9] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.0013 | read E[i4, i6] (i0=0, i4=0, i5=0); read E[i4, i6] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.0013 | read B[i1, i3] (i0=0, i1=0, i2=0); read B[i1, i3] (i0=0) |
| n^3 | 4.55 | level | 3 | (21/8)·n^3 | 0.219 | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0) (+1) |
| n^3 | 4.55 | level | 3 | (21/8)·n^3 | 0.219 | write A[i1, i2] (i0=0); write D[i4, i5] (i0=0) (+1) |
| n^3 | 2.62 | level | 1 | (21/8)·n^3 + (21/8)·n^2 | 0.219 | write A[i1, i2] (i0=0); read A[i1, i2] (i0=0) (+5) |
| n^3 | 0.65 | level | 3 | (3/8)·n^3 | 0.0312 | write A[i1, i2] (i0=0); write D[i4, i5] (i0=0) (+2) |
| n^3 | 0.375 | level | 1 | (3/8)·n^3 | 0.0312 | read A[i1, i2] (i0=0); read D[i4, i5] (i0=0) (+2) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read D[i9, i8] (i0=0, i8=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read F[i6, i5] (i0=0, i4=0, i5=0); read F[i6, i5] (i0=0, i5=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read D[i9, i8] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.234 | level | (7/8)·n^2 | (1/4)·n^2 - 4·n | 0.0208/n | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0, i4=0, i5=0) (+1) |
| n^3 | 0.117 | level | (7/8)·n^2 | (1/8)·n^2 - 2·n | 0.0104/n | write G[i7, i8] (i0=0, i7=0); write G[i7, i8] (i0=0) |
| n^3 | 0.108 | level | (3/4)·n^2 + 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0104/n | read A[i7, i9] (i0=0, i8=0) |
| n^3 | 0.0955 | ramp | (1/2)·n^2 + (5/2)·n - 2  →  (3/4)·n^2 - 2 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0104/n | write D[i4, i5] (i0=0) |
| n^3 | 0.0947 | level | (3/4)·n^2 + (1/4)·n | (7/64)·n^2 + (-15/8)·n + 2 | 0.00911/n | read F[i6, i5] (i0=0, i4=0) |
| n^3 | 0.0947 | level | (3/4)·n^2 + (1/4)·n | (7/64)·n^2 + (-15/8)·n + 2 | 0.00911/n | read C[i3, i2] (i0=0, i1=0) |
| n^3 | 0.0884 | level | (1/2)·n^2 + 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0104/n | write A[i1, i2] (i0=0) |
| n^3 | 0.0547 | ramp | (1/8)·n^2 + (21/8)·n - 2  →  (3/8)·n^2 + (1/8)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.00911/n | read D[i9, i8] (i0=0, i7=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read D[i9, i8] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read D[i9, i8] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read D[i9, i8] (i0=0, i9=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read F[i6, i5] (i0=0, i4=0, i6=0); read F[i6, i5] (i0=0, i6=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read C[i3, i2] (i0=0, i1=0, i3=0); read C[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read D[i9, i8] (i0=0, i8=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read F[i6, i5] (i0=0, i4=0, i5=0); read F[i6, i5] (i0=0, i5=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read D[i9, i8] (i0=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^3 | 0.0135 | level | (3/4)·n^2 + (1/4)·n | (1/64)·n^2 + (-3/8)·n + 2 | 0.0013/n | read F[i6, i5] (i0=0, i4=0) |
| n^3 | 0.0135 | level | (3/4)·n^2 + (1/4)·n | (1/64)·n^2 + (-3/8)·n + 2 | 0.0013/n | read C[i3, i2] (i0=0, i1=0) |
| n^3 | 0.00764 | ramp | (1/8)·n^2 + (33/8)·n - 14  →  (3/8)·n^2 + (-3/4)·n - 9 | (1/64)·n^2 + (-3/8)·n + 2 | 0.0013/n | read D[i9, i8] (i0=0, i7=0) |
| n^2.5 | 2.78 | level | (9/8)·n + 1 | (21/8)·n^2 | 0.219/n | read C[i3, i2] (i0=0); read F[i6, i5] (i0=0) (+2) |
| n^2.5 | 2.78 | level | (9/8)·n - 6 | (21/8)·n^2 | 0.219/n | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0) (+1) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.0729/n | read D[i9, i8] (i0=0, i7=0, i9=0); read D[i9, i8] (i0=0, i9=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.0729/n | read F[i6, i5] (i0=0, i4=0, i6=0); read F[i6, i5] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.0729/n | read C[i3, i2] (i0=0, i1=0, i3=0); read C[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.0729/n | write A[i1, i2] (i0=0); read A[i7, i9] (i0=0, i9=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.0729/n | read E[i4, i6] (i0=0, i4=0, i5=0, i6=0); read E[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.0729/n | read B[i1, i3] (i0=0, i1=0, i2=0, i3=0); read B[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.398 | level | (9/8)·n - 5 | (3/8)·n^2 - 3·n | 0.0312/n | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0) (+1) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0104/n | read A[i7, i9] (i0=0, i9=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0104/n | read E[i4, i6] (i0=0, i4=0, i5=0, i6=0); read E[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0104/n | read B[i1, i3] (i0=0, i1=0, i2=0, i3=0); read B[i1, i3] (i0=0, i3=0) |
| n^2 | 2.81 | level | (7/8)·n^2 | 3·n | 0.25·n^-2 | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0) (+2) |
| n^2 | 0.935 | level | (7/8)·n^2 | n | 0.0833·n^-2 | read E[i4, i6] (i0=0, i4=0, i5=0); read E[i4, i6] (i0=0, i5=0) |
| n^2 | 0.935 | level | (7/8)·n^2 | n | 0.0833·n^-2 | read B[i1, i3] (i0=0, i1=0, i2=0); read B[i1, i3] (i0=0, i2=0) |
| n^2 | 0.935 | level | (7/8)·n^2 | n | 0.0833·n^-2 | write G[i7, i8] (i0=0, i7=0); write G[i7, i8] (i0=0) |
| n^2 | 0.866 | level | (3/4)·n^2 + (-1/8)·n + 1 | n - 2 | 0.0833·n^-2 | read A[i7, i9] (i0=0, i8=0) |
| n^2 | 0.866 | level | (3/4)·n^2 + 1 | n - 2 | 0.0833·n^-2 | read A[i7, i9] (i0=0, i8=0, i9=0) |
| n^2 | 0.824 | ramp | (5/8)·n^2 + (1/4)·n  →  (3/4)·n^2 + (-1/8)·n | n - 2 | 0.0833·n^-2 | write D[i4, i5] (i0=0, i5=0) |
| n^2 | 0.786 | ramp | (1/2)·n^2 + (3/2)·n - 1  →  (3/4)·n^2 + (-1/4)·n + 2 | n - 2 | 0.0833·n^-2 | write D[i4, i5] (i0=0) |
| n^2 | 0.755 | ramp | (3/4)·n^2 + (1/8)·n + 1  →  (3/4)·n^2 + (1/4)·n | (7/8)·n - 1 | 0.0729·n^-2 | read F[i6, i5] (i0=0, i4=0, i5=0) |
| n^2 | 0.755 | ramp | (3/4)·n^2 + (1/8)·n + 1  →  (3/4)·n^2 + (1/4)·n | (7/8)·n - 1 | 0.0729·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0) |
| n^2 | 0.755 | ramp | (3/4)·n^2 + (1/8)·n + 1  →  (3/4)·n^2 + (1/4)·n | (7/8)·n - 1 | 0.0729·n^-2 | read F[i6, i5] (i0=0, i4=0) |
| n^2 | 0.755 | ramp | (3/4)·n^2 + (1/8)·n + 1  →  (3/4)·n^2 + (1/4)·n | (7/8)·n - 1 | 0.0729·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + (-1/8)·n + 1 | n - 2 | 0.0833·n^-2 | write A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.707 | level | (1/2)·n^2 + 1 | n - 2 | 0.0833·n^-2 | write A[i1, i2] (i0=0) |
| n^2 | 0.487 | ramp | (1/4)·n^2 + (3/8)·n  →  (3/8)·n^2 | (7/8)·n - 1 | 0.0729·n^-2 | read D[i9, i8] (i0=0, i7=0) |
| n^2 | 0.433 | ramp | (1/8)·n^2 + (13/8)·n - 1  →  (3/8)·n^2 + (-1/4)·n + 3 | (7/8)·n - 1 | 0.0729·n^-2 | read D[i9, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i4=0); read F[i6, i5] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i1=0); read C[i3, i2] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0, i9=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i4=0, i6=0); read F[i6, i5] (i0=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i1=0, i3=0); read C[i3, i2] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0, i8=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0, i8=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i4=0, i5=0); read F[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i4=0, i5=0); read F[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0); read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.0833·n^-2 | read B[i1, i3] (i0=0); read D[i9, i8] (i0=0, i8=0, i9=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i4=0, i5=0, i6=0); read F[i6, i5] (i0=0, i5=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0, i3=0); read C[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + 1 | (1/8)·n - 2 | 0.0104·n^-2 | read A[i7, i9] (i0=0, i7=0, i8=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + (11/2)·n + (7/4) | (1/8)·n + (-9/8) | 0.0104·n^-2 | read F[i6, i5] (i0=0, i4=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0104·n^-2 | read F[i6, i5] (i0=0, i4=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + (11/2)·n + (7/4) | (1/8)·n + (-9/8) | 0.0104·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0104·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0104·n^-2 | read F[i6, i5] (i0=0, i4=0, i6=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0104·n^-2 | read C[i3, i2] (i0=0, i1=0, i3=0) |
| n^2 | 0.106 | ramp | (3/4)·n^2 + (1/8)·n + 2  →  (3/4)·n^2 + (1/4)·n | (1/8)·n - 1 | 0.0104·n^-2 | read F[i6, i5] (i0=0, i4=0, i5=0) |
| n^2 | 0.106 | ramp | (3/4)·n^2 + (1/8)·n + 2  →  (3/4)·n^2 + (1/4)·n | (1/8)·n - 1 | 0.0104·n^-2 | read C[i3, i2] (i0=0, i1=0, i2=0) |
| n^2 | 0.106 | ramp | (3/4)·n^2 + (1/8)·n + 1  →  (3/4)·n^2 + (1/4)·n - 1 | (1/8)·n - 1 | 0.0104·n^-2 | read F[i6, i5] (i0=0, i4=0) |
| n^2 | 0.106 | ramp | (3/4)·n^2 + (1/8)·n + 1  →  (3/4)·n^2 + (1/4)·n - 1 | (1/8)·n - 1 | 0.0104·n^-2 | read C[i3, i2] (i0=0, i1=0) |
| n^2 | 0.103 | ramp | (3/4)·n^2 + 2  →  (3/4)·n^2 + (1/8)·n - 1 | (1/8)·n - 2 | 0.0104·n^-2 | write D[i4, i5] (i0=0) |
| n^2 | 0.0982 | ramp | (5/8)·n^2 + n + 1  →  (3/4)·n^2 - 2·n + 1 | (1/8)·n - 2 | 0.0104·n^-2 | read A[i7, i9] (i0=0, i8=0) |
| n^2 | 0.0884 | level | (1/2)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0104·n^-2 | write D[i4, i5] (i0=0, i4=0) |
| n^2 | 0.0782 | ramp | (3/8)·n^2 + n + 1  →  (1/2)·n^2 - 2·n + 1 | (1/8)·n - 2 | 0.0104·n^-2 | write A[i1, i2] (i0=0, i1=0) |
| n^2 | 0.0728 | ramp | (3/8)·n^2 + 10  →  (3/8)·n^2 + n - 14 | (1/8)·n - 2 | 0.0104·n^-2 | write A[i1, i2] (i0=0) |
| n^2 | 0.0727 | ramp | (3/8)·n^2 + (1/8)·n + 2  →  (3/8)·n^2 + (1/4)·n - 1 | (1/8)·n - 2 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0, i9=0) |
| n^2 | 0.068 | ramp | (1/4)·n^2 + (9/8)·n  →  (3/8)·n^2 + (-7/8)·n | (1/8)·n - 1 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0) |
| n^2 | 0.0605 | ramp | (1/8)·n^2 + (25/8)·n - 7  →  (3/8)·n^2 - 2·n + 11 | (1/8)·n - 1 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n - 2 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (-1/8)·n + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i7=0, i8=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/8)·n | 1 | 0.0833·n^-3 | read F[i6, i5] (i0=0, i4=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/8)·n | 1 | 0.0833·n^-3 | read C[i3, i2] (i0=0, i1=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/4)·n | 1 | 0.0833·n^-3 | read F[i6, i5] (i0=0, i4=0, i6=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/4)·n | 1 | 0.0833·n^-3 | read C[i3, i2] (i0=0, i1=0, i3=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/4)·n | 1 | 0.0833·n^-3 | read F[i6, i5] (i0=0, i4=0, i5=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/8)·n + 1 | 1 | 0.0833·n^-3 | read F[i6, i5] (i0=0, i4=0, i5=0, i6=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/4)·n | 1 | 0.0833·n^-3 | read C[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (1/8)·n + 1 | 1 | 0.0833·n^-3 | read C[i3, i2] (i0=0, i1=0, i2=0, i3=0) |
| n^1 | 0.866 | level | (3/4)·n^2 | 1 | 0.0833·n^-3 | write D[i4, i5] (i0=0, i5=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i7=0, i8=0, i9=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + 1 | 1 | 0.0833·n^-3 | write D[i4, i5] (i0=0) |
| n^1 | 0.866 | level | (3/4)·n^2 - n + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i8=0, i9=0) |
| n^1 | 0.791 | level | (5/8)·n^2 + (-1/8)·n + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i8=0) |
| n^1 | 0.707 | level | (1/2)·n^2 - n + 1 | 1 | 0.0833·n^-3 | write A[i1, i2] (i0=0, i1=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/8)·n | 1 | 0.0833·n^-3 | write D[i4, i5] (i0=0, i4=0, i5=0); write D[i4, i5] (i0=0, i5=0) |
| n^1 | 0.707 | level | (1/2)·n^2 + (1/4)·n | 1 | 0.0833·n^-3 | write D[i4, i5] (i0=0, i4=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | 1 | 0.0833·n^-3 | write A[i1, i2] (i0=0, i1=0, i2=0); read A[i7, i9] (i0=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (1/8)·n | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0, i9=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + 2 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i7=0, i8=0, i9=0); read D[i9, i8] (i0=0, i7=0, i8=0, i9=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (7/8)·n - 6 | 1 | 0.0833·n^-3 | write A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + 2 | 1 | 0.0833·n^-3 | write A[i1, i2] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (21/8) | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0, i8=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0, i8=0) |

Three chained matmuls: three copies of gemm's n^4 term, total coefficient 3 × 0.0442 = 0.133, at the same (1/8)n^2-line boundary. The 1:2:3 gemm:2mm:3mm ratio is exact in the term table.

## 3mm — single-shot  [`exact`]

Accesses $A(n) = 12·n^3 + 3·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.133·n^4  +  3.18·n^3.5  +  15.4·n^3  +  11.9·n^2.5  +  11·n^2  +  5.67·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.00911 | read D[i9, i8] (i0=0) |
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.00911 | read F[i6, i5] (i0=0) |
| n^4 | 0.0387 | level | (1/8)·n^2 + (3/8)·n + 1 | (7/64)·n^3 + (-127/64)·n^2 + (31/8)·n - 2 | 0.00911 | read C[i3, i2] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.0013 | read D[i9, i8] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.0013 | read F[i6, i5] (i0=0) |
| n^4 | 0.00552 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/64)·n^3 + (-25/64)·n^2 + (19/8)·n - 2 | 0.0013 | read C[i3, i2] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0638 | read D[i9, i8] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0638 | read F[i6, i5] (i0=0) |
| n^3.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^3 + (-7/8)·n^2 | 0.0638 | read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.00911 | read D[i9, i8] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.00911 | read F[i6, i5] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^3 + (-7/8)·n^2 | 0.00911 | read C[i3, i2] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.00911 | read A[i7, i9] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.00911 | read E[i4, i6] (i0=0) |
| n^3.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^3 + (-7/4)·n^2 | 0.00911 | read B[i1, i3] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.0013 | read A[i7, i9] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.0013 | read E[i4, i6] (i0=0) |
| n^3.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.0013 | read B[i1, i3] (i0=0) |
| n^3 | 5.2 | level | 3 | 3·n^3 | 0.25 | read B[i1, i3] (i0=0); write A[i1, i2] (i0=0) (+5) |
| n^3 | 3.03 | level | 3 | (7/4)·n^3 | 0.146 | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0) |
| n^3 | 3 | level | 1 | 3·n^3 | 0.25 | read A[i1, i2] (i0=0); read D[i4, i5] (i0=0) (+2) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 | 0.0729 | read A[i7, i9] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read D[i9, i8] (i0=0, i8=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read F[i6, i5] (i0=0, i5=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read D[i9, i8] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read F[i6, i5] (i0=0) |
| n^3 | 0.308 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n + 1 | (7/8)·n^2 + (-15/8)·n + 1 | 0.0729/n | read C[i3, i2] (i0=0) |
| n^3 | 0.108 | level | (3/4)·n^2 + 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0104/n | read A[i7, i9] (i0=0, i8=0) |
| n^3 | 0.0547 | ramp | (1/8)·n^2 + (21/8)·n - 2  →  (3/8)·n^2 + (1/8)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.00911/n | read D[i9, i8] (i0=0, i7=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read D[i9, i8] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read D[i9, i8] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read F[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read F[i6, i5] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (29/8) | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read C[i3, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read D[i9, i8] (i0=0, i9=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read F[i6, i5] (i0=0, i6=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0104/n | read C[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read D[i9, i8] (i0=0, i8=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read F[i6, i5] (i0=0, i5=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read C[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read D[i9, i8] (i0=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read F[i6, i5] (i0=0) |
| n^3 | 0.0431 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0104/n | read C[i3, i2] (i0=0) |
| n^3 | 0.00764 | ramp | (1/8)·n^2 + (33/8)·n - 14  →  (3/8)·n^2 + (-3/4)·n - 9 | (1/64)·n^2 + (-3/8)·n + 2 | 0.0013/n | read D[i9, i8] (i0=0, i7=0) |
| n^2.5 | 2.78 | level | (9/8)·n + 1 | (21/8)·n^2 | 0.219/n | read C[i3, i2] (i0=0); read F[i6, i5] (i0=0) (+1) |
| n^2.5 | 2.78 | level | (9/8)·n - 6 | (21/8)·n^2 | 0.219/n | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0) (+1) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.0729/n | read D[i9, i8] (i0=0, i9=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.0729/n | read F[i6, i5] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 | 0.0729/n | read C[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.0729/n | read A[i7, i9] (i0=0, i9=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.0729/n | read E[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^2 | 0.0729/n | read B[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.398 | level | (9/8)·n - 5 | (3/8)·n^2 - 3·n | 0.0312/n | read B[i1, i3] (i0=0); read E[i4, i6] (i0=0) (+1) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0104/n | read A[i7, i9] (i0=0, i9=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0104/n | read E[i4, i6] (i0=0, i6=0) |
| n^2.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^2 - n | 0.0104/n | read B[i1, i3] (i0=0, i3=0) |
| n^2 | 2.62 | level | 1 | (21/8)·n^2 | 0.219/n | write A[i1, i2] (i0=0); write D[i4, i5] (i0=0) (+1) |
| n^2 | 0.866 | level | (3/4)·n^2 + (-1/8)·n + 1 | n - 2 | 0.0833·n^-2 | read A[i7, i9] (i0=0, i8=0) |
| n^2 | 0.866 | level | (3/4)·n^2 + 1 | n - 2 | 0.0833·n^-2 | read A[i7, i9] (i0=0, i8=0, i9=0) |
| n^2 | 0.487 | ramp | (1/4)·n^2 + (3/8)·n  →  (3/8)·n^2 | (7/8)·n - 1 | 0.0729·n^-2 | read D[i9, i8] (i0=0, i7=0) |
| n^2 | 0.433 | ramp | (1/8)·n^2 + (13/8)·n - 1  →  (3/8)·n^2 + (-1/4)·n + 3 | (7/8)·n - 1 | 0.0729·n^-2 | read D[i9, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0, i9=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0, i8=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0, i8=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.0833·n^-2 | read D[i9, i8] (i0=0, i8=0, i9=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.0833·n^-2 | read F[i6, i5] (i0=0, i5=0, i6=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | n - 1 | 0.0833·n^-2 | read C[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.108 | level | (3/4)·n^2 + 1 | (1/8)·n - 2 | 0.0104·n^-2 | read A[i7, i9] (i0=0, i7=0, i8=0) |
| n^2 | 0.0982 | ramp | (5/8)·n^2 + n + 1  →  (3/4)·n^2 - 2·n + 1 | (1/8)·n - 2 | 0.0104·n^-2 | read A[i7, i9] (i0=0, i8=0) |
| n^2 | 0.0727 | ramp | (3/8)·n^2 + (1/8)·n + 2  →  (3/8)·n^2 + (1/4)·n - 1 | (1/8)·n - 2 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0, i9=0) |
| n^2 | 0.068 | ramp | (1/4)·n^2 + (9/8)·n  →  (3/8)·n^2 + (-7/8)·n | (1/8)·n - 1 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0) |
| n^2 | 0.0605 | ramp | (1/8)·n^2 + (25/8)·n - 7  →  (3/8)·n^2 - 2·n + 11 | (1/8)·n - 1 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n - 2 | 0.0104·n^-2 | read D[i9, i8] (i0=0, i7=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + (-1/8)·n + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i7=0, i8=0) |
| n^1 | 0.866 | level | (3/4)·n^2 + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i7=0, i8=0, i9=0) |
| n^1 | 0.866 | level | (3/4)·n^2 - n + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i8=0, i9=0) |
| n^1 | 0.791 | level | (5/8)·n^2 + (-1/8)·n + 1 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i8=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (1/8)·n | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0, i9=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + 2 | 1 | 0.0833·n^-3 | read A[i7, i9] (i0=0, i7=0, i8=0, i9=0); read D[i9, i8] (i0=0, i7=0, i8=0, i9=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (21/8) | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0, i8=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n | 1 | 0.0833·n^-3 | read D[i9, i8] (i0=0, i7=0, i8=0) |

Three chained matmuls: three copies of gemm's n^4 term, total coefficient 3 × 0.0442 = 0.133, at the same (1/8)n^2-line boundary. The 1:2:3 gemm:2mm:3mm ratio is exact in the term table.

## atax — infinite-repeat  [`exact`]

Accesses $A(n) = 8·n^2 + 2·n$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^3  +  0.22·n^2.5  +  12.5·n^2  +  3.77·n^1.5  +  3.71·n^1  +  1.47·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/4)·n + 4 | 0.0156 | read C[i2, i3] (i0=0) |
| n^2.5 | 0.069 | ramp | (3/8)·n + 3  →  (1/2)·n | (7/64)·n^2 + (-7/4)·n | 0.0137 | read D[i3] (i0=0, i2=0); read D[i3] (i0=0) |
| n^2.5 | 0.0687 | ramp | (3/8)·n + 2  →  (1/2)·n - 1 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read A[i4] (i0=0, i2=1); read A[i4] (i0=0) |
| n^2.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n^2 - 2·n | 0.0156 | read C[i2, i4] (i0=0) |
| n^2.5 | 0.00989 | ramp | (3/8)·n + 4  →  (1/2)·n + 1 | (1/64)·n^2 + (-1/4)·n | 0.00195 | read D[i3] (i0=0, i2=0); read D[i3] (i0=0) |
| n^2.5 | 0.0096 | ramp | (3/8)·n + 3  →  (1/2)·n | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read A[i4] (i0=0) |
| n^2 | 3.25 | level | 3 | (15/8)·n^2 | 0.234 | read D[i3] (i0=0, i2=0); write A[i4] (i0=0, i2=0) (+3) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 | 0.219 | write A[i4] (i0=0, i2=0); read C[i2, i4] (i0=0) (+1) |
| n^2 | 2.84 | level | 3 | (105/64)·n^2 + (7/8)·n | 0.205 | write B[i2] (i0=0); read B[i2] (i0=0, i4=0) (+1) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.109 | read A[i4] (i0=0, i2=0); read A[i4] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 + (7/8)·n | 0.109 | write A[i1] (i0=0); write B[i2] (i0=0) (+2) |
| n^2 | 0.406 | level | 3 | (15/64)·n^2 + (1/8)·n | 0.0293 | write B[i2] (i0=0); read B[i2] (i0=0, i4=0) (+1) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n - 2 | 0.125/n | read C[i2, i3] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n - 2 | 0.125/n | read C[i2, i3] (i0=0, i3=0) |
| n^2 | 0.25 | level | 4 | (1/8)·n^2 - n | 0.0156 | read B[i2] (i0=0) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 | 0.0156 | read B[i2] (i0=0, i3=0); read B[i2] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n - 2 | 0.0156/n | read C[i2, i3] (i0=0, i2=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (21/8) | (1/8)·n + (-9/8) | 0.0156/n | read C[i2, i3] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n - 2 | 0.0156/n | read C[i2, i3] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (-5/8)·n | (1/8)·n - 2 | 0.0156/n | write B[i2] (i0=0) |
| n^1.5 | 0.619 | level | (1/2)·n + 1 | (7/8)·n | 0.109/n | read D[i3] (i0=0, i2=0); read D[i3] (i0=0) |
| n^1.5 | 0.619 | level | (1/2)·n | (7/8)·n | 0.109/n | read A[i4] (i0=0, i2=1, i4=0); read A[i4] (i0=0, i4=0) |
| n^1.5 | 0.612 | level | (3/8)·n + 1 | n | 0.125/n | read D[i3] (i0=0, i2=0); read A[i4] (i0=0, i2=0) (+2) |
| n^1.5 | 0.536 | level | (3/8)·n + 2 | (7/8)·n | 0.109/n | read D[i3] (i0=0, i2=0, i3=0); read D[i3] (i0=0, i3=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 2 | n | 0.125/n | read C[i2, i4] (i0=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 2 | n | 0.125/n | read C[i2, i4] (i0=0, i4=0) |
| n^1.5 | 0.0884 | level | (1/2)·n + 2 | (1/8)·n | 0.0156/n | read D[i3] (i0=0, i2=0); read D[i3] (i0=0) |
| n^1.5 | 0.0884 | level | (1/2)·n + 1 | (1/8)·n - 1 | 0.0156/n | read A[i4] (i0=0, i4=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n | 0.0156/n | read D[i3] (i0=0, i2=0, i3=0); read D[i3] (i0=0, i3=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 1 | (1/8)·n - 2 | 0.0156/n | read D[i3] (i0=0, i2=0); read A[i4] (i0=0, i2=0) |
| n^1.5 | 0.0514 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/8)·n - 2 | 0.0156/n | write A[i1] (i0=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n | 0.109/n | write B[i2] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read C[i2, i3] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read C[i2, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read C[i2, i3] (i0=0, i2=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (21/8) | 1 | 0.125·n^-2 | read C[i2, i3] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read C[i2, i3] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/8)·n | 1 | 0.125·n^-2 | write B[i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/8)·n | 1 | 0.125·n^-2 | write B[i2] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 1 | 1 | 0.125·n^-2 | read D[i3] (i0=0, i2=0); read A[i4] (i0=0, i2=0, i4=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.125·n^-2 | write A[i1] (i0=0, i1=0) |
| n^0.5 | 0.354 | level | (1/8)·n | 1 | 0.125·n^-2 | write A[i1] (i0=0); write B[i2] (i0=0) |

Infinite repeat adds the single decisive term: the matrix wraparound at distance (1/8)n^2 + (3/8)n with population n^2/8 and coefficient 0.0442 — the same constant as gemm's matrix term. This is the closed form of the Krylov-solver argument: across iterations the matrix is genuine reuse, worth keeping resident; single-shot files it as unavoidable cold traffic.

## atax — single-shot  [`exact`]

Accesses $A(n) = 8·n^2 + 2·n$ (exact on n ≡ 0 mod 8); DMD order $n^{2.5}$, headroom **+0.5**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.219·n^2.5  +  11.7·n^2  +  3.72·n^1.5  +  2.11·n^1  +  0.612·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^2.5 | 0.069 | ramp | (3/8)·n + 3  →  (1/2)·n | (7/64)·n^2 + (-7/4)·n | 0.0137 | read D[i3] (i0=0) |
| n^2.5 | 0.0687 | ramp | (3/8)·n + 2  →  (1/2)·n - 1 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read A[i4] (i0=0) |
| n^2.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n^2 - 2·n | 0.0156 | read C[i2, i4] (i0=0) |
| n^2.5 | 0.00963 | ramp | (3/8)·n + 4  →  (1/2)·n + 1 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read D[i3] (i0=0) |
| n^2.5 | 0.0096 | ramp | (3/8)·n + 3  →  (1/2)·n | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read A[i4] (i0=0) |
| n^2 | 4.98 | level | 3 | (23/8)·n^2 + n | 0.359 | write B[i2] (i0=0); read B[i2] (i0=0, i4=0) (+2) |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 | 0.328 | read C[i2, i3] (i0=0); read D[i3] (i0=0) (+1) |
| n^2 | 1 | level | 1 | n^2 | 0.125 | read B[i2] (i0=0, i3=0); read B[i2] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.109 | read A[i4] (i0=0) |
| n^2 | 0.25 | level | 4 | (1/8)·n^2 - n | 0.0156 | read C[i2, i4] (i0=0, i4=0); read B[i2] (i0=0) |
| n^1.5 | 0.619 | level | (1/2)·n + 1 | (7/8)·n | 0.109/n | read D[i3] (i0=0) |
| n^1.5 | 0.619 | level | (1/2)·n | (7/8)·n | 0.109/n | read A[i4] (i0=0, i4=0) |
| n^1.5 | 0.612 | level | (3/8)·n + 1 | n | 0.125/n | read A[i4] (i0=0, i2=0); read A[i4] (i0=0) |
| n^1.5 | 0.536 | level | (3/8)·n + 2 | (7/8)·n | 0.109/n | read D[i3] (i0=0, i3=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 2 | n | 0.125/n | read C[i2, i4] (i0=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 2 | n | 0.125/n | read C[i2, i4] (i0=0, i4=0) |
| n^1.5 | 0.0884 | level | (1/2)·n + 2 | (1/8)·n - 1 | 0.0156/n | read D[i3] (i0=0) |
| n^1.5 | 0.0884 | level | (1/2)·n + 1 | (1/8)·n - 1 | 0.0156/n | read A[i4] (i0=0, i4=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n - 1 | 0.0156/n | read D[i3] (i0=0, i3=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 1 | (1/8)·n - 2 | 0.0156/n | read A[i4] (i0=0, i2=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n | 0.109/n | write B[i2] (i0=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.109/n | write A[i1] (i0=0); write B[i2] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 1 | 1 | 0.125·n^-2 | read A[i4] (i0=0, i2=0, i4=0) |

Single-shot, the matrix C is streamed once and never re-touched: no n^2-distance term exists. The top terms are vector reuses — `read D[i3]`/`read A[i4]` ramps between (3/8)n and (1/2)n lines and the row-window level at (1/4)n + 2 — giving d = 2.5, headroom +0.5. The right transformation at this scope is fusion/interchange of the two nests.

## bicg — infinite-repeat  [`exact`]

Accesses $A(n) = 7·n^2 + n$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^3  +  0.153·n^2.5  +  13.8·n^2  +  2.45·n^1.5  +  6.2·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0442 | level | (1/8)·n^2 + (1/2)·n | (1/8)·n^2 + (-9/4)·n + 4 | 0.0179 | read D[i1, i2] (i0=0) |
| n^2.5 | 0.067 | level | (3/8)·n + 2 | (7/64)·n^2 + (-7/4)·n | 0.0156 | read D[i1, i2] (i0=0, i1=0); read E[i2] (i0=0, i1=0) (+1) |
| n^2.5 | 0.067 | level | (3/8)·n + 1 | (7/64)·n^2 + (-7/4)·n | 0.0156 | read B[i2] (i0=0, i1=0); read B[i2] (i0=0, i1=1) (+1) |
| n^2.5 | 0.00957 | level | (3/8)·n + 4 | (1/64)·n^2 + (-1/4)·n | 0.00223 | read E[i2] (i0=0, i1=0); read E[i2] (i0=0) |
| n^2.5 | 0.00957 | level | (3/8)·n + 3 | (1/64)·n^2 + (-1/4)·n | 0.00223 | read B[i2] (i0=0, i1=0); read B[i2] (i0=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 + (-7/8)·n | 0.25 | write B[i2] (i0=0, i1=0); read B[i2] (i0=0) (+1) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 | 0.125 | read E[i2] (i0=0, i1=0); read E[i2] (i0=0) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 | 0.125 | read D[i1, i2] (i0=0) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 | 0.125 | read C[i1] (i0=0, i2=0); read C[i1] (i0=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 | 0.125 | read A[i1] (i0=0, i2=0); read A[i1] (i0=0) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.125 | write A[i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/2)·n | n - 2 | 0.143/n | read D[i1, i2] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/2)·n | n - 2 | 0.143/n | read D[i1, i2] (i0=0, i2=0) |
| n^2 | 0.306 | level | 6 | (1/8)·n^2 - n | 0.0179 | read C[i1] (i0=0) |
| n^2 | 0.25 | level | 4 | (1/8)·n^2 | 0.0179 | read A[i1] (i0=0, i2=0); read A[i1] (i0=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (7/8)·n | 0.0179 | read B[i2] (i0=0, i1=0); write B[i2] (i0=0, i1=0) (+1) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 | 0.0179 | write A[i1] (i0=0, i2=0); write A[i1] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.0179/n | read D[i1, i2] (i0=0, i1=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (11/8)·n + (7/2) | (1/8)·n + (-9/8) | 0.0179/n | read D[i1, i2] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.0179/n | read D[i1, i2] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (-1/2)·n + 1 | (1/8)·n - 2 | 0.0179/n | read C[i1] (i0=0, i2=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (-1/2)·n - 1 | (1/8)·n - 2 | 0.0179/n | write A[i1] (i0=0) |
| n^1.5 | 0.612 | level | (3/8)·n + 2 | n | 0.143/n | read B[i2] (i0=0, i1=0); read D[i1, i2] (i0=0, i1=0) (+3) |
| n^1.5 | 0.612 | level | (3/8)·n + 2 | n | 0.143/n | read B[i2] (i0=0, i1=0, i2=0); read D[i1, i2] (i0=0, i1=0, i2=0) (+3) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.125/n | read B[i2] (i0=0, i1=0); read B[i2] (i0=0, i1=1) (+1) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.125/n | read B[i2] (i0=0, i1=0, i2=0); read B[i2] (i0=0, i1=1, i2=0) (+1) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n | 0.0179/n | read E[i2] (i0=0, i1=0); read E[i2] (i0=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 4 | (1/8)·n | 0.0179/n | read E[i2] (i0=0, i1=0, i2=0); read E[i2] (i0=0, i2=0) |
| n^1 | 2.14 | level | 6 | (7/8)·n | 0.125/n | read C[i1] (i0=0, i2=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.125/n | write A[i1] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.143·n^-2 | read D[i1, i2] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.143·n^-2 | read D[i1, i2] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n + 1 | 1 | 0.143·n^-2 | read C[i1] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.143·n^-2 | read D[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (11/8)·n + (7/2) | 1 | 0.143·n^-2 | read D[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.143·n^-2 | read D[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n + 1 | 1 | 0.143·n^-2 | read C[i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n - 1 | 1 | 0.143·n^-2 | write A[i1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n - 1 | 1 | 0.143·n^-2 | write A[i1] (i0=0) |

The matrix wraparound term appears (0.0442·n^3 at the (1/8)n^2-line boundary), lifting d to 3.0, headroom +1.0.

## bicg — single-shot  [`exact`]

Accesses $A(n) = 7·n^2 + n$ (exact on n ≡ 0 mod 8); DMD order $n^{2.5}$, headroom **+0.5**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.153·n^2.5  +  12.8·n^2  +  2.45·n^1.5  +  3.02·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^2.5 | 0.067 | level | (3/8)·n + 2 | (7/64)·n^2 + (-7/4)·n | 0.0156 | read E[i2] (i0=0) |
| n^2.5 | 0.067 | level | (3/8)·n + 1 | (7/64)·n^2 + (-7/4)·n | 0.0156 | read B[i2] (i0=0) |
| n^2.5 | 0.00957 | level | (3/8)·n + 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00223 | read E[i2] (i0=0) |
| n^2.5 | 0.00957 | level | (3/8)·n + 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00223 | read B[i2] (i0=0) |
| n^2 | 3.91 | level | 5 | (7/4)·n^2 | 0.25 | read D[i1, i2] (i0=0); read E[i2] (i0=0) |
| n^2 | 2 | level | 4 | n^2 | 0.143 | read B[i2] (i0=0, i2=0); read A[i1] (i0=0, i2=0) (+1) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 | 0.125 | read E[i2] (i0=0, i2=0); read C[i1] (i0=0) |
| n^2 | 1.73 | level | 3 | n^2 | 0.143 | read B[i2] (i0=0, i2=0); write B[i2] (i0=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 | 0.125 | read B[i2] (i0=0) |
| n^2 | 1.41 | level | 2 | n^2 | 0.143 | write A[i1] (i0=0) |
| n^2 | 0.306 | level | 6 | (1/8)·n^2 - n | 0.0179 | read E[i2] (i0=0, i2=0); read C[i1] (i0=0) |
| n^1.5 | 0.612 | level | (3/8)·n + 2 | n - 1 | 0.143/n | read B[i2] (i0=0); read E[i2] (i0=0) |
| n^1.5 | 0.612 | level | (3/8)·n + 2 | n - 1 | 0.143/n | read B[i2] (i0=0, i2=0); read E[i2] (i0=0, i2=0) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.125/n | read B[i2] (i0=0) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.125/n | read B[i2] (i0=0, i2=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n - 1 | 0.0179/n | read E[i2] (i0=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 4 | (1/8)·n - 1 | 0.0179/n | read E[i2] (i0=0, i2=0) |
| n^1 | 2.14 | level | 6 | (7/8)·n | 0.125/n | read C[i1] (i0=0, i2=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.125/n | write A[i1] (i0=0) |

Same structure as atax: matrix streamed once, vector ramps only (d = 2.5, +0.5).

## cholesky — infinite-repeat  [`exact`]

Accesses $A(n) = (2/3)·n^3 + n^2 + (1/3)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.00347·n^4  +  0.00607·n^3.5  +  1.17·n^3  +  0.298·n^2.5  +  26.5·n^2  +  0.611·n^1.5  +  77.1·n^1  +  172·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.00268 | ramp | 39  →  (1/16)·n^2 + (1/2)·n | (49/3072)·n^3 + (-45/64)·n^2 + (479/48)·n - 45 | 0.0239 | read A[i2, i3] (i0=0) |
| n^4 | 0.00037 | ramp | 64  →  (1/16)·n^2 + (1/2)·n | (7/3072)·n^3 + (-1/8)·n^2 + (107/48)·n - 13 | 0.00342 | read A[i2, i3] (i0=0) |
| n^4 | 0.00037 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (7/3072)·n^3 + (-15/128)·n^2 + (23/12)·n - 10 | 0.00342 | read A[i2, i3] (i0=0) |
| n^4 | 4.95e-05 | ramp | 89  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/3072)·n^3 + (-3/128)·n^2 + (13/24)·n - 4 | 0.000488 | read A[i2, i3] (i0=0) |
| n^3.5 | 0.00455 | ramp | 8  →  (1/4)·n | (1/64)·n^3 + (-129/128)·n^2 + (341/16)·n - 147 | 0.0234 | read A[i1, i3] (i0=0) |
| n^3.5 | 0.00079 | ramp | 8  →  (1/4)·n | (1/384)·n^3 + (-9/64)·n^2 + (59/24)·n - 14 | 0.00391 | read A[i1, i3] (i0=0) |
| n^3.5 | 0.000736 | ramp | 9  →  (1/4)·n - 1 | (1/384)·n^3 + (-25/128)·n^2 + (229/48)·n - 38 | 0.00391 | read A[i1, i3] (i0=0) |
| n^3 | 0.253 | level | 3 | (7/48)·n^3 + (-133/32)·n^2 + (931/24)·n - 119 | 0.219 | read A[i2, i3] (i0=0) |
| n^3 | 0.221 | level | 3 | (49/384)·n^3 + (-469/128)·n^2 + (1687/48)·n - 112 | 0.191 | read A[i1, i3] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 + (-27/8)·n^2 + (125/4)·n - 98 | 0.188 | read A[i1, i2] (i0=0, i3=0); read A[i1, i2] (i0=0) |
| n^3 | 0.109 | level | 1 | (7/64)·n^3 + (-351/128)·n^2 + (369/16)·n - 65 | 0.164 | write A[i1, i2] (i0=0) |
| n^3 | 0.0631 | level | 3 | (7/192)·n^3 + (-133/128)·n^2 + (413/48)·n - 21 | 0.0547 | read A[i1, i2] (i0=0) |
| n^3 | 0.0618 | ramp | 18  →  (1/16)·n^2 + (1/2)·n | (49/128)·n^2 + (-131/16)·n + 43 | 0.574/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.0618 | ramp | 18  →  (1/16)·n^2 + (1/2)·n | (49/128)·n^2 + (-131/16)·n + 43 | 0.574/n | read A[i2, i3] (i0=0) |
| n^3 | 0.0316 | level | 3 | (7/384)·n^3 + (-15/128)·n^2 + (-5/48)·n - 1 | 0.0273 | read A[i1, i3] (i0=0) |
| n^3 | 0.0182 | level | 1 | (7/384)·n^3 + (9/64)·n^2 + (-8/3)·n + 3 | 0.0273 | write A[i1, i2] (i0=0) |
| n^3 | 0.0182 | level | 1 | (7/384)·n^3 + (-63/128)·n^2 + (175/48)·n - 7 | 0.0273 | write A[i1, i2] (i0=0) |
| n^3 | 0.0156 | level | 1 | (1/64)·n^3 + (-33/128)·n^2 + (17/16)·n | 0.0234 | write A[i1, i2] (i0=0) |
| n^3 | 0.0137 | level | (1/16)·n^2 + (1/2)·n | (7/128)·n^2 + (-39/16)·n + 27 | 0.082/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.00902 | level | 3 | (1/192)·n^3 + (-11/128)·n^2 + (77/48)·n - 11 | 0.00781 | read A[i1, i2] (i0=0) |
| n^3 | 0.00871 | ramp | 32  →  (1/16)·n^2 + (1/2)·n - 1 | (7/128)·n^2 + (-23/16)·n + 9 | 0.082/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00866 | ramp | 35  →  (1/16)·n^2 + (1/2)·n - 1 | (7/128)·n^2 + (-25/16)·n + 11 | 0.082/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.0086 | ramp | 29  →  (1/16)·n^2 + (-1/4)·n - 1 | (7/128)·n^2 + (-23/16)·n + 9 | 0.082/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.0086 | ramp | 29  →  (1/16)·n^2 + (-1/4)·n - 1 | (7/128)·n^2 + (-23/16)·n + 9 | 0.082/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00837 | ramp | 60  →  (1/16)·n^2 + (1/2)·n | (7/128)·n^2 + (-37/16)·n + 24 | 0.082/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00836 | ramp | 59  →  (1/16)·n^2 + (1/2)·n - 1 | (7/128)·n^2 + (-37/16)·n + 24 | 0.082/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00718 | ramp | 64  →  (1/16)·n^2 + (1/2)·n | (3/64)·n^2 - 2·n + 21 | 0.0703/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00521 | level | 1 | (1/192)·n^3 + (-3/128)·n^2 + (563/48)·n - 37 | 0.00781 | read A[i1, i1] (i0=0, i1=0); read A[i2, i2] (i0=0, i1=1, i2=0) (+9) |
| n^3 | 0.00195 | level | (1/16)·n^2 + (1/2)·n + (7/16) | (1/128)·n^2 + (-29/64)·n + (825/128) | 0.0117/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.00195 | level | (1/16)·n^2 + (1/2)·n | (1/128)·n^2 + (-7/16)·n + 6 | 0.0117/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.00119 | ramp | 60  →  (1/16)·n^2 + (-1/8)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00119 | ramp | 56  →  (1/16)·n^2 + (-1/4)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00118 | ramp | 54  →  (1/16)·n^2 + (-1/4)·n - 2 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.00118 | ramp | 54  →  (1/16)·n^2 + (-1/4)·n - 2 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00114 | ramp | 89  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00113 | ramp | 88  →  (1/16)·n^2 + (-1/4)·n - 2 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^2.5 | 0.114 | ramp | 6  →  (1/4)·n | (7/16)·n^2 + (-63/4)·n + 141 | 0.656/n | read A[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.0981 | ramp | 6  →  (1/4)·n | (3/8)·n^2 + (-105/8)·n + 114 | 0.562/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0168 | ramp | 6  →  (1/4)·n | (1/16)·n^2 + (-7/4)·n + 12 | 0.0938/n | read A[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.0168 | ramp | 6  →  (1/4)·n | (1/16)·n^2 + (-7/4)·n + 12 | 0.0938/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0162 | ramp | 4  →  (1/4)·n - 2 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0938/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0162 | ramp | 7  →  (1/4)·n - 1 | (1/16)·n^2 + (-21/8)·n + 27 | 0.0938/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0148 | ramp | 7  →  (1/4)·n - 1 | (3/64)·n^2 + (-15/8)·n + 18 | 0.0703/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.00247 | ramp | 7  →  (1/4)·n - 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.00234 | ramp | 8  →  (1/4)·n - 2 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0117/n | read A[i1, i4] (i0=0) |
| n^2 | 3.79 | level | 3 | (35/16)·n^2 + (-147/4)·n + 155 | 3.28/n | read A[i2, i3] (i0=0) |
| n^2 | 2.27 | level | 3 | (21/16)·n^2 + (-161/8)·n + 77 | 1.97/n | read A[i1, i3] (i0=0) |
| n^2 | 1.86 | level | 2 | (21/16)·n^2 + (-35/4)·n + 6 | 1.97/n | read A[i2, i3] (i0=0) |
| n^2 | 1.62 | level | 3 | (15/16)·n^2 + (-125/8)·n + 65 | 1.41/n | read A[i1, i2] (i0=0) |
| n^2 | 1.56 | level | 1 | (25/16)·n^2 - 15·n + 40 | 2.34/n | write A[i1, i2] (i0=0) |
| n^2 | 1.33 | level | 2 | (15/16)·n^2 - 10·n + 30 | 1.41/n | read A[i1, i2] (i0=0) |
| n^2 | 1.33 | level | 2 | (15/16)·n^2 + (-13/4)·n + 2 | 1.41/n | read A[i2, i2] (i0=0); read A[i1, i2] (i0=0) |
| n^2 | 1.31 | level | 1 | (21/16)·n^2 + (-97/8)·n + 33 | 1.97/n | read A[i1, i3] (i0=0) |
| n^2 | 1.08 | level | 3 | (5/8)·n^2 + (-55/8)·n + 15 | 0.938/n | read A[i1, i2] (i0=0) |
| n^2 | 1.06 | level | 2 | (3/4)·n^2 + (-9/4)·n | 1.12/n | read A[i1, i2] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 + (-63/8)·n + 28 | 1.31/n | write A[i1, i2] (i0=0); write A[i1, i1] (i0=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-63/8)·n + 35 | 0.656/n | read A[i1, i3] (i0=0) |
| n^2 | 0.707 | level | 2 | (1/2)·n^2 + (-19/2)·n + 46 | 0.75/n | read A[i1, i1] (i0=0, i1=0); read A[i1, i2] (i0=0, i1=1, i2=0) (+2) |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-57/8)·n + 33 | 0.562/n | read A[i1, i2] (i0=0) |
| n^2 | 0.56 | ramp | 6  →  (1/16)·n^2 + (3/8)·n + 1 | (35/8)·n - 30 | 6.56·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.541 | level | 2 | (49/128)·n^2 + (-105/16)·n + 28 | 0.574/n | read A[i1, i4] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 + (-15/8)·n + 2 | 0.656/n | read A[i1, i3] (i0=0) |
| n^2 | 0.375 | level | 1 | (3/8)·n^2 + (-9/8)·n | 0.562/n | write A[i1, i2] (i0=0) |
| n^2 | 0.336 | ramp | 5  →  (1/16)·n^2 + (3/8)·n + 1 | (21/8)·n - 26 | 3.94·n^-2 | read A[i2, i3] (i0=0, i2=1, i3=0); read A[i2, i3] (i0=0, i2=7, i3=0) (+1) |
| n^2 | 0.312 | level | 1 | (5/16)·n^2 + (-5/8)·n | 0.469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.312 | level | 1 | (5/16)·n^2 + (-45/8)·n + 25 | 0.469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.219 | level | (1/16)·n^2 + (1/2)·n | (7/8)·n - 9 | 1.31·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.219 | level | (1/16)·n^2 + (1/2)·n | (7/8)·n - 16 | 1.31·n^-2 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^2 | 0.219 | level | (1/16)·n^2 + (1/2)·n | (7/8)·n - 16 | 1.31·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.188 | level | 1 | (3/16)·n^2 + (-15/8)·n + 3 | 0.281/n | write A[i1, i2] (i0=0) |
| n^2 | 0.128 | ramp | 29  →  (1/16)·n^2 + (1/2)·n | n - 17 | 1.5·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 + (-9/8)·n + 8 | 0.188/n | write A[i1, i2] (i0=0); write A[i1, i1] (i0=0) |
| n^2 | 0.113 | ramp | 14  →  (1/16)·n^2 + (1/2)·n | (7/8)·n - 8 | 1.31·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.112 | ramp | 16  →  (1/16)·n^2 + (1/2)·n | (7/8)·n - 9 | 1.31·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.112 | ramp | 33  →  (1/16)·n^2 + (1/2)·n | (7/8)·n - 15 | 1.31·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.112 | ramp | 12  →  (1/16)·n^2 + (3/8)·n | (7/8)·n - 8 | 1.31·n^-2 | read A[i2, i2] (i0=0, i2=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0938/n | read A[i1, i3] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0938/n | read A[i1, i2] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0938/n | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-13/8)·n + 11 | 0.0938/n | read A[i1, i2] (i0=0, i1=1, i2=0); read A[i1, i2] (i0=0, i1=2, i2=0) (+2) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0938/n | read A[i1, i2] (i0=0) |
| n^2 | 0.0967 | ramp | 16  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 7 | 1.12·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0964 | ramp | 36  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 13 | 1.12·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0964 | ramp | 36  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 13 | 1.12·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0964 | ramp | 36  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 13 | 1.12·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (1/4)·n | 0.0938/n | read A[i1, i2] (i0=0, i2=1, i3=0); read A[i1, i2] (i0=0) |
| n^2 | 0.0786 | ramp | 11  →  (1/16)·n^2 + (-3/8)·n + 1 | (5/8)·n - 5 | 0.938·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0773 | level | 2 | (7/128)·n^2 + (-21/16)·n + 7 | 0.082/n | read A[i1, i4] (i0=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0938/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0938/n | write A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.047 | ramp | 11  →  (1/16)·n^2 + (-3/8)·n + 1 | (3/8)·n - 6 | 0.562·n^-2 | read A[i2, i3] (i0=0, i2=1, i3=0); read A[i2, i3] (i0=0, i2=7, i3=0) (+1) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n + (7/16) | (1/8)·n + (-17/8) | 0.188·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n + (7/16) | (1/8)·n + (-33/8) | 0.188·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n + (3/4) | (1/8)·n + (-13/4) | 0.188·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n | (1/8)·n - 4 | 0.188·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n + (7/16) | (1/8)·n + (-25/8) | 0.188·n^-2 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n | (1/8)·n - 3 | 0.188·n^-2 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n | (1/8)·n - 4 | 0.188·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n + (7/16) | (1/8)·n + (-33/8) | 0.188·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n | (1/8)·n - 4 | 0.188·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n + (7/16) | (1/8)·n + (-25/8) | 0.188·n^-2 | read A[i1, i1] (i0=0, i4=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (1/2)·n | (1/8)·n - 3 | 0.188·n^-2 | read A[i1, i1] (i0=0, i4=0) |
| n^2 | 0.0159 | ramp | 14  →  (1/16)·n^2 + (-1/8)·n | (1/8)·n - 1 | 0.188·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0159 | ramp | 33  →  (1/16)·n^2 + (-1/8)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0159 | ramp | 32  →  (1/16)·n^2 + (-1/8)·n - 1 | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0158 | ramp | 12  →  (1/16)·n^2 + (-1/4)·n | (1/8)·n - 1 | 0.188·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0158 | ramp | 30  →  (1/16)·n^2 + (-1/4)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0158 | ramp | 29  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0157 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 3 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0157 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 3 | 0.188·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0157 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 3 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0157 | ramp | 27  →  (1/16)·n^2 + (-3/8)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i2] (i0=0, i2=0) |
| n^1.5 | 0.291 | ramp | 6  →  (1/4)·n | (7/8)·n - 15 | 1.31·n^-2 | read A[i1, i4] (i0=0, i4=0) |
| n^1.5 | 0.187 | ramp | 5  →  (1/8)·n + 2 | (3/4)·n - 12 | 1.12·n^-2 | read A[i1, i4] (i0=0) |
| n^1.5 | 0.0418 | ramp | 6  →  (1/4)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i1, i4] (i0=0, i4=0) |
| n^1.5 | 0.0312 | ramp | 5  →  (1/8)·n + 2 | (1/8)·n - 2 | 0.188·n^-2 | read A[i1, i4] (i0=0) |
| n^1.5 | 0.0308 | ramp | 6  →  (1/8)·n + 2 | (1/8)·n - 3 | 0.188·n^-2 | read A[i1, i4] (i0=0) |
| n^1.5 | 0.0296 | ramp | 3  →  (1/8)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 14 | level | 4 | 7·n - 98 | 10.5·n^-2 | read A[i1, i3] (i0=0, i3=0) |
| n^1 | 7.58 | level | 3 | (35/8)·n - 36 | 6.56·n^-2 | read A[i2, i3] (i0=0) |
| n^1 | 7.07 | level | 2 | 5·n - 30 | 7.5·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 5.25 | level | 1 | (21/4)·n - 21 | 7.88·n^-2 | write A[i1, i1] (i0=0) |
| n^1 | 3.71 | level | 2 | (21/8)·n - 6 | 3.94·n^-2 | read A[i2, i3] (i0=0) |
| n^1 | 3.71 | level | 2 | (21/8)·n - 21 | 3.94·n^-2 | read A[i1, i1] (i0=0) |
| n^1 | 2.83 | level | 2 | 2·n - 16 | 3·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 2.62 | level | 1 | (21/8)·n | 3.94·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 2.62 | level | 1 | (21/8)·n | 3.94·n^-2 | read A[i1, i1] (i0=0) |
| n^1 | 2.12 | level | 2 | (3/2)·n - 12 | 2.25·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 2 | level | 4 | n - 8 | 1.5·n^-2 | read A[i1, i2] (i0=0, i1=2, i2=0); read A[i2, i3] (i0=0, i1=3, i2=1, i3=0) (+2) |
| n^1 | 1.75 | level | (1/16)·n^2 + (1/2)·n | 7 | 10.5·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 1.5 | level | 1 | (3/2)·n - 12 | 2.25·n^-2 | read A[i1, i1] (i0=0); write A[i1, i1] (i0=0) |
| n^1 | 1.41 | level | 2 | n - 9 | 1.5·n^-2 | read A[i1, i1] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 5 | 7.5·n^-3 | read A[i1, i1] (i0=0, i1=0); read A[i1, i2] (i0=0, i1=1, i2=0) (+3) |
| n^1 | 1.25 | level | (1/16)·n^2 + (1/2)·n | 5 | 7.5·n^-3 | read A[i1, i1] (i0=0, i1=0); read A[i1, i2] (i0=0, i1=1, i2=0) (+3) |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 1.31·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 1.31·n^-2 | read A[i1, i1] (i0=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n + (-63/8) | 1.31·n^-2 | read A[i1, i1] (i0=0, i4=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 1.31·n^-2 | read A[i1, i1] (i0=0, i4=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n - 1 | 1.31·n^-2 | read A[i2, i2] (i0=0) |
| n^1 | 1 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 4 | 6·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 1 | level | (1/16)·n^2 + (1/2)·n | 4 | 6·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n - 7 | 1.31·n^-2 | write A[i1, i1] (i0=0) |
| n^1 | 0.75 | level | 1 | (3/4)·n - 6 | 1.12·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 0.5 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 2 | 3·n^-3 | read A[i1, i2] (i0=0, i1=16, i3=0); read A[i1, i1] (i0=0, i1=8, i4=0) |
| n^1 | 0.5 | level | (1/16)·n^2 + (1/2)·n | 2 | 3·n^-3 | read A[i1, i2] (i0=0, i1=16, i3=0); read A[i1, i1] (i0=0, i1=8, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (3/4) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (3/4) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (3/4) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 1 | 1.5·n^-3 | read A[i1, i1] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i1] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (3/4) | 1 | 1.5·n^-3 | read A[i1, i1] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n | 1 | 1.5·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | 1 | (1/4)·n - 2 | 0.375·n^-2 | read A[i1, i1] (i0=0); write A[i1, i1] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (7/16) | 1 | 1.5·n^-3 | read A[i1, i1] (i0=0, i4=0) |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.188·n^-2 | read A[i1, i4] (i0=0) |
| n^0 | 14 | level | 4 | 7 | 10.5·n^-3 | read A[i1, i4] (i0=0, i4=0) |
| n^0 | 6.63 | level | 11 | 2 | 3·n^-3 | read A[i2, i3] (i0=0, i1=9, i2=7, i3=0); read A[i2, i3] (i0=0, i1=9, i2=8, i3=0) |
| n^0 | 6 | level | 1 | 6 | 9·n^-3 | read A[i1, i1] (i0=0, i4=0) |
| n^0 | 5.39 | level | 29 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i1=17, i2=15, i3=0) |
| n^0 | 5.29 | level | 28 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i1=17, i2=8, i3=0) |
| n^0 | 5.1 | level | 26 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 5.1 | level | 26 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 5 | level | 25 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.9 | level | 24 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.9 | level | 24 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.8 | level | 23 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.69 | level | 22 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.69 | level | 22 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.58 | level | 21 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.47 | level | 20 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.47 | level | 20 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.36 | level | 19 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.24 | level | 18 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.24 | level | 18 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.12 | level | 17 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4 | level | 16 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4 | level | 16 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 3.87 | level | 15 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 3.74 | level | 14 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 3.61 | level | 13 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i1=10, i2=8, i3=0) |
| n^0 | 3.16 | level | 10 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=9, i2=0) |
| n^0 | 3 | level | 9 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 3 | level | 9 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.83 | level | 8 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.83 | level | 8 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.83 | level | 8 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2.65 | level | 7 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.65 | level | 7 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.65 | level | 7 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2.45 | level | 6 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.45 | level | 6 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.45 | level | 6 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2.24 | level | 5 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.24 | level | 5 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.24 | level | 5 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2 | level | 4 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2 | level | 4 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |

The factorization re-reads the trailing triangle: `read A[i2,i3]` ramps from tens of lines up to (1/16)n^2 + (1/2)n (the triangular footprint), population ~n^3/60, coefficient 0.0027 + smaller siblings — headroom +1.0 with the *smallest* leading coefficients in the suite. The term list quantifies exactly how latent cholesky's tiling payoff is: the n^4 term exists, but its boundary crossover against the n^3 bulk sits far beyond the sizes any flat-miss-curve sweep reaches.

## cholesky — single-shot  [`exact`]

Accesses $A(n) = (2/3)·n^3 + n^2 + (1/3)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.00347·n^4  +  0.00607·n^3.5  +  1.15·n^3  +  0.298·n^2.5  +  27.6·n^2  +  0.611·n^1.5  +  53.7·n^1  +  173·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.00268 | ramp | 39  →  (1/16)·n^2 + (1/2)·n | (49/3072)·n^3 + (-45/64)·n^2 + (479/48)·n - 45 | 0.0239 | read A[i2, i3] (i0=0) |
| n^4 | 0.00037 | ramp | 64  →  (1/16)·n^2 + (1/2)·n | (7/3072)·n^3 + (-1/8)·n^2 + (107/48)·n - 13 | 0.00342 | read A[i2, i3] (i0=0) |
| n^4 | 0.00037 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (7/3072)·n^3 + (-15/128)·n^2 + (23/12)·n - 10 | 0.00342 | read A[i2, i3] (i0=0) |
| n^4 | 4.95e-05 | ramp | 89  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/3072)·n^3 + (-3/128)·n^2 + (13/24)·n - 4 | 0.000488 | read A[i2, i3] (i0=0) |
| n^3.5 | 0.00455 | ramp | 8  →  (1/4)·n | (1/64)·n^3 + (-129/128)·n^2 + (341/16)·n - 147 | 0.0234 | read A[i1, i3] (i0=0) |
| n^3.5 | 0.00079 | ramp | 8  →  (1/4)·n | (1/384)·n^3 + (-9/64)·n^2 + (59/24)·n - 14 | 0.00391 | read A[i1, i3] (i0=0) |
| n^3.5 | 0.000736 | ramp | 9  →  (1/4)·n - 1 | (1/384)·n^3 + (-25/128)·n^2 + (229/48)·n - 38 | 0.00391 | read A[i1, i3] (i0=0) |
| n^3 | 0.505 | level | 3 | (7/24)·n^3 + (-133/16)·n^2 + (931/12)·n - 238 | 0.438 | read A[i1, i3] (i0=0); read A[i2, i3] (i0=0) |
| n^3 | 0.289 | level | 3 | (1/6)·n^3 - 5·n^2 + (299/6)·n - 164 | 0.25 | read A[i2, i3] (i0=0, i1=2, i2=1, i3=0); read A[i1, i2] (i0=0) |
| n^3 | 0.146 | level | 1 | (7/48)·n^3 + (-119/32)·n^2 + (371/12)·n - 84 | 0.219 | write A[i1, i2] (i0=0) |
| n^3 | 0.0618 | ramp | 18  →  (1/16)·n^2 + (1/2)·n | (49/128)·n^2 + (-131/16)·n + 43 | 0.574/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.0618 | ramp | 18  →  (1/16)·n^2 + (1/2)·n | (49/128)·n^2 + (-131/16)·n + 43 | 0.574/n | read A[i2, i3] (i0=0) |
| n^3 | 0.0208 | level | 1 | (1/48)·n^3 + (-9/32)·n^2 + (11/12)·n | 0.0312 | write A[i1, i2] (i0=0) |
| n^3 | 0.00871 | ramp | 32  →  (1/16)·n^2 + (1/2)·n - 1 | (7/128)·n^2 + (-23/16)·n + 9 | 0.082/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00866 | ramp | 35  →  (1/16)·n^2 + (1/2)·n - 1 | (7/128)·n^2 + (-25/16)·n + 11 | 0.082/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.0086 | ramp | 29  →  (1/16)·n^2 + (-1/4)·n - 1 | (7/128)·n^2 + (-23/16)·n + 9 | 0.082/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.0086 | ramp | 29  →  (1/16)·n^2 + (-1/4)·n - 1 | (7/128)·n^2 + (-23/16)·n + 9 | 0.082/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00837 | ramp | 60  →  (1/16)·n^2 + (1/2)·n | (7/128)·n^2 + (-37/16)·n + 24 | 0.082/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00836 | ramp | 59  →  (1/16)·n^2 + (1/2)·n - 1 | (7/128)·n^2 + (-37/16)·n + 24 | 0.082/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00718 | ramp | 64  →  (1/16)·n^2 + (1/2)·n | (3/64)·n^2 - 2·n + 21 | 0.0703/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00119 | ramp | 60  →  (1/16)·n^2 + (-1/8)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00119 | ramp | 56  →  (1/16)·n^2 + (-1/4)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00118 | ramp | 54  →  (1/16)·n^2 + (-1/4)·n - 2 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i3] (i0=0, i3=0) |
| n^3 | 0.00118 | ramp | 54  →  (1/16)·n^2 + (-1/4)·n - 2 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00114 | ramp | 89  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^3 | 0.00113 | ramp | 88  →  (1/16)·n^2 + (-1/4)·n - 2 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0117/n | read A[i2, i3] (i0=0) |
| n^2.5 | 0.114 | ramp | 6  →  (1/4)·n | (7/16)·n^2 + (-63/4)·n + 141 | 0.656/n | read A[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.0981 | ramp | 6  →  (1/4)·n | (3/8)·n^2 + (-105/8)·n + 114 | 0.562/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0168 | ramp | 6  →  (1/4)·n | (1/16)·n^2 + (-7/4)·n + 12 | 0.0938/n | read A[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.0168 | ramp | 6  →  (1/4)·n | (1/16)·n^2 + (-7/4)·n + 12 | 0.0938/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0162 | ramp | 4  →  (1/4)·n - 2 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0938/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0162 | ramp | 7  →  (1/4)·n - 1 | (1/16)·n^2 + (-21/8)·n + 27 | 0.0938/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0148 | ramp | 7  →  (1/4)·n - 1 | (3/64)·n^2 + (-15/8)·n + 18 | 0.0703/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.00247 | ramp | 7  →  (1/4)·n - 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0117/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.00234 | ramp | 8  →  (1/4)·n - 2 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0117/n | read A[i1, i4] (i0=0) |
| n^2 | 3.79 | level | 3 | (35/16)·n^2 + (-259/8)·n + 119 | 3.28/n | read A[i2, i3] (i0=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 + (-49/2)·n + 84 | 2.62/n | read A[i1, i3] (i0=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 + (-49/2)·n + 84 | 2.62/n | read A[i1, i2] (i0=0) |
| n^2 | 2.62 | level | 1 | (21/8)·n^2 + (-189/8)·n + 56 | 3.94/n | write A[i1, i2] (i0=0) |
| n^2 | 2.47 | level | 2 | (7/4)·n^2 - 14·n + 35 | 2.62/n | read A[i1, i2] (i0=0) |
| n^2 | 1.86 | level | 2 | (21/16)·n^2 + (-49/8)·n | 1.97/n | read A[i2, i3] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 - 14·n + 35 | 2.62/n | read A[i1, i3] (i0=0) |
| n^2 | 1.41 | level | 2 | n^2 - 11·n + 47 | 1.5/n | read A[i2, i2] (i0=0, i1=2, i2=0); read A[i1, i2] (i0=0) (+1) |
| n^2 | 0.866 | level | 3 | (1/2)·n^2 + (-19/2)·n + 45 | 0.75/n | read A[i1, i2] (i0=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-63/8)·n + 35 | 0.656/n | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-63/8)·n + 35 | 0.656/n | read A[i1, i3] (i0=0) |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-63/8)·n + 35 | 0.656/n | read A[i1, i4] (i0=0) |
| n^2 | 0.56 | ramp | 6  →  (1/16)·n^2 + (3/8)·n + 1 | (35/8)·n - 30 | 6.56·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.541 | level | 2 | (49/128)·n^2 + (-7/16)·n | 0.574/n | read A[i2, i2] (i0=0) |
| n^2 | 0.5 | level | 1 | (1/2)·n^2 + (19/2)·n - 36 | 0.75/n | read A[i2, i2] (i0=0, i1=1, i2=0); read A[i1, i3] (i0=0, i3=0) (+4) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 - 7·n + 28 | 0.656/n | write A[i1, i2] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 - 7·n + 28 | 0.656/n | write A[i1, i1] (i0=0) |
| n^2 | 0.375 | level | 1 | (3/8)·n^2 + (-3/8)·n | 0.562/n | write A[i1, i2] (i0=0) |
| n^2 | 0.336 | ramp | 5  →  (1/16)·n^2 + (3/8)·n + 1 | (21/8)·n - 26 | 3.94·n^-2 | read A[i2, i3] (i0=0, i2=1, i3=0); read A[i2, i3] (i0=0, i2=7, i3=0) (+1) |
| n^2 | 0.128 | ramp | 29  →  (1/16)·n^2 + (1/2)·n | n - 17 | 1.5·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.113 | ramp | 14  →  (1/16)·n^2 + (1/2)·n | (7/8)·n - 8 | 1.31·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.112 | ramp | 16  →  (1/16)·n^2 + (1/2)·n | (7/8)·n - 9 | 1.31·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.112 | ramp | 33  →  (1/16)·n^2 + (1/2)·n | (7/8)·n - 15 | 1.31·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.112 | ramp | 12  →  (1/16)·n^2 + (3/8)·n | (7/8)·n - 8 | 1.31·n^-2 | read A[i2, i2] (i0=0, i2=0) |
| n^2 | 0.0967 | ramp | 16  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 7 | 1.12·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0964 | ramp | 36  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 13 | 1.12·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0964 | ramp | 36  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 13 | 1.12·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0964 | ramp | 36  →  (1/16)·n^2 + (1/2)·n | (3/4)·n - 13 | 1.12·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0786 | ramp | 11  →  (1/16)·n^2 + (-3/8)·n + 1 | (5/8)·n - 5 | 0.938·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0773 | level | 2 | (7/128)·n^2 + (-7/16)·n | 0.082/n | read A[i2, i2] (i0=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-1/2)·n | 0.0938/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-1/2)·n | 0.0938/n | write A[i1, i1] (i0=0) |
| n^2 | 0.047 | ramp | 11  →  (1/16)·n^2 + (-3/8)·n + 1 | (3/8)·n - 6 | 0.562·n^-2 | read A[i2, i3] (i0=0, i2=1, i3=0); read A[i2, i3] (i0=0, i2=7, i3=0) (+1) |
| n^2 | 0.0159 | ramp | 14  →  (1/16)·n^2 + (-1/8)·n | (1/8)·n - 1 | 0.188·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0159 | ramp | 33  →  (1/16)·n^2 + (-1/8)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0159 | ramp | 32  →  (1/16)·n^2 + (-1/8)·n - 1 | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0158 | ramp | 12  →  (1/16)·n^2 + (-1/4)·n | (1/8)·n - 1 | 0.188·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0158 | ramp | 30  →  (1/16)·n^2 + (-1/4)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0158 | ramp | 29  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0157 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 3 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0157 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 3 | 0.188·n^-2 | read A[i2, i3] (i0=0, i3=0) |
| n^2 | 0.0157 | ramp | 55  →  (1/16)·n^2 + (-1/4)·n - 1 | (1/8)·n - 3 | 0.188·n^-2 | read A[i2, i3] (i0=0) |
| n^2 | 0.0157 | ramp | 27  →  (1/16)·n^2 + (-3/8)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i2, i2] (i0=0, i2=0) |
| n^1.5 | 0.291 | ramp | 6  →  (1/4)·n | (7/8)·n - 15 | 1.31·n^-2 | read A[i1, i4] (i0=0, i4=0) |
| n^1.5 | 0.187 | ramp | 5  →  (1/8)·n + 2 | (3/4)·n - 12 | 1.12·n^-2 | read A[i1, i4] (i0=0) |
| n^1.5 | 0.0418 | ramp | 6  →  (1/4)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i1, i4] (i0=0, i4=0) |
| n^1.5 | 0.0312 | ramp | 5  →  (1/8)·n + 2 | (1/8)·n - 2 | 0.188·n^-2 | read A[i1, i4] (i0=0) |
| n^1.5 | 0.0308 | ramp | 6  →  (1/8)·n + 2 | (1/8)·n - 3 | 0.188·n^-2 | read A[i1, i4] (i0=0) |
| n^1.5 | 0.0296 | ramp | 3  →  (1/8)·n | (1/8)·n - 2 | 0.188·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 14 | level | 4 | 7·n - 98 | 10.5·n^-2 | read A[i1, i3] (i0=0, i3=0) |
| n^1 | 9.9 | level | 2 | 7·n - 35 | 10.5·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 5.25 | level | 1 | (21/4)·n - 21 | 7.88·n^-2 | write A[i1, i1] (i0=0) |
| n^1 | 4.95 | level | 2 | (7/2)·n - 28 | 5.25·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 4.95 | level | 2 | (7/2)·n - 28 | 5.25·n^-2 | read A[i1, i1] (i0=0) |
| n^1 | 3.5 | level | 1 | (7/2)·n - 7 | 5.25·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 3.5 | level | 1 | (7/2)·n - 7 | 5.25·n^-2 | read A[i1, i1] (i0=0) |
| n^1 | 2 | level | 4 | n - 8 | 1.5·n^-2 | read A[i2, i3] (i0=0, i1=3, i2=1, i3=0); read A[i1, i4] (i0=0, i1=9) (+1) |
| n^1 | 1.41 | level | 2 | n - 9 | 1.5·n^-2 | read A[i1, i1] (i0=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 1.31·n^-2 | read A[i1, i4] (i0=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 1.31·n^-2 | read A[i1, i1] (i0=0, i4=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n - 7 | 1.31·n^-2 | write A[i1, i1] (i0=0) |
| n^1 | 0.75 | level | 1 | (3/4)·n | 1.12·n^-2 | write A[i1, i1] (i0=0) |
| n^1 | 0.125 | level | 1 | (1/8)·n | 0.188·n^-2 | write A[i1, i1] (i0=0) |
| n^0 | 14 | level | 4 | 7 | 10.5·n^-3 | read A[i1, i4] (i0=0, i4=0) |
| n^0 | 7 | level | 1 | 7 | 10.5·n^-3 | read A[i1, i1] (i0=0, i4=0) |
| n^0 | 6.63 | level | 11 | 2 | 3·n^-3 | read A[i2, i3] (i0=0, i1=9, i2=7, i3=0); read A[i2, i3] (i0=0, i1=9, i2=8, i3=0) |
| n^0 | 5.39 | level | 29 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i1=17, i2=15, i3=0) |
| n^0 | 5.29 | level | 28 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i1=17, i2=8, i3=0) |
| n^0 | 5.1 | level | 26 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 5.1 | level | 26 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 5 | level | 25 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.9 | level | 24 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.9 | level | 24 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.8 | level | 23 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.69 | level | 22 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.69 | level | 22 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.58 | level | 21 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.47 | level | 20 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.47 | level | 20 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.36 | level | 19 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4.24 | level | 18 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.24 | level | 18 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4.12 | level | 17 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 4 | level | 16 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 4 | level | 16 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 3.87 | level | 15 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i2=8, i3=0) |
| n^0 | 3.74 | level | 14 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 3.61 | level | 13 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i1=10, i2=8, i3=0) |
| n^0 | 3.16 | level | 10 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=9, i2=0) |
| n^0 | 3 | level | 9 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 3 | level | 9 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.83 | level | 8 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.83 | level | 8 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.83 | level | 8 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2.65 | level | 7 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.65 | level | 7 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.65 | level | 7 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2.45 | level | 6 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.45 | level | 6 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.45 | level | 6 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2.24 | level | 5 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.24 | level | 5 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2.24 | level | 5 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 2 | level | 4 | 1 | 1.5·n^-3 | read A[i2, i3] (i0=0, i3=0) |
| n^0 | 2 | level | 4 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0) |

The factorization re-reads the trailing triangle: `read A[i2,i3]` ramps from tens of lines up to (1/16)n^2 + (1/2)n (the triangular footprint), population ~n^3/60, coefficient 0.0027 + smaller siblings — headroom +1.0 with the *smallest* leading coefficients in the suite. The term list quantifies exactly how latent cholesky's tiling payoff is: the n^4 term exists, but its boundary crossover against the n^3 bulk sits far beyond the sizes any flat-miss-curve sweep reaches.

## convolution — infinite-repeat  [`exact`]

Accesses $A(n) = 163·n^2 - 2934·n + 13203$ (exact on n ≡ 0 mod 16); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=512, 1 at n=528.

**DMD spectrum:**  0.125·n^3  +  1.12·n^2.5  +  399·n^2  +  35.8·n^1.5  +  199·n^1  +  48·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0625 | level | (1/4)·n^2 + (-13/4)·n + 34 | (1/8)·n^2 + (-51/8)·n + 76 | 0.000767 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 + (-9/4)·n + 27 | (1/8)·n^2 + (-33/8)·n + 27 | 0.000767 | write C[i1, i2] (i0=0, i1=0); write C[i1, i2] (i0=0) |
| n^2.5 | 1.12 | level | (5/4)·n + 15 | n^2 - 42·n + 320 | 0.00613 | read B[i1 + i3, i2 + i4] (i0=0, i3=0); read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 89.1 | level | 2 | 63·n^2 - 1134·n + 5103 | 0.387 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 89.1 | level | 2 | 63·n^2 - 1134·n + 5103 | 0.387 | read A[i3, i4] (i0=0) |
| n^2 | 37.3 | level | 37 | (49/8)·n^2 + (-889/8)·n + 504 | 0.0376 | read A[i3, i4] (i0=0, i1=0); read A[i3, i4] (i0=0) |
| n^2 | 36.8 | level | 36 | (49/8)·n^2 + (-945/8)·n + 567 | 0.0376 | read A[i3, i4] (i0=0, i1=0, i4=0); read A[i3, i4] (i0=0, i4=0) |
| n^2 | 36.8 | level | 36 | (49/8)·n^2 + (-889/8)·n + 504 | 0.0376 | read B[i1 + i3, i2 + i4] (i0=0, i1=0); read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 36.5 | level | 37 | 6·n^2 - 110·n + 504 | 0.0368 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=8, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i1=0, i4=0) (+2) |
| n^2 | 6.08 | level | 37 | n^2 - 17·n + 72 | 0.00613 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=8, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i1=0, i4=0) (+2) |
| n^2 | 5.92 | level | 35 | n^2 - 25·n + 144 | 0.00613 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=8, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i1=0, i4=0) (+2) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-119/8)·n + 63 | 0.00537 | read A[i3, i4] (i0=0, i1=0, i4=0); read A[i3, i4] (i0=0, i4=0) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read A[i3, i4] (i0=0, i1=0); read A[i3, i4] (i0=0) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read A[i3, i4] (i0=0, i1=0, i3=0); read A[i3, i4] (i0=0, i3=0) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | write C[i1, i2] (i0=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-135/8)·n + 81 | 0.00537 | read A[i3, i4] (i0=0, i1=0, i4=0); read A[i3, i4] (i0=0, i4=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-135/8)·n + 81 | 0.00537 | read A[i3, i4] (i0=0, i1=0, i3=0, i4=0); read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0); read B[i1 + i3, i2 + i4] (i0=0, i3=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read B[i1 + i3, i2 + i4] (i0=0, i1=0); read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-175/8)·n + 126 | 0.00537 | read A[i3, i4] (i0=0, i1=0); read A[i3, i4] (i0=0) |
| n^2 | 4.56 | level | 37 | (3/4)·n^2 + (-55/4)·n + 63 | 0.0046 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.76 | level | 37 | (1/8)·n^2 + (-17/8)·n + 9 | 0.000767 | read A[i3, i4] (i0=0, i1=0, i4=0); read A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.76 | level | 37 | (1/8)·n^2 + (-17/8)·n + 9 | 0.000767 | read A[i3, i4] (i0=0, i1=0, i3=0, i4=0); read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.76 | level | 37 | (1/8)·n^2 + (-17/8)·n + 9 | 0.000767 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.75 | level | 36 | (1/8)·n^2 + (-25/8)·n + 18 | 0.000767 | read A[i3, i4] (i0=0, i1=0); read A[i3, i4] (i0=0) |
| n^2 | 0.75 | level | 36 | (1/8)·n^2 + (-25/8)·n + 18 | 0.000767 | read A[i3, i4] (i0=0, i1=0, i3=0); read A[i3, i4] (i0=0, i3=0) |
| n^2 | 0.74 | level | 35 | (1/8)·n^2 + (-25/8)·n + 18 | 0.000767 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 34 | n - 19 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 34 | n - 19 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 35 | n - 18 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i1=0); read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 27 | n - 9 | 0.00613/n | write C[i1, i2] (i0=0, i1=0); write C[i1, i2] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-3/4)·n + (39/2) | n - 9 | 0.00613/n | write C[i1, i2] (i0=0, i1=0); write C[i1, i2] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 27 | n - 9 | 0.00613/n | write C[i1, i2] (i0=0, i1=0, i2=0); write C[i1, i2] (i0=0, i2=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 35 | n - 19 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^2 | 0.411 | ramp | (1/4)·n^2 - 4·n + 54  →  (1/4)·n^2 + (-19/8)·n + 25 | (7/8)·n - 28 | 0.00537/n | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^2 | 0.411 | ramp | (1/4)·n^2 + (-17/4)·n + 58  →  (1/4)·n^2 + (-5/2)·n + 24 | (7/8)·n - 28 | 0.00537/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 0.125 | level | (1/4)·n^2 + (-9/4)·n + 26 | (1/4)·n - 8 | 0.00153/n | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0); read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 0.0588 | ramp | (1/4)·n^2 + (-5/2)·n + 30  →  (1/4)·n^2 + (-19/8)·n + 25 | (1/8)·n - 4 | 0.000767/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 0.0586 | ramp | (1/4)·n^2 + (-17/4)·n + 58  →  (1/4)·n^2 + (-13/4)·n + 18 | (1/8)·n - 4 | 0.000767/n | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1.5 | 7.83 | level | (5/4)·n + 16 | 7·n - 70 | 0.0429/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i3=0, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1.5 | 6.71 | level | (5/4)·n + 16 | 6·n - 60 | 0.0368/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 8 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 9 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 10 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 11 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 12 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 13 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 14 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 7 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i3=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 8 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 9 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 10 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 11 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 12 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 13 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 7 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 14 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i3=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 16 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 16 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i3=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 16 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 48.7 | level | 37 | 8·n - 80 | 0.0491/n | read A[i3, i4] (i0=0, i2=0, i3=0, i4=0); read A[i3, i4] (i0=0, i2=0, i4=0) |
| n^1 | 36.5 | level | 37 | 6·n - 53 | 0.0368/n | read A[i3, i4] (i0=0, i1=0, i2=0, i4=8); read A[i3, i4] (i0=0, i2=0, i4=8) |
| n^1 | 36 | level | 36 | 6·n - 60 | 0.0368/n | read A[i3, i4] (i0=0, i2=0, i4=0) |
| n^1 | 6.08 | level | 37 | n - 1 | 0.00613/n | read A[i3, i4] (i0=0, i1=0, i2=0, i3=0, i4=0); read A[i3, i4] (i0=0, i1=0, i2=0, i3=0, i4=8) (+5) |
| n^1 | 6.08 | level | 37 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i4=8) |
| n^1 | 6.08 | level | 37 | n - 9 | 0.00613/n | read A[i3, i4] (i0=0, i1=0, i2=0, i4=8); read A[i3, i4] (i0=0, i2=0, i4=8) |
| n^1 | 6.08 | level | 37 | n - 9 | 0.00613/n | read A[i3, i4] (i0=0, i1=0, i2=0, i3=0, i4=8); read A[i3, i4] (i0=0, i2=0, i3=0, i4=8) |
| n^1 | 6 | level | 36 | n - 9 | 0.00613/n | read A[i3, i4] (i0=0, i1=0, i2=0, i4=0); read A[i3, i4] (i0=0, i1=0, i2=0, i4=8) (+2) |
| n^1 | 6 | level | 36 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i4=0) |
| n^1 | 6 | level | 36 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 4·n + 47 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-15/4)·n + 44 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-7/2)·n + 41 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 38 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 35 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 32 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 29 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-17/4)·n + 50 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-25/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-23/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-21/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-19/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i3=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-25/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-23/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-21/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-19/8)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-17/4)·n + 50 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 4·n + 47 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-15/4)·n + 44 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-7/2)·n + 41 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 38 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 35 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 32 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 29 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 26 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-25/8)·n + 34 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 33 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-23/8)·n + 32 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 31 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-21/8)·n + 30 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 29 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-19/8)·n + 28 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 27 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + (113/4) | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i3=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-17/4)·n + 51 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 4·n + 48 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-15/4)·n + 45 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-7/2)·n + 42 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 39 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 36 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 33 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 30 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 27 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 4·n + 48 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-15/4)·n + 45 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-7/2)·n + 42 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 39 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 36 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 33 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 30 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-17/4)·n + 51 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-13/4)·n + 35 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-25/8)·n + 34 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 - 3·n + 33 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-23/8)·n + 32 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-11/4)·n + 31 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-21/8)·n + 30 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/2)·n + 29 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-19/8)·n + 28 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 27 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i1=0, i2=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 27 | 1 | 0.00613·n^-2 | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^0 | 42 | level | 36 | 7 | 0.0429·n^-2 | read A[i3, i4] (i0=0, i1=0, i2=0, i4=0); read A[i3, i4] (i0=0, i1=0, i2=0, i4=8) (+1) |
| n^0 | 6 | level | 36 | 1 | 0.00613·n^-2 | read A[i3, i4] (i0=0, i1=0, i2=0, i3=0, i4=0); read A[i3, i4] (i0=0, i1=0, i2=0, i3=0, i4=8) (+1) |

Across invocations the image itself wraps at (1/4)n^2 + O(n) lines (0.0625 + 0.0625 for the two image arrays), giving d = 3.0. The filter never leaves cache at any realistic size.

## convolution — single-shot  [`exact`]

Accesses $A(n) = 163·n^2 - 2934·n + 13203$ (exact on n ≡ 0 mod 8); DMD order $n^{2.5}$, headroom **+0.5**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  1.12·n^2.5  +  394·n^2  +  35.8·n^1.5  +  163·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^2.5 | 1.12 | level | (5/4)·n + 15 | n^2 - 42·n + 320 | 0.00613 | read B[i1 + i3, i2 + i4] (i0=0, i3=0); read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 89.1 | level | 2 | 63·n^2 - 1134·n + 5103 | 0.387 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 89.1 | level | 2 | 63·n^2 - 1134·n + 5103 | 0.387 | read A[i3, i4] (i0=0) |
| n^2 | 37.3 | level | 37 | (49/8)·n^2 + (-889/8)·n + 504 | 0.0376 | read A[i3, i4] (i0=0) |
| n^2 | 36.8 | level | 36 | (49/8)·n^2 + (-945/8)·n + 567 | 0.0376 | read A[i3, i4] (i0=0, i4=0) |
| n^2 | 36.8 | level | 36 | (49/8)·n^2 + (-889/8)·n + 504 | 0.0376 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 36.5 | level | 37 | 6·n^2 - 110·n + 504 | 0.0368 | read B[i1 + i3, i2 + i4] (i0=0, i3=8, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i4=0) |
| n^2 | 6.08 | level | 37 | n^2 - 17·n + 72 | 0.00613 | read B[i1 + i3, i2 + i4] (i0=0, i3=8, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i4=0) |
| n^2 | 5.92 | level | 35 | n^2 - 25·n + 144 | 0.00613 | read B[i1 + i3, i2 + i4] (i0=0, i3=8, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i4=0) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-119/8)·n + 63 | 0.00537 | read A[i3, i4] (i0=0, i4=0) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read A[i3, i4] (i0=0) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read A[i3, i4] (i0=0, i3=0) |
| n^2 | 5.32 | level | 37 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | write C[i1, i2] (i0=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-135/8)·n + 81 | 0.00537 | read A[i3, i4] (i0=0, i4=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-135/8)·n + 81 | 0.00537 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read B[i1 + i3, i2 + i4] (i0=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-127/8)·n + 72 | 0.00537 | read B[i1 + i3, i2 + i4] (i0=0, i3=0) |
| n^2 | 5.25 | level | 36 | (7/8)·n^2 + (-175/8)·n + 126 | 0.00537 | read A[i3, i4] (i0=0) |
| n^2 | 4.56 | level | 37 | (3/4)·n^2 + (-55/4)·n + 63 | 0.0046 | read B[i1 + i3, i2 + i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.76 | level | 37 | (1/8)·n^2 + (-17/8)·n + 9 | 0.000767 | read A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.76 | level | 37 | (1/8)·n^2 + (-17/8)·n + 9 | 0.000767 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.76 | level | 37 | (1/8)·n^2 + (-17/8)·n + 9 | 0.000767 | read B[i1 + i3, i2 + i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.75 | level | 36 | (1/8)·n^2 + (-25/8)·n + 18 | 0.000767 | read A[i3, i4] (i0=0) |
| n^2 | 0.75 | level | 36 | (1/8)·n^2 + (-25/8)·n + 18 | 0.000767 | read A[i3, i4] (i0=0, i3=0) |
| n^2 | 0.74 | level | 35 | (1/8)·n^2 + (-25/8)·n + 18 | 0.000767 | read B[i1 + i3, i2 + i4] (i0=0, i3=0, i4=0) |
| n^1.5 | 7.83 | level | (5/4)·n + 16 | 7·n - 70 | 0.0429/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i3=0, i4=0); read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1.5 | 6.71 | level | (5/4)·n + 16 | 6·n - 60 | 0.0368/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 8 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 9 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 10 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 11 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 12 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 13 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 14 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 7 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i3=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 8 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 9 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 10 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 11 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 12 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 13 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 7 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 14 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i3=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 16 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 16 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i3=0) |
| n^1.5 | 1.12 | level | (5/4)·n + 16 | n - 10 | 0.00613/n | read B[i1 + i3, i2 + i4] (i0=0, i2=0, i4=0) |
| n^1 | 48.7 | level | 37 | 8·n - 80 | 0.0491/n | read A[i3, i4] (i0=0, i2=0, i3=0, i4=0); read A[i3, i4] (i0=0, i2=0, i4=0) |
| n^1 | 36.5 | level | 37 | 6·n - 60 | 0.0368/n | read A[i3, i4] (i0=0, i2=0, i4=8) |
| n^1 | 36 | level | 36 | 6·n - 60 | 0.0368/n | read A[i3, i4] (i0=0, i2=0, i4=0) |
| n^1 | 6.08 | level | 37 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i3=0, i4=8); read A[i3, i4] (i0=0, i2=0, i4=0) (+1) |
| n^1 | 6.08 | level | 37 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i4=8) |
| n^1 | 6.08 | level | 37 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i4=8) |
| n^1 | 6.08 | level | 37 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i3=0, i4=8) |
| n^1 | 6 | level | 36 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i4=0); read A[i3, i4] (i0=0, i2=0, i4=8) |
| n^1 | 6 | level | 36 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i4=0) |
| n^1 | 6 | level | 36 | n - 10 | 0.00613/n | read A[i3, i4] (i0=0, i2=0, i3=0, i4=0) |

Two-scale kernel (image n × n, filter fixed 9 × 9; all parameters except the filter extent bound to n). Single-shot: the sliding image window `read B[i1+i3, i2+i4]` is re-read across output columns at (5/4)n + 15 lines (nine rows' working set), order n^2.5; the filter lives at constant distances (2 and 37 lines — fully resident) and carries the n^2 bulk with huge constants (81 touches per output). Headroom +0.5 over the n^2 access count.

## correlation — infinite-repeat  [`exact`]

Accesses $A(n) = 2·n^3 + (21/2)·n^2 + (11/2)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0153·n^4  +  1.33·n^3.5  +  2.46·n^3  +  19·n^2.5  +  45.6·n^2  +  44.1·n^1.5  +  50·n^1  +  31.9·n^0.5  +  1·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0115 | ramp | 5·n + 32  →  (1/8)·n^2 + (9/8)·n - 8 | (3/64)·n^3 + (-171/64)·n^2 + (309/8)·n - 36 | 0.0234 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^4 | 0.00193 | ramp | 4·n + 22  →  (1/8)·n^2 + (9/8)·n - 14 | (1/128)·n^3 + (-7/16)·n^2 + (49/8)·n - 3 | 0.00391 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^4 | 0.00183 | ramp | 6·n + 38  →  (1/8)·n^2 + (9/8)·n - 16 | (1/128)·n^3 + (-73/128)·n^2 + (169/16)·n - 10 | 0.00391 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.619 | level | 2·n + 2 | (7/16)·n^3 + (-63/8)·n^2 + 35·n | 0.219 | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^3.5 | 0.475 | level | 2·n + 2 | (43/128)·n^3 + (-99/16)·n^2 + 28·n | 0.168 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.0884 | level | 2·n + 3 | (1/16)·n^3 + (-19/8)·n^2 + 22·n | 0.0312 | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^3.5 | 0.0663 | level | 2·n + 2 | (3/64)·n^3 + (-3/8)·n^2 | 0.0234 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.0663 | level | 2·n + 2 | (3/64)·n^3 + (-3/8)·n^2 | 0.0234 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.011 | level | 2·n + 2 | (1/128)·n^3 + (-1/16)·n^2 | 0.00391 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.663 | level | 3 | (49/128)·n^3 + (-91/16)·n^2 + 21·n | 0.191 | write D[i7, i8] (i0=0, i9=0); write D[i7, i8] (i0=0) |
| n^3 | 0.383 | level | 1 | (49/128)·n^3 + (-91/16)·n^2 + 21·n | 0.191 | read D[i7, i8] (i0=0, i9=0); read D[i7, i8] (i0=0) |
| n^3 | 0.177 | ramp | 4·n + 23  →  (1/8)·n^2 + (9/8)·n - 8 | (3/4)·n^2 + (-75/4)·n + 18 | 0.375/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.15 | ramp | 3·n + 14  →  (1/8)·n^2 + (9/8)·n - 9 | (5/8)·n^2 - 10·n | 0.312/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.148 | ramp | 4·n + 23  →  (1/8)·n^2 + (9/8)·n - 9 | (5/8)·n^2 + (-125/8)·n + 15 | 0.312/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-21/16)·n^2 + 7·n | 0.0273 | write D[i7, i8] (i0=0, i9=0); write D[i7, i8] (i0=0) |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-7/16)·n^2 | 0.0273 | write D[i7, i8] (i0=0, i9=0); write D[i7, i8] (i0=0) |
| n^3 | 0.0939 | ramp | n + 5  →  (1/8)·n^2 + (9/8)·n - 1 | (21/64)·n^2 + (-3/2)·n + 1 | 0.164/n | write D[i8, i7] (i0=0) |
| n^3 | 0.0547 | level | 1 | (7/128)·n^3 + (-21/16)·n^2 + 7·n | 0.0273 | read D[i7, i8] (i0=0, i9=0); read D[i7, i8] (i0=0) |
| n^3 | 0.0547 | level | 1 | (7/128)·n^3 + (-7/16)·n^2 | 0.0273 | read D[i7, i8] (i0=0, i9=0); read D[i7, i8] (i0=0) |
| n^3 | 0.0451 | ramp | (1/8)·n^2 + (1/8)·n + 9  →  (1/4)·n^2 - 4·n + 36 | (1/8)·n^2 - 4·n + 2 | 0.0625/n | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^3 | 0.0409 | ramp | (1/8)·n^2 + (1/8)·n + 2  →  (1/8)·n^2 + (1/4)·n - 1 | (1/8)·n^2 - 3·n + 1 | 0.0625/n | read B[i4, i3] (i0=0, i4=0); read B[i4, i3] (i0=0) |
| n^3 | 0.036 | ramp | (5/2)·n - 4  →  (1/8)·n^2 + (1/4)·n - 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0625/n | read B[i5, i6] (i0=0) |
| n^3 | 0.0347 | ramp | (9/2)·n + 19  →  (1/8)·n^2 + (11/8)·n - 19 | (1/8)·n^2 + (-17/4)·n + 8 | 0.0625/n | read B[i9, i8] (i0=0, i7=0) |
| n^3 | 0.03 | ramp | 3·n + 19  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i8=6) |
| n^3 | 0.0299 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 14 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i9, i8] (i0=0, i8=0, i9=0); read B[i9, i8] (i0=0, i8=0) |
| n^3 | 0.0296 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 14 | (1/8)·n^2 - 3·n + 1 | 0.0625/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.0296 | ramp | 4·n + 21  →  (1/8)·n^2 + (9/8)·n - 15 | (1/8)·n^2 - 3·n | 0.0625/n | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^3 | 0.0296 | ramp | 4·n + 20  →  (1/8)·n^2 + (9/8)·n - 16 | (1/8)·n^2 - 3·n | 0.0625/n | read B[i9, i8] (i0=0, i8=7, i9=0); read B[i9, i8] (i0=0, i8=7) |
| n^3 | 0.0295 | ramp | 4·n + 28  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n^2 + (-25/8)·n + 3 | 0.0625/n | read B[i9, i8] (i0=0, i8=14, i9=0); read B[i9, i8] (i0=0, i8=14) |
| n^3 | 0.0295 | ramp | 4·n + 22  →  (1/8)·n^2 + (9/8)·n - 14 | (1/8)·n^2 + (-25/8)·n + 3 | 0.0625/n | read B[i9, i8] (i0=0, i8=8, i9=0); read B[i9, i8] (i0=0, i8=8) |
| n^3 | 0.0292 | ramp | 5·n + 29  →  (1/8)·n^2 + (9/8)·n - 16 | (1/8)·n^2 + (-33/8)·n + 4 | 0.0625/n | read B[i9, i8] (i0=0, i8=15, i9=0); read B[i9, i8] (i0=0, i8=15) |
| n^3 | 0.0287 | ramp | 4·n + 30  →  (1/8)·n^2 + (9/8)·n - 18 | (1/8)·n^2 + (-17/4)·n + 8 | 0.0625/n | read B[i9, i8] (i0=0) |
| n^3 | 0.0273 | level | (1/4)·n^2 + (1/4)·n | (7/128)·n^2 + (-35/16)·n + 21 | 0.0273/n | write D[i7, i8] (i0=0) |
| n^3 | 0.0237 | ramp | (1/4)·n^2 + (-5/8)·n + 21  →  (1/4)·n^2 + (1/4)·n - 6 | (7/128)·n^2 + (-37/16)·n + 24 | 0.0273/n | write D[i8, i7] (i0=0) |
| n^3 | 0.0154 | ramp | 2·n + 11  →  (1/8)·n^2 + (9/8)·n - 7 | (7/128)·n^2 + (-9/16)·n + 1 | 0.0273/n | write D[i8, i7] (i0=0) |
| n^3 | 0.0135 | level | 3 | (1/128)·n^3 + (-1/16)·n^2 | 0.00391 | write A[i1] (i0=0); read B[i2, i1] (i0=0) (+9) |
| n^3 | 0.0133 | ramp | 2·n + 12  →  (1/8)·n^2 + (9/8)·n - 1 | (3/64)·n^2 + (-3/8)·n | 0.0234/n | write D[i8, i7] (i0=0) |
| n^3 | 0.0121 | ramp | 4·n + 23  →  (1/8)·n^2 + (9/8)·n - 8 | (3/64)·n^2 + (-15/8)·n + 18 | 0.0234/n | read B[i9, i8] (i0=0) |
| n^3 | 0.00781 | level | 1 | (1/128)·n^3 + (-1/16)·n^2 + n | 0.00391 | read B[i9, i8] (i0=0, i7=0, i8=0); read D[i7, i8] (i0=0, i9=0) (+2) |
| n^3 | 0.00391 | level | (1/4)·n^2 + 2·n + (7/4) | (1/128)·n^2 + (-13/64)·n + (153/128) | 0.00391/n | write D[i7, i8] (i0=0) |
| n^3 | 0.00391 | level | (1/4)·n^2 + (1/4)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.00391/n | write D[i7, i8] (i0=0) |
| n^3 | 0.00323 | ramp | (1/4)·n^2 + (-5/8)·n + 22  →  (1/4)·n^2 + (1/4)·n - 13 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00391/n | write D[i8, i7] (i0=0) |
| n^3 | 0.00212 | ramp | 3·n + 20  →  (1/8)·n^2 + (9/8)·n - 7 | (1/128)·n^2 + (-3/16)·n + 1 | 0.00391/n | write D[i8, i7] (i0=0) |
| n^3 | 0.00193 | ramp | 5·n + 29  →  (1/8)·n^2 + (9/8)·n - 16 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00391/n | read B[i9, i8] (i0=0) |
| n^2.5 | 4.77 | level | 2·n + 2 | (27/8)·n^2 - 27·n | 1.69/n | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^2.5 | 2.65 | level | 2·n + 2 | (15/8)·n^2 - 15·n | 0.938/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^2.5 | 1.88 | level | n + 2 | (15/8)·n^2 | 0.938/n | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i9, i7] (i0=0, i8=8, i9=0); read B[i9, i7] (i0=0, i8=8) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i9, i8] (i0=0, i8=7, i9=0); read B[i9, i8] (i0=0, i8=7) |
| n^2.5 | 1.06 | level | 2·n + 3 | (3/4)·n^2 - 12·n | 0.375/n | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^2.5 | 1.06 | level | 2·n + 2 | (3/4)·n^2 - 6·n | 0.375/n | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i8=6) |
| n^2.5 | 0.888 | ramp | n + 5  →  2·n + 2 | (3/4)·n^2 + (-15/2)·n + 12 | 0.375/n | read B[i9, i7] (i0=0, i8=0) |
| n^2.5 | 0.888 | ramp | n + 4  →  2·n + 1 | (3/4)·n^2 + (-15/2)·n + 12 | 0.375/n | read B[i9, i7] (i0=0) |
| n^2.5 | 0.875 | level | n + 2 | (7/8)·n^2 | 0.438/n | read B[i4, i3] (i0=0, i4=0); read B[i4, i3] (i0=0) |
| n^2.5 | 0.875 | level | n + 1 | (7/8)·n^2 | 0.438/n | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2.5 | 0.75 | level | n + 2 | (3/4)·n^2 | 0.375/n | read B[i9, i7] (i0=0, i8=1, i9=0); read B[i9, i7] (i0=0, i8=1) |
| n^2.5 | 0.177 | level | 2·n + 2 | (1/8)·n^2 + (-7/8)·n - 1 | 0.0625/n | read B[i9, i7] (i0=0, i8=1, i9=0); read B[i9, i7] (i0=0, i8=1) |
| n^2.5 | 0.177 | level | 2·n + 3 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i9, i7] (i0=0, i8=8, i9=0); read B[i9, i7] (i0=0, i8=8) |
| n^2.5 | 0.177 | level | 2·n + 4 | (1/8)·n^2 - n | 0.0625/n | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^2.5 | 0.153 | level | (3/8)·n + 1 | (1/4)·n^2 + (-21/4)·n + 5 | 0.125/n | read A[i6] (i0=0); read C[i6] (i0=0) |
| n^2.5 | 0.148 | ramp | n + 4  →  2·n + 1 | (1/8)·n^2 + (-5/4)·n + 2 | 0.0625/n | read B[i9, i7] (i0=0, i8=1) |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 | 1.31/n | read A[i6] (i0=0, i5=0); read C[i6] (i0=0, i5=0) (+3) |
| n^2 | 3.75 | level | 1 | (15/4)·n^2 - 15·n | 1.88/n | read D[i7, i8] (i0=0, i9=0); read D[i7, i8] (i0=0) |
| n^2 | 3.25 | level | 3 | (15/8)·n^2 - 15·n | 0.938/n | write D[i7, i8] (i0=0, i9=0); write D[i7, i8] (i0=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 + (-7/8)·n | 0.875/n | read A[i3] (i0=0); write C[i3] (i0=0) |
| n^2 | 2.65 | level | 2 | (15/8)·n^2 | 0.938/n | write D[i7, i8] (i0=0, i9=0); write D[i7, i8] (i0=0) |
| n^2 | 1.88 | level | 1 | (15/8)·n^2 | 0.938/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 + (21/4)·n | 0.875/n | write A[i1] (i0=0); read A[i1] (i0=0) (+3) |
| n^2 | 1.5 | level | 1 | (3/2)·n^2 | 0.75/n | read D[i7, i8] (i0=0, i8=0, i9=0); read D[i7, i8] (i0=0, i8=0) (+1) |
| n^2 | 1.3 | level | 3 | (3/4)·n^2 - 6·n | 0.375/n | write D[i7, i8] (i0=0, i8=6, i9=0); write D[i7, i8] (i0=0, i8=6) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.438/n | write A[i1] (i0=0) |
| n^2 | 1.08 | level | 3 | (5/8)·n^2 - 5·n | 0.312/n | write D[i7, i8] (i0=0, i9=0); write D[i7, i8] (i0=0) |
| n^2 | 1.06 | level | 2 | (3/4)·n^2 | 0.375/n | write D[i7, i8] (i0=0, i8=0, i9=0); write D[i7, i8] (i0=0, i8=0) |
| n^2 | 0.884 | level | 2 | (5/8)·n^2 | 0.312/n | write D[i7, i8] (i0=0, i9=0); write D[i7, i8] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.438/n | read B[i5, i6] (i0=0) |
| n^2 | 0.75 | level | 1 | (3/4)·n^2 - 6·n | 0.375/n | read D[i7, i8] (i0=0, i8=6, i9=0); read D[i7, i8] (i0=0, i8=6) |
| n^2 | 0.625 | level | 1 | (5/8)·n^2 | 0.312/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^2 | 0.625 | level | 1 | (5/8)·n^2 - 5·n | 0.312/n | read D[i7, i8] (i0=0, i9=0); read D[i7, i8] (i0=0) |
| n^2 | 0.625 | level | 1 | (5/8)·n^2 | 0.312/n | read D[i7, i8] (i0=0, i9=0); read D[i7, i8] (i0=0) |
| n^2 | 0.541 | level | 2 | (49/128)·n^2 + (-7/16)·n | 0.191/n | write D[i7, i8] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-9/4)·n + 22 | n - 1 | 0.5·n^-2 | read B[i2, i1] (i0=0, i1=8, i2=0); read B[i2, i1] (i0=0, i1=8) |
| n^2 | 0.5 | level | (1/4)·n^2 - 4·n + 36 | n - 1 | 0.5·n^-2 | read B[i2, i1] (i0=0, i1=8, i2=0); read B[i2, i1] (i0=0, i1=8) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 4 | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i1=0, i2=0); read B[i2, i1] (i0=0, i1=0) |
| n^2 | 0.5 | level | (1/4)·n^2 - 2·n + 11 | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i1=0, i2=0); read B[i2, i1] (i0=0, i1=0) |
| n^2 | 0.438 | level | (1/4)·n^2 + (1/4)·n | (7/8)·n - 21 | 0.438·n^-2 | write D[i7, i8] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 - 7·n + 28 | 0.219/n | read D[i7, i8] (i0=0) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 + (3/4)·n | 0.125/n | read B[i4, i3] (i0=0, i4=0); read A[i3] (i0=0, i4=0) (+3) |
| n^2 | 0.375 | level | (1/4)·n^2 + (-5/8)·n + 15 | (3/4)·n - 12 | 0.375·n^-2 | write D[i8, i7] (i0=0, i7=8) |
| n^2 | 0.375 | level | (1/4)·n^2 + (1/4)·n | (3/4)·n - 12 | 0.375·n^-2 | write D[i7, i8] (i0=0) |
| n^2 | 0.375 | level | (1/4)·n^2 + (-5/8)·n + 8 | (3/4)·n - 6 | 0.375·n^-2 | write D[i8, i7] (i0=0, i7=0) |
| n^2 | 0.375 | level | 1 | (3/8)·n^2 - 2·n | 0.188/n | read D[i7, i8] (i0=0, i8=0, i9=0); read D[i7, i8] (i0=0, i8=0) (+1) |
| n^2 | 0.354 | level | (1/8)·n^2 + (11/8)·n - 9 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i7=0, i9=0); read B[i9, i8] (i0=0, i7=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 2 | n - 1 | 0.5·n^-2 | read B[i4, i3] (i0=0, i3=8, i4=0); read B[i4, i3] (i0=0, i3=8) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n | n | 0.5·n^-2 | read B[i4, i3] (i0=0, i4=0); read B[i4, i3] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 9 | n - 1 | 0.5·n^-2 | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n + (15/4) | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + n + (31/8) | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 2 | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 1 | n | 0.5·n^-2 | read B[i4, i3] (i0=0, i3=0, i4=0); read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n | n | 0.5·n^-2 | read B[i5, i6] (i0=0, i5=0, i6=0); read B[i5, i6] (i0=0, i6=0) |
| n^2 | 0.251 | ramp | (7/2)·n + 12  →  (1/8)·n^2 + (1/8)·n + 21 | n - 2 | 0.5·n^-2 | read B[i9, i8] (i0=0, i7=0, i8=15) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 | 0.125/n | read D[i7, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i8=6, i9=0) (+2) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (5/8)·n - 1 | 0.125/n | read A[i1] (i0=0); write A[i1] (i0=0) (+3) |
| n^2 | 0.247 | ramp | (5/2)·n + 5  →  (1/8)·n^2 + (1/8)·n + 11 | n - 2 | 0.5·n^-2 | read B[i9, i8] (i0=0, i7=0, i8=7) |
| n^2 | 0.242 | ramp | (3/2)·n - 1  →  (1/8)·n^2 + (1/8)·n + 2 | n - 2 | 0.5·n^-2 | read B[i9, i7] (i0=0, i7=0, i8=0) |
| n^2 | 0.242 | ramp | (3/2)·n - 2  →  (1/8)·n^2 + (1/8)·n + 1 | n - 2 | 0.5·n^-2 | read B[i5, i6] (i0=0) |
| n^2 | 0.231 | ramp | (1/4)·n^2 + (-3/8)·n + 15  →  (1/4)·n^2 + (1/4)·n - 2 | (1/2)·n - 12 | 0.25·n^-2 | write D[i8, i7] (i0=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 - n | 0.0625/n | write D[i7, i8] (i0=0, i8=6, i9=0); write D[i7, i8] (i0=0, i8=6) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 - n | 0.0625/n | write D[i7, i8] (i0=0, i8=0, i9=0); write D[i7, i8] (i0=0, i8=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 | 0.0625/n | write B[i5, i6] (i0=0) |
| n^2 | 0.182 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n | 0.375·n^-2 | write D[i7, i7] (i0=0) |
| n^2 | 0.182 | ramp | n + 4  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n - 1 | 0.375·n^-2 | write D[i8, i7] (i0=0, i8=0) |
| n^2 | 0.182 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 2 | (3/4)·n - 1 | 0.375·n^-2 | write D[i8, i7] (i0=0) |
| n^2 | 0.18 | ramp | 3·n + 14  →  (1/8)·n^2 + (9/8)·n - 8 | (3/4)·n - 12 | 0.375·n^-2 | read B[i9, i8] (i0=0) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 | 0.0625/n | write D[i7, i8] (i0=0, i8=6, i9=0); write D[i7, i8] (i0=0, i8=6) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 | 0.0625/n | write D[i7, i8] (i0=0, i8=0, i9=0); write D[i7, i8] (i0=0, i8=0) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 | 0.0625/n | read B[i2, i1] (i0=0, i2=0); write A[i1] (i0=0, i2=0) (+1) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 - n | 0.0625/n | read D[i7, i8] (i0=0, i8=6, i9=0); read D[i7, i8] (i0=0, i8=6) |
| n^2 | 0.0773 | level | 2 | (7/128)·n^2 + (-7/16)·n | 0.0273/n | write D[i7, i8] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (9/8)·n + (85/8) | (1/8)·n + (-25/8) | 0.0625·n^-2 | write D[i8, i7] (i0=0, i7=8) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-5/8)·n + 15 | (1/8)·n - 3 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i7=8) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-5/8)·n + 14 | (1/8)·n - 3 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i7=8) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (9/8)·n + (29/8) | (1/8)·n + (-17/8) | 0.0625·n^-2 | write D[i8, i7] (i0=0, i7=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-5/8)·n + 8 | (1/8)·n - 2 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i7=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-5/8)·n + 7 | (1/8)·n - 2 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i7=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + 2·n + (7/4) | (1/8)·n + (-25/8) | 0.0625·n^-2 | write D[i7, i7] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/4)·n | (1/8)·n - 3 | 0.0625·n^-2 | write D[i7, i7] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + 2·n + (7/4) | (1/8)·n + (-9/8) | 0.0625·n^-2 | write D[i7, i8] (i0=0, i7=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | write D[i7, i8] (i0=0, i7=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | write D[i7, i8] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + 2·n + (7/4) | (1/8)·n + (-9/8) | 0.0625·n^-2 | write D[i7, i8] (i0=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-3/2)·n + 8 | 0.0312/n | read D[i7, i8] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + 2·n + (7/4) | (1/8)·n + (-9/8) | 0.0625·n^-2 | write D[i7, i8] (i0=0, i8=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | write D[i7, i8] (i0=0, i8=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + 2·n + (7/4) | (1/8)·n + (-9/8) | 0.0625·n^-2 | write C[i3] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | write C[i3] (i0=0) |
| n^2 | 0.0594 | ramp | (1/4)·n^2 + (1/8)·n + 2  →  (1/4)·n^2 + (1/4)·n - 1 | (1/8)·n - 2 | 0.0625·n^-2 | write A[i1] (i0=0) |
| n^2 | 0.0592 | ramp | (1/4)·n^2 + (-5/8)·n + 14  →  (1/4)·n^2 + (1/4)·n - 7 | (1/8)·n - 2 | 0.0625·n^-2 | write D[i8, i7] (i0=0) |
| n^2 | 0.0578 | ramp | (1/4)·n^2 + (1/8)·n + 3  →  (1/4)·n^2 + (1/4)·n - 1 | (1/8)·n - 3 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i8=0) |
| n^2 | 0.0577 | ramp | (1/4)·n^2 + (-1/2)·n + 18  →  (1/4)·n^2 + (1/4)·n - 6 | (1/8)·n - 3 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i8=5) |
| n^2 | 0.0576 | ramp | (1/4)·n^2 + (-5/8)·n + 22  →  (1/4)·n^2 + (1/4)·n - 6 | (1/8)·n - 3 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i8=7) |
| n^2 | 0.0576 | ramp | (1/4)·n^2 + (-5/8)·n + 21  →  (1/4)·n^2 + (1/4)·n - 7 | (1/8)·n - 3 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i8=6) |
| n^2 | 0.055 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-7/4)·n + 2 | (1/4)·n - 5 | 0.125·n^-2 | read A[i6] (i0=0, i5=0); read C[i6] (i0=0, i5=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | read B[i5, i6] (i0=0) |
| n^2 | 0.0415 | ramp | (1/8)·n^2 + (-7/8)·n + 3  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3] (i0=0, i4=0) |
| n^2 | 0.0401 | ramp | (1/8)·n^2 + (1/4)·n + 27  →  (1/8)·n^2 + (11/8)·n - 18 | (1/8)·n - 4 | 0.0625·n^-2 | read B[i9, i8] (i0=0, i7=0, i9=0) |
| n^2 | 0.0302 | ramp | 2·n + 11  →  (1/8)·n^2 + (9/8)·n - 7 | (1/8)·n - 1 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i8=0) |
| n^2 | 0.0302 | ramp | 2·n + 10  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n - 1 | 0.0625·n^-2 | write D[i8, i7] (i0=0) |
| n^2 | 0.0302 | ramp | 2·n + 10  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n - 1 | 0.0625·n^-2 | write D[i7, i7] (i0=0) |
| n^2 | 0.0293 | ramp | 5·n + 28  →  (1/8)·n^2 + (9/8)·n - 17 | (1/8)·n - 4 | 0.0625·n^-2 | read B[i9, i8] (i0=0, i9=0) |
| n^2 | 0.029 | ramp | (27/8)·n + 9  →  (1/8)·n^2 + (1/8)·n - 15 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i9, i8] (i0=0, i7=0) |
| n^2 | 0.029 | ramp | (19/8)·n - 2  →  (1/8)·n^2 + (-3/4)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read B[i5, i6] (i0=0, i5=0) |
| n^2 | 0.0288 | ramp | 3·n + 20  →  (1/8)·n^2 + (1/8)·n - 16 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i9, i8] (i0=0) |
| n^1.5 | 6 | level | n + 3 | 6·n - 12 | 3·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.87 | ramp | 3·n + 21  →  4·n + 18 | n - 2 | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=15) |
| n^1.5 | 1.84 | level | (3/8)·n + 1 | 3·n - 3 | 1.5·n^-2 | read A[i6] (i0=0, i6=0); read C[i6] (i0=0, i6=0) (+1) |
| n^1.5 | 1.73 | level | 3·n + 13 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=8, i9=0); read B[i9, i8] (i0=0, i8=8) |
| n^1.5 | 1.73 | level | 3·n + 14 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 15 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 16 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 17 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 18 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 19 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=14, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 12 | n | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.58 | ramp | 2·n + 12  →  3·n + 9 | n - 2 | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=7) |
| n^1.5 | 1.41 | level | 2·n + 5 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 6 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 7 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 8 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 9 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 10 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i8=6) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=0, i9=0); read B[i9, i8] (i0=0, i8=0) |
| n^1.5 | 1.22 | level | (3/8)·n + 1 | 2·n - 2 | 1·n^-2 | read A[i6] (i0=0); read C[i6] (i0=0) |
| n^1.5 | 1.21 | ramp | n + 4  →  2·n + 1 | n - 2 | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.06 | level | 2·n + 2 | (3/4)·n - 6 | 0.375·n^-2 | read B[i9, i7] (i0=0) |
| n^1.5 | 1.06 | level | 2·n + 3 | (3/4)·n - 6 | 0.375·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^1.5 | 0.75 | level | n + 3 | (3/4)·n - 6 | 0.375·n^-2 | read B[i9, i7] (i0=0, i9=0) |
| n^1.5 | 0.75 | level | n + 4 | (3/4)·n - 6 | 0.375·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 0.125 | level | n + 3 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i9, i7] (i0=0, i8=1, i9=0) |
| n^1 | 5.25 | level | 1 | (21/4)·n - 21 | 2.62·n^-2 | read D[i7, i8] (i0=0) |
| n^1 | 3.5 | level | (1/4)·n^2 + (1/4)·n | 7 | 3.5·n^-3 | write D[i7, i8] (i0=0) |
| n^1 | 2.5 | level | (1/4)·n^2 + (1/4)·n | 5 | 2.5·n^-3 | write D[i7, i8] (i0=0) |
| n^1 | 2 | level | (1/4)·n^2 + (1/4)·n | 4 | 2·n^-3 | write D[i8, i7] (i0=0) |
| n^1 | 1.75 | level | 1 | (7/4)·n | 0.875·n^-2 | write D[i7, i8] (i0=0, i8=0); read D[i7, i8] (i0=0, i8=0) |
| n^1 | 1 | level | (1/4)·n^2 + 2·n + (7/4) | 2 | 1·n^-3 | write A[i1] (i0=0); write C[i3] (i0=0, i3=0) |
| n^1 | 1 | level | (1/4)·n^2 + (1/4)·n | 2 | 1·n^-3 | write A[i1] (i0=0); write C[i3] (i0=0, i3=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n - 7 | 0.438·n^-2 | read D[i7, i8] (i0=0, i8=7) |
| n^1 | 0.75 | level | 1 | (3/4)·n - 6 | 0.375·n^-2 | read D[i7, i8] (i0=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (-3/4)·n + 1 | 2 | 1·n^-3 | read A[i6] (i0=0, i5=0, i6=0); read C[i6] (i0=0, i5=0, i6=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (-17/4) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=5) |
| n^1 | 0.5 | level | (1/4)·n^2 + (5/4)·n + (17/2) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=5) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/2)·n + 12 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=5) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/8)·n + 10 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 8 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 6 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + 4 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (7/4)·n + (-9/2) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (9/8)·n + (85/8) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/8)·n + 15 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (7/4)·n + (-11/2) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=6) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (-21/4) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=6) |
| n^1 | 0.5 | level | (1/4)·n^2 + (9/8)·n + (77/8) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=6) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/8)·n + 14 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=6) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (-21/4) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (23/8) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 2 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=8, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (3/2)·n + (5/4) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (7/4)·n + (3/2) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write D[i7, i7] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write D[i7, i7] (i0=0, i7=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i7, i7] (i0=0, i7=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (7/4)·n + (3/2) | 1 | 0.5·n^-3 | write D[i7, i7] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i7, i8] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write D[i7, i8] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i7, i8] (i0=0, i8=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write D[i7, i8] (i0=0, i8=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + (5/4)·n + (5/2) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=5) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/2)·n + 6 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=5) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/8)·n + 5 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 4 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 3 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (15/8) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 1 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write D[i7, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i7, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i7, i7] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write C[i3] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write C[i3] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 1 | 1 | 0.5·n^-3 | write A[i1] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (15/8) | 1 | 0.5·n^-3 | write A[i1] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | write D[i7, i8] (i0=0, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write D[i7, i8] (i0=0, i8=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (9/8)·n + (29/8) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/8)·n + 8 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (9/8)·n + (21/8) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/8)·n + 7 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (9/8)·n + (21/8) | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=6) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/8)·n + 7 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i7=0, i8=6) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2·n + (7/4) | 1 | 0.5·n^-3 | write D[p0 - 1, p0 - 1] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n - 9 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 18 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i7=0, i8=15, i9=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 9 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i7=0, i8=7, i9=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (31/8) | 1 | 0.5·n^-3 | read B[i2, i1] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + n + (23/8) | 1 | 0.5·n^-3 | read B[i2, i1] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i7=0, i8=0, i9=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | read B[i5, i6] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 2 | 1 | 0.5·n^-3 | read A[i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 1 | 1 | 0.5·n^-3 | read A[i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/4)·n + 2 | 1 | 0.5·n^-3 | read A[i6] (i0=0, i5=0, i6=8) |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0625·n^-2 | read D[i7, i8] (i0=0, i8=0) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i9, i7] (i0=0, i8=0) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 2 | level | 4·n + 19 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i8=15, i9=0) |
| n^0.5 | 1.73 | level | 3·n + 10 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i8=7, i9=0) |
| n^0.5 | 1.54 | level | (19/8)·n + 3 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i7=0, i8=7) |
| n^0.5 | 1.41 | level | 2·n + 11 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i8=7) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=0) |
| n^0.5 | 1.22 | level | (3/8)·n | 2 | 1·n^-3 | read B[i5, i6] (i0=0, i5=0); read A[i6] (i0=0, i5=0) (+1) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.5·n^-3 | read B[i5, i6] (i0=0, i5=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 2 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i7=0, i8=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0) |
| n^0.5 | 1 | level | n + 4 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0) |
| n^0.5 | 1 | level | n + 2 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=0) |
| n^0 | 1 | level | 1 | 1 | 0.5·n^-3 | write D[p0 - 1, p0 - 1] (i0=0) |

Structurally identical to covariance after normalization: same 4n → (1/8)n^2 ramps on `read B[i9,i8]`, same 2n-line column levels, d = 4.0, headroom +1.0.

## correlation — single-shot  [`exact`]

Accesses $A(n) = 2·n^3 + (21/2)·n^2 + (11/2)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0161·n^4  +  1.33·n^3.5  +  2.11·n^3  +  19.4·n^2.5  +  25.3·n^2  +  34.5·n^1.5  +  7.68·n^1  +  30.3·n^0.5  +  1·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0121 | ramp | 4·n + 23  →  (1/8)·n^2 + (9/8)·n - 8 | (3/64)·n^3 + (-15/8)·n^2 + 18·n | 0.0234 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^4 | 0.00202 | ramp | 4·n + 22  →  (1/8)·n^2 + (9/8)·n - 14 | (1/128)·n^3 + (-5/16)·n^2 + 3·n | 0.00391 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^4 | 0.00193 | ramp | 5·n + 29  →  (1/8)·n^2 + (9/8)·n - 16 | (1/128)·n^3 + (-7/16)·n^2 + 6·n | 0.00391 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.619 | level | 2·n + 2 | (7/16)·n^3 + (-63/8)·n^2 + 35·n | 0.219 | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^3.5 | 0.398 | level | 2·n + 2 | (9/32)·n^3 + (-165/32)·n^2 + (207/8)·n - 14 | 0.141 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.0884 | level | 2·n + 3 | (1/16)·n^3 + (-19/8)·n^2 + 22·n | 0.0312 | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^3.5 | 0.0773 | level | 2·n + 2 | (7/128)·n^3 + (-21/16)·n^2 + 7·n - 7 | 0.0273 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.0663 | level | 2·n + 2 | (3/64)·n^3 + (-3/8)·n^2 | 0.0234 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.0663 | level | 2·n + 2 | (3/64)·n^3 + (-3/8)·n^2 | 0.0234 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3.5 | 0.011 | level | 2·n + 2 | (1/128)·n^3 + (-1/16)·n^2 | 0.00391 | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.866 | level | 3 | (1/2)·n^3 + (-11/2)·n^2 + 35·n | 0.25 | read B[i4, i3] (i0=0); read A[i3] (i0=0) (+6) |
| n^3 | 0.5 | level | 1 | (1/2)·n^3 + 2·n^2 + (7/2)·n | 0.25 | read A[i1] (i0=0); write A[i1] (i0=0) (+4) |
| n^3 | 0.18 | ramp | 3·n + 14  →  (1/8)·n^2 + (9/8)·n - 8 | (3/4)·n^2 - 12·n | 0.375/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.15 | ramp | 3·n + 14  →  (1/8)·n^2 + (9/8)·n - 8 | (5/8)·n^2 + (-79/8)·n - 2 | 0.312/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.0939 | ramp | n + 5  →  (1/8)·n^2 + (9/8)·n - 1 | (21/64)·n^2 + (-3/2)·n + 1 | 0.164/n | write D[i8, i7] (i0=0) |
| n^3 | 0.042 | ramp | (1/8)·n^2 + (1/8)·n + 2  →  (1/8)·n^2 + (1/4)·n - 1 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i4, i3] (i0=0, i4=0); read B[i4, i3] (i0=0) |
| n^3 | 0.036 | ramp | (5/2)·n - 4  →  (1/8)·n^2 + (1/4)·n - 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0625/n | read B[i5, i6] (i0=0) |
| n^3 | 0.0355 | ramp | (7/2)·n + 12  →  (1/8)·n^2 + (11/8)·n - 19 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0625/n | read B[i9, i8] (i0=0, i7=0) |
| n^3 | 0.0299 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 14 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i9, i8] (i0=0, i8=0, i9=0); read B[i9, i8] (i0=0, i8=0) |
| n^3 | 0.0299 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 14 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i9, i8] (i0=0, i9=0); read B[i9, i8] (i0=0) |
| n^3 | 0.0299 | ramp | 3·n + 19  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0625/n | read B[i9, i8] (i0=0, i8=6) |
| n^3 | 0.0296 | ramp | 3·n + 12  →  (1/8)·n^2 + (9/8)·n - 15 | (1/8)·n^2 - 3·n + 1 | 0.0625/n | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^3 | 0.0296 | ramp | 4·n + 20  →  (1/8)·n^2 + (9/8)·n - 16 | (1/8)·n^2 - 3·n | 0.0625/n | read B[i9, i8] (i0=0, i8=7, i9=0); read B[i9, i8] (i0=0, i8=7) |
| n^3 | 0.029 | ramp | 3·n + 21  →  (1/8)·n^2 + (9/8)·n - 18 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0625/n | read B[i9, i8] (i0=0) |
| n^3 | 0.0154 | ramp | 2·n + 11  →  (1/8)·n^2 + (9/8)·n - 7 | (7/128)·n^2 + (-9/16)·n + 1 | 0.0273/n | write D[i8, i7] (i0=0) |
| n^3 | 0.0133 | ramp | 2·n + 12  →  (1/8)·n^2 + (9/8)·n - 1 | (3/64)·n^2 + (-3/8)·n | 0.0234/n | write D[i8, i7] (i0=0) |
| n^3 | 0.00212 | ramp | 3·n + 20  →  (1/8)·n^2 + (9/8)·n - 7 | (1/128)·n^2 + (-3/16)·n + 1 | 0.00391/n | write D[i8, i7] (i0=0) |
| n^2.5 | 4.95 | level | 2·n + 2 | (7/2)·n^2 - 28·n | 1.75/n | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^2.5 | 2.65 | level | 2·n + 2 | (15/8)·n^2 + (-57/4)·n - 6 | 0.938/n | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^2.5 | 2.62 | level | n + 2 | (21/8)·n^2 | 1.31/n | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i9, i7] (i0=0, i8=8, i9=0); read B[i9, i7] (i0=0, i8=8) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i9, i8] (i0=0, i8=7, i9=0); read B[i9, i8] (i0=0, i8=7) |
| n^2.5 | 1.06 | level | 2·n + 3 | (3/4)·n^2 - 12·n | 0.375/n | read B[i9, i7] (i0=0, i9=0); read B[i9, i7] (i0=0) |
| n^2.5 | 1.06 | level | 2·n + 2 | (3/4)·n^2 + (-27/4)·n + 6 | 0.375/n | read B[i9, i8] (i0=0, i8=6) |
| n^2.5 | 1.04 | ramp | n + 4  →  2·n + 1 | (7/8)·n^2 + (-35/4)·n + 14 | 0.438/n | read B[i9, i7] (i0=0) |
| n^2.5 | 0.888 | ramp | n + 5  →  2·n + 2 | (3/4)·n^2 + (-15/2)·n + 12 | 0.375/n | read B[i9, i7] (i0=0, i8=0) |
| n^2.5 | 0.875 | level | n + 2 | (7/8)·n^2 + (-7/8)·n | 0.438/n | read B[i4, i3] (i0=0, i4=0); read B[i4, i3] (i0=0) |
| n^2.5 | 0.875 | level | n + 1 | (7/8)·n^2 + (-7/8)·n | 0.438/n | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2.5 | 0.398 | level | 2·n + 2 | (9/32)·n^2 + (-39/8)·n + 21 | 0.141/n | read B[i9, i8] (i0=0) |
| n^2.5 | 0.177 | level | 2·n + 3 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i9, i7] (i0=0, i8=8, i9=0); read B[i9, i7] (i0=0, i8=8) |
| n^2.5 | 0.177 | level | 2·n + 4 | (1/8)·n^2 - 2·n + 1 | 0.0625/n | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^2.5 | 0.153 | level | (3/8)·n + 1 | (1/4)·n^2 + (-17/4)·n + 4 | 0.125/n | read A[i6] (i0=0); read C[i6] (i0=0) |
| n^2 | 7.79 | level | 3 | (9/2)·n^2 - 36·n | 2.25/n | write D[i7, i8] (i0=0) |
| n^2 | 4.95 | level | 2 | (7/2)·n^2 | 1.75/n | write D[i7, i8] (i0=0) |
| n^2 | 2.5 | level | 1 | (5/2)·n^2 + (1/8)·n | 1.25/n | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 | 0.438/n | read C[i6] (i0=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 | 0.438/n | read A[i6] (i0=0) |
| n^2 | 1.41 | level | 2 | n^2 | 0.5/n | write A[i1] (i0=0); read B[i9, i7] (i0=0, i7=0, i8=0, i9=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.438/n | read B[i9, i8] (i0=0, i8=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.438/n | read B[i5, i6] (i0=0) |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-7/8)·n | 0.219/n | write D[i7, i8] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (11/8)·n - 9 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i7=0, i9=0); read B[i9, i8] (i0=0, i7=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n | n | 0.5·n^-2 | read B[i4, i3] (i0=0, i4=0); read B[i4, i3] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 1 | n - 1 | 0.5·n^-2 | read B[i4, i3] (i0=0, i3=0, i4=0); read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n | n - 1 | 0.5·n^-2 | read B[i5, i6] (i0=0, i5=0, i6=0); read B[i5, i6] (i0=0, i6=0) |
| n^2 | 0.247 | ramp | (5/2)·n + 5  →  (1/8)·n^2 + (1/8)·n + 11 | n - 2 | 0.5·n^-2 | read B[i9, i8] (i0=0, i7=0, i8=7) |
| n^2 | 0.242 | ramp | (3/2)·n - 1  →  (1/8)·n^2 + (1/8)·n + 2 | n - 2 | 0.5·n^-2 | read B[i9, i7] (i0=0, i7=0, i8=0) |
| n^2 | 0.242 | ramp | (3/2)·n - 2  →  (1/8)·n^2 + (1/8)·n + 1 | n - 2 | 0.5·n^-2 | read B[i5, i6] (i0=0) |
| n^2 | 0.182 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n | 0.375·n^-2 | write D[i7, i7] (i0=0) |
| n^2 | 0.182 | ramp | n + 4  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n - 1 | 0.375·n^-2 | write D[i8, i7] (i0=0, i8=0) |
| n^2 | 0.182 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 2 | (3/4)·n - 1 | 0.375·n^-2 | write D[i8, i7] (i0=0) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 + (-1/8)·n | 0.0625/n | read B[i9, i8] (i0=0, i8=6) |
| n^2 | 0.0561 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-7/4)·n + 2 | (1/4)·n - 4 | 0.125·n^-2 | read A[i6] (i0=0, i5=0); read C[i6] (i0=0, i5=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | read B[i5, i6] (i0=0) |
| n^2 | 0.0415 | ramp | (1/8)·n^2 + (-7/8)·n + 3  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3] (i0=0, i4=0) |
| n^2 | 0.0412 | ramp | (1/8)·n^2 + (1/4)·n + 18  →  (1/8)·n^2 + (11/8)·n - 18 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i9, i8] (i0=0, i7=0, i9=0) |
| n^2 | 0.0302 | ramp | 2·n + 11  →  (1/8)·n^2 + (9/8)·n - 7 | (1/8)·n - 1 | 0.0625·n^-2 | write D[i8, i7] (i0=0, i8=0) |
| n^2 | 0.0302 | ramp | 2·n + 10  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n - 1 | 0.0625·n^-2 | write D[i8, i7] (i0=0) |
| n^2 | 0.0302 | ramp | 2·n + 10  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n - 1 | 0.0625·n^-2 | write D[i7, i7] (i0=0) |
| n^2 | 0.0296 | ramp | 4·n + 19  →  (1/8)·n^2 + (9/8)·n - 17 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i9, i8] (i0=0, i9=0) |
| n^2 | 0.029 | ramp | (27/8)·n + 9  →  (1/8)·n^2 + (1/8)·n - 15 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i9, i8] (i0=0, i7=0) |
| n^2 | 0.029 | ramp | (19/8)·n - 2  →  (1/8)·n^2 + (-3/4)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read B[i5, i6] (i0=0, i5=0) |
| n^2 | 0.0288 | ramp | 3·n + 20  →  (1/8)·n^2 + (1/8)·n - 16 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i9, i8] (i0=0) |
| n^1.5 | 6 | level | n + 3 | 6·n - 12 | 3·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.73 | level | 3·n + 12 | n - 1 | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.73 | level | 3·n + 5 | n - 1 | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.58 | ramp | 2·n + 12  →  3·n + 9 | n - 2 | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=7) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0); read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.41 | level | 2·n + 4 | n - 1 | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.41 | level | 2·n + 5 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 6 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 7 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 8 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 9 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 10 | n - 1 | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=6) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i9, i8] (i0=0, i8=0, i9=0); read B[i9, i8] (i0=0, i8=0) |
| n^1.5 | 1.24 | level | 2·n + 2 | (7/8)·n - 7 | 0.438·n^-2 | read B[i9, i7] (i0=0) |
| n^1.5 | 1.22 | level | (3/8)·n + 1 | 2·n - 2 | 1·n^-2 | read A[i6] (i0=0); read C[i6] (i0=0) |
| n^1.5 | 1.22 | level | (3/8)·n + 1 | 2·n - 2 | 1·n^-2 | read A[i6] (i0=0, i6=0); read C[i6] (i0=0, i6=0) |
| n^1.5 | 1.21 | ramp | n + 4  →  2·n + 1 | n - 2 | 0.5·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1.5 | 1.06 | level | 2·n + 3 | (3/4)·n - 6 | 0.375·n^-2 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^1.5 | 0.875 | level | n + 3 | (7/8)·n - 7 | 0.438·n^-2 | read B[i9, i7] (i0=0, i9=0) |
| n^1.5 | 0.875 | level | n + 2 | (7/8)·n | 0.438·n^-2 | read B[i4, i3] (i0=0) |
| n^1.5 | 0.875 | level | n + 1 | (7/8)·n | 0.438·n^-2 | read B[i2, i1] (i0=0) |
| n^1.5 | 0.75 | level | n + 4 | (3/4)·n - 6 | 0.375·n^-2 | read B[i9, i7] (i0=0, i8=0) |
| n^1 | 1.75 | level | 1 | (7/4)·n | 0.875·n^-2 | write A[i1] (i0=0); write C[i3] (i0=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.438·n^-2 | read A[i3] (i0=0, i4=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.438·n^-2 | write D[i7, i8] (i0=0, i8=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (-3/4)·n + 1 | 2 | 1·n^-3 | read A[i6] (i0=0, i5=0, i6=0); read C[i6] (i0=0, i5=0, i6=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n - 9 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 9 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i7=0, i8=7, i9=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i7=0, i8=0, i9=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 1 | 1 | 0.5·n^-3 | read B[i4, i3] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | read B[i5, i6] (i0=0, i6=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 1 | 1 | 0.5·n^-3 | read A[i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.5·n^-3 | read B[i5, i6] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 2 | 1 | 0.5·n^-3 | read A[i3] (i0=0, i3=0, i4=0) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i9, i7] (i0=0, i8=0) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 1.73 | level | 3·n + 10 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i8=7, i9=0) |
| n^0.5 | 1.54 | level | (19/8)·n + 3 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i7=0, i8=7) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=0) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0, i9=0) |
| n^0.5 | 1.41 | level | 2·n + 10 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i8=6, i9=0); read B[i9, i8] (i0=0, i9=0) (+1) |
| n^0.5 | 1.41 | level | 2·n + 11 | 1 | 0.5·n^-3 | read B[i9, i8] (i0=0, i8=7) |
| n^0.5 | 1.22 | level | (3/8)·n | 2 | 1·n^-3 | read B[i5, i6] (i0=0, i5=0); read A[i6] (i0=0, i5=0) (+1) |
| n^0.5 | 1.17 | level | (11/8)·n - 2 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i7=0, i8=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.5·n^-3 | read B[i5, i6] (i0=0, i5=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0) |
| n^0.5 | 1 | level | n + 2 | 1 | 0.5·n^-3 | write D[i8, i7] (i0=0, i8=0) |
| n^0.5 | 1 | level | n + 4 | 1 | 0.5·n^-3 | read B[i9, i7] (i0=0, i8=0) |
| n^0 | 1 | level | 1 | 1 | 0.5·n^-3 | write D[p0 - 1, p0 - 1] (i0=0) |

Structurally identical to covariance after normalization: same 4n → (1/8)n^2 ramps on `read B[i9,i8]`, same 2n-line column levels, d = 4.0, headroom +1.0.

## covariance — infinite-repeat  [`exact`]

Accesses $A(n) = 2·n^3 + (21/2)·n^2 + (11/2)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0161·n^4  +  1.33·n^3.5  +  2.17·n^3  +  19.5·n^2.5  +  39.3·n^2  +  28.7·n^1.5  +  39.9·n^1  +  31.5·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0142 | ramp | 4·n + 22  →  (1/8)·n^2 + (9/8)·n - 8 | (7/128)·n^3 + (-35/16)·n^2 + 21·n | 0.0273 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^4 | 0.00193 | ramp | 5·n + 29  →  (1/8)·n^2 + (9/8)·n - 16 | (1/128)·n^3 + (-7/16)·n^2 + 6·n | 0.00391 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3.5 | 0.619 | level | 2·n + 2 | (7/16)·n^3 + (-63/8)·n^2 + 35·n | 0.219 | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^3.5 | 0.464 | level | 2·n + 2 | (21/64)·n^3 + (-357/64)·n^2 + (105/4)·n - 21 | 0.164 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3.5 | 0.0884 | level | 2·n + 3 | (1/16)·n^3 + (-19/8)·n^2 + 22·n | 0.0312 | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^3.5 | 0.0773 | level | 2·n + 2 | (7/128)·n^3 + (-7/16)·n^2 | 0.0273 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3.5 | 0.0773 | level | 2·n + 2 | (7/128)·n^3 + (-21/16)·n^2 + 7·n | 0.0273 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3 | 0.663 | level | 3 | (49/128)·n^3 + (-91/16)·n^2 + 21·n | 0.191 | write C[i5, i6] (i0=0, i7=0); write C[i5, i6] (i0=0) |
| n^3 | 0.383 | level | 1 | (49/128)·n^3 + (-91/16)·n^2 + 21·n | 0.191 | read C[i5, i6] (i0=0, i7=0); read C[i5, i6] (i0=0) |
| n^3 | 0.21 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 8 | (7/8)·n^2 - 14·n | 0.438/n | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3 | 0.15 | ramp | 3·n + 14  →  (1/8)·n^2 + (9/8)·n - 9 | (5/8)·n^2 - 10·n | 0.312/n | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3 | 0.109 | ramp | n + 5  →  (1/8)·n^2 + (9/8)·n - 1 | (49/128)·n^2 + (-33/16)·n + 2 | 0.191/n | write C[i6, i5] (i0=0) |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-21/16)·n^2 + 7·n | 0.0273 | write C[i5, i6] (i0=0, i7=0); write C[i5, i6] (i0=0) |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-7/16)·n^2 | 0.0273 | write C[i5, i6] (i0=0, i7=0); write C[i5, i6] (i0=0) |
| n^3 | 0.0547 | level | 1 | (7/128)·n^3 + (-21/16)·n^2 + 7·n | 0.0273 | read C[i5, i6] (i0=0, i7=0); read C[i5, i6] (i0=0) |
| n^3 | 0.0547 | level | 1 | (7/128)·n^3 + (-7/16)·n^2 | 0.0273 | read C[i5, i6] (i0=0, i7=0); read C[i5, i6] (i0=0) |
| n^3 | 0.0467 | ramp | (1/8)·n^2 + (1/8)·n + 32  →  (1/4)·n^2 - 4·n + 36 | (1/8)·n^2 - 3·n | 0.0625/n | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^3 | 0.0359 | ramp | (19/8)·n - 4  →  (1/8)·n^2 + (1/8)·n - 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0625/n | read B[i3, i4] (i0=0) |
| n^3 | 0.0354 | ramp | (27/8)·n + 12  →  (1/8)·n^2 + (5/4)·n - 19 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0625/n | read B[i7, i6] (i0=0, i5=0) |
| n^3 | 0.03 | ramp | 3·n + 19  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i6=7) |
| n^3 | 0.0299 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 14 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i7, i6] (i0=0, i6=1, i7=0); read B[i7, i6] (i0=0, i6=1) |
| n^3 | 0.0296 | ramp | 3·n + 12  →  (1/8)·n^2 + (9/8)·n - 15 | (1/8)·n^2 - 3·n + 1 | 0.0625/n | read B[i7, i5] (i0=0, i6=0, i7=0); read B[i7, i5] (i0=0, i6=0) |
| n^3 | 0.0296 | ramp | 4·n + 20  →  (1/8)·n^2 + (9/8)·n - 16 | (1/8)·n^2 - 3·n | 0.0625/n | read B[i7, i6] (i0=0, i6=8, i7=0); read B[i7, i6] (i0=0, i6=8) |
| n^3 | 0.029 | ramp | 3·n + 21  →  (1/8)·n^2 + (9/8)·n - 18 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0625/n | read B[i7, i6] (i0=0) |
| n^3 | 0.0273 | level | (1/4)·n^2 + (1/8)·n | (7/128)·n^2 + (-37/16)·n + 24 | 0.0273/n | write C[i5, i6] (i0=0) |
| n^3 | 0.0237 | ramp | (1/4)·n^2 + (-3/4)·n + 21  →  (1/4)·n^2 + (1/8)·n - 13 | (7/128)·n^2 + (-37/16)·n + 18 | 0.0273/n | write C[i6, i5] (i0=0) |
| n^3 | 0.0154 | ramp | 2·n + 12  →  (1/8)·n^2 + (9/8)·n - 1 | (7/128)·n^2 + (-9/16)·n + 1 | 0.0273/n | write C[i6, i5] (i0=0) |
| n^3 | 0.0135 | level | 3 | (1/128)·n^3 + (-1/16)·n^2 | 0.00391 | read B[i2, i1] (i0=0); read A[i4] (i0=0, i4=0) (+3) |
| n^3 | 0.00781 | level | 1 | (1/128)·n^3 + (-1/16)·n^2 + n | 0.00391 | read B[i7, i6] (i0=0, i5=0, i6=0); read C[i5, i6] (i0=0, i7=0) (+1) |
| n^3 | 0.00391 | level | (1/4)·n^2 + (1/8)·n | (1/128)·n^2 + (-9/16)·n + 10 | 0.00391/n | write C[i5, i6] (i0=0) |
| n^3 | 0.00323 | ramp | (1/4)·n^2 + (-3/4)·n + 22  →  (1/4)·n^2 + (1/8)·n - 13 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00391/n | write C[i6, i5] (i0=0) |
| n^2.5 | 4.95 | level | 2·n + 2 | (7/2)·n^2 - 28·n | 1.75/n | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^2.5 | 2.65 | level | 2·n + 2 | (15/8)·n^2 - 15·n | 0.938/n | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^2.5 | 2.62 | level | n + 2 | (21/8)·n^2 | 1.31/n | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i7, i5] (i0=0, i6=9, i7=0); read B[i7, i5] (i0=0, i6=9) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i7, i6] (i0=0, i6=8, i7=0); read B[i7, i6] (i0=0, i6=8) |
| n^2.5 | 1.06 | level | 2·n + 3 | (3/4)·n^2 - 12·n | 0.375/n | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^2.5 | 1.06 | level | 2·n + 2 | (3/4)·n^2 - 6·n | 0.375/n | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i6=7) |
| n^2.5 | 1.04 | ramp | n + 5  →  2·n + 2 | (7/8)·n^2 + (-35/4)·n + 14 | 0.438/n | read B[i7, i5] (i0=0, i6=0) |
| n^2.5 | 1.04 | ramp | n + 4  →  2·n + 1 | (7/8)·n^2 + (-35/4)·n + 14 | 0.438/n | read B[i7, i5] (i0=0) |
| n^2.5 | 0.875 | level | n + 1 | (7/8)·n^2 + (-7/8)·n | 0.438/n | read B[i7, i5] (i0=0, i6=1, i7=0); read B[i7, i5] (i0=0, i6=1) |
| n^2.5 | 0.875 | level | n + 1 | (7/8)·n^2 + (-7/8)·n | 0.438/n | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2.5 | 0.464 | level | 2·n + 2 | (21/64)·n^2 + (-21/4)·n + 21 | 0.164/n | read B[i7, i6] (i0=0) |
| n^2.5 | 0.177 | level | 2·n + 3 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i7, i5] (i0=0, i6=9, i7=0); read B[i7, i5] (i0=0, i6=9) |
| n^2.5 | 0.148 | ramp | n + 3  →  2·n | (1/8)·n^2 + (-5/4)·n + 2 | 0.0625/n | read B[i7, i5] (i0=0, i6=1) |
| n^2.5 | 0.0625 | level | (1/4)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0625/n | read A[i4] (i0=0) |
| n^2 | 4.5 | level | 1 | (9/2)·n^2 - 15·n | 2.25/n | read C[i5, i6] (i0=0, i7=0); read C[i5, i6] (i0=0) |
| n^2 | 3.71 | level | 2 | (21/8)·n^2 | 1.31/n | write C[i5, i6] (i0=0, i7=0); write C[i5, i6] (i0=0) |
| n^2 | 3.25 | level | 3 | (15/8)·n^2 - 15·n | 0.938/n | write C[i5, i6] (i0=0, i7=0); write C[i5, i6] (i0=0) |
| n^2 | 2.62 | level | 1 | (21/8)·n^2 + (49/8)·n + 1 | 1.31/n | write A[i1] (i0=0); read A[i1] (i0=0) (+5) |
| n^2 | 2.47 | level | 2 | (7/4)·n^2 | 0.875/n | read B[i2, i1] (i0=0); write A[i1] (i0=0) (+3) |
| n^2 | 1.88 | level | 1 | (15/8)·n^2 | 0.938/n | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^2 | 1.41 | level | 2 | n^2 | 0.5/n | read A[i4] (i0=0, i3=0); read A[i4] (i0=0) (+1) |
| n^2 | 1.31 | level | 1 | (21/16)·n^2 - 7·n + 28 | 0.656/n | read C[i5, i6] (i0=0); write C[i5, i6] (i0=0) |
| n^2 | 1.3 | level | 3 | (3/4)·n^2 - 6·n | 0.375/n | write C[i5, i6] (i0=0, i7=0); write C[i5, i6] (i0=0) |
| n^2 | 1.3 | level | 3 | (3/4)·n^2 - 6·n | 0.375/n | write C[i5, i6] (i0=0, i6=7, i7=0); write C[i5, i6] (i0=0, i6=7) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.438/n | write B[i3, i4] (i0=0) |
| n^2 | 1.06 | level | 2 | (3/4)·n^2 | 0.375/n | write C[i5, i6] (i0=0, i7=0); write C[i5, i6] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.438/n | read B[i3, i4] (i0=0) |
| n^2 | 0.75 | level | 1 | (3/4)·n^2 - 6·n | 0.375/n | read C[i5, i6] (i0=0, i7=0); read C[i5, i6] (i0=0) |
| n^2 | 0.75 | level | 1 | (3/4)·n^2 | 0.375/n | read C[i5, i6] (i0=0, i7=0); read C[i5, i6] (i0=0) |
| n^2 | 0.75 | level | 1 | (3/4)·n^2 - 6·n | 0.375/n | read C[i5, i6] (i0=0, i6=7, i7=0); read C[i5, i6] (i0=0, i6=7) |
| n^2 | 0.75 | level | 1 | (3/4)·n^2 | 0.375/n | read B[i7, i6] (i0=0, i6=1) |
| n^2 | 0.625 | level | 1 | (5/8)·n^2 | 0.312/n | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^2 | 0.541 | level | 2 | (49/128)·n^2 + (-7/16)·n | 0.191/n | write C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 - 2·n + 11 | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i1=0, i2=0); read B[i2, i1] (i0=0, i1=0) |
| n^2 | 0.438 | level | (1/4)·n^2 + (1/8)·n | (7/8)·n - 15 | 0.438·n^-2 | write C[i5, i6] (i0=0) |
| n^2 | 0.438 | level | (1/4)·n^2 + (1/8)·n | (7/8)·n - 15 | 0.438·n^-2 | write C[i5, i6] (i0=0) |
| n^2 | 0.375 | level | (1/4)·n^2 + (-3/4)·n + 15 | (3/4)·n - 12 | 0.375·n^-2 | write C[i6, i5] (i0=0, i5=8) |
| n^2 | 0.375 | level | (1/4)·n^2 + (-3/4)·n + 8 | (3/4)·n - 6 | 0.375·n^-2 | write C[i6, i5] (i0=0, i5=0) |
| n^2 | 0.375 | level | 1 | (3/8)·n^2 + (-1/4)·n | 0.188/n | read A[i1] (i0=0); write A[i1] (i0=0) (+5) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n - 9 | n - 1 | 0.5·n^-2 | read B[i7, i6] (i0=0, i5=0, i7=0); read B[i7, i6] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 9 | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 1 | n | 0.5·n^-2 | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n | n - 1 | 0.5·n^-2 | read B[i3, i4] (i0=0, i3=0, i4=0); read B[i3, i4] (i0=0, i4=0) |
| n^2 | 0.354 | level | 2 | (1/4)·n^2 | 0.125/n | write A[i1] (i0=0); write C[i5, i6] (i0=0, i6=0, i7=0) (+1) |
| n^2 | 0.288 | ramp | (1/4)·n^2 + (-5/8)·n + 18  →  (1/4)·n^2 + (1/8)·n - 2 | (5/8)·n - 15 | 0.312·n^-2 | write C[i6, i5] (i0=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 | 0.125/n | read C[i5, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i6=7, i7=0) (+2) |
| n^2 | 0.246 | ramp | (19/8)·n + 5  →  (1/8)·n^2 + 11 | n - 2 | 0.5·n^-2 | read B[i7, i6] (i0=0, i5=0, i6=8) |
| n^2 | 0.241 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + 2 | n - 2 | 0.5·n^-2 | read B[i7, i5] (i0=0, i5=0, i6=0) |
| n^2 | 0.241 | ramp | (11/8)·n - 2  →  (1/8)·n^2 + 1 | n - 2 | 0.5·n^-2 | read B[i3, i4] (i0=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 - n | 0.0625/n | write C[i5, i6] (i0=0, i6=7, i7=0); write C[i5, i6] (i0=0, i6=7) |
| n^2 | 0.212 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 2 | (7/8)·n - 2 | 0.438·n^-2 | write C[i6, i5] (i0=0) |
| n^2 | 0.188 | level | 1 | (3/16)·n^2 + (-5/2)·n + 8 | 0.0938/n | read C[i5, i6] (i0=0); write C[i5, i6] (i0=0) |
| n^2 | 0.182 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n | 0.375·n^-2 | write C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.182 | ramp | n + 4  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n - 1 | 0.375·n^-2 | write C[i6, i5] (i0=0, i6=1) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 | 0.0625/n | write C[i5, i6] (i0=0, i6=7, i7=0); write C[i5, i6] (i0=0, i6=7) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 - n | 0.0625/n | read C[i5, i6] (i0=0, i6=7, i7=0); read C[i5, i6] (i0=0, i6=7) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 + (1/8)·n | 0.0625/n | write C[i5, i6] (i0=0, i5=0, i6=1); write C[i5, i6] (i0=0, i6=1) (+1) |
| n^2 | 0.0773 | level | 2 | (7/128)·n^2 + (-7/16)·n | 0.0273/n | write C[i5, i6] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-3/4)·n + 15 | (1/8)·n - 3 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i5=8) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-3/4)·n + 14 | (1/8)·n - 3 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i5=8) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/8)·n | (1/8)·n - 4 | 0.0625·n^-2 | write C[i5, i6] (i0=0, i6=8) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/8)·n | (1/8)·n - 4 | 0.0625·n^-2 | write C[i5, i6] (i0=0, i5=8) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/8)·n | (1/8)·n - 3 | 0.0625·n^-2 | write C[i5, i6] (i0=0, i6=1) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/8)·n | (1/8)·n - 3 | 0.0625·n^-2 | write C[i5, i6] (i0=0, i5=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-3/4)·n + 8 | (1/8)·n - 2 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i5=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (-3/4)·n + 7 | (1/8)·n - 2 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i5=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/8)·n | (1/8)·n - 4 | 0.0625·n^-2 | write C[i5, i6] (i0=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/8)·n | (1/8)·n - 3 | 0.0625·n^-2 | write C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.0625 | level | (1/4)·n^2 + (1/8)·n | (1/8)·n - 2 | 0.0625·n^-2 | write A[i1] (i0=0) |
| n^2 | 0.0577 | ramp | (1/4)·n^2 + 3  →  (1/4)·n^2 + (1/8)·n - 1 | (1/8)·n - 3 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i6=1) |
| n^2 | 0.056 | ramp | (1/4)·n^2 + (-3/4)·n + 22  →  (1/4)·n^2 + (1/8)·n - 13 | (1/8)·n - 4 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i6=8) |
| n^2 | 0.056 | ramp | (1/4)·n^2 + (-3/4)·n + 21  →  (1/4)·n^2 + (1/8)·n - 14 | (1/8)·n - 4 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i6=7) |
| n^2 | 0.056 | ramp | (1/4)·n^2 + (-3/4)·n + 21  →  (1/4)·n^2 + (1/8)·n - 14 | (1/8)·n - 4 | 0.0625·n^-2 | write C[i6, i5] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/8)·n | (1/8)·n - 2 | 0.0625·n^-2 | read B[i3, i4] (i0=0) |
| n^2 | 0.0412 | ramp | (1/8)·n^2 + (1/8)·n + 18  →  (1/8)·n^2 + (5/4)·n - 18 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0, i5=0, i7=0) |
| n^2 | 0.0302 | ramp | 2·n + 11  →  (1/8)·n^2 + (9/8)·n - 7 | (1/8)·n - 1 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i6=1) |
| n^2 | 0.0302 | ramp | 2·n + 10  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n - 1 | 0.0625·n^-2 | write C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.0296 | ramp | 4·n + 19  →  (1/8)·n^2 + (9/8)·n - 17 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0, i7=0) |
| n^2 | 0.0289 | ramp | (13/4)·n + 12  →  (1/8)·n^2 + (1/8)·n - 16 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0, i5=0) |
| n^2 | 0.0289 | ramp | (9/4)·n - 2  →  (1/8)·n^2 + (-7/8)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read B[i3, i4] (i0=0, i3=0) |
| n^2 | 0.0288 | ramp | 3·n + 20  →  (1/8)·n^2 + (1/8)·n - 16 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0) |
| n^2 | 0.0279 | ramp | (5/4)·n - 1  →  (1/8)·n^2 + (-15/8)·n + 2 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i4] (i0=0, i3=0) |
| n^1.5 | 6 | level | n + 3 | 6·n - 12 | 3·n^-2 | read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 1.73 | level | 3·n + 12 | n - 1 | 0.5·n^-2 | read B[i7, i5] (i0=0, i6=0, i7=0); read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 1.58 | ramp | 2·n + 12  →  3·n + 9 | n - 2 | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=8) |
| n^1.5 | 1.41 | level | 2·n + 5 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 6 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 7 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 8 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 9 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^1.5 | 1.41 | level | 2·n + 10 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i6=7) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=1, i7=0); read B[i7, i6] (i0=0, i6=1) |
| n^1.5 | 1.24 | level | 2·n + 2 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0) |
| n^1.5 | 1.24 | level | 2·n + 3 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^1.5 | 1.21 | ramp | n + 4  →  2·n + 1 | n - 2 | 0.5·n^-2 | read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 1 | level | n + 2 | n | 0.5·n^-2 | read B[i7, i5] (i0=0, i6=0, i7=0); read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 0.875 | level | n + 3 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0, i7=0) |
| n^1.5 | 0.875 | level | n + 4 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 0.875 | level | n + 1 | (7/8)·n | 0.438·n^-2 | read B[i7, i5] (i0=0, i6=1) |
| n^1.5 | 0.875 | level | n + 1 | (7/8)·n | 0.438·n^-2 | read B[i2, i1] (i0=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 1 | 0.5·n^-2 | read A[i4] (i0=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 1 | 0.5·n^-2 | read A[i4] (i0=0, i4=0) |
| n^1.5 | 0.177 | level | 2·n + 1 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i7, i5] (i0=0, i6=1) |
| n^1.5 | 0.125 | level | n + 2 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i7, i5] (i0=0, i6=1, i7=0) |
| n^1 | 5.25 | level | 1 | (21/4)·n - 21 | 2.62·n^-2 | read C[i5, i6] (i0=0) |
| n^1 | 3.5 | level | (1/4)·n^2 + (1/8)·n | 7 | 3.5·n^-3 | write C[i5, i6] (i0=0) |
| n^1 | 3 | level | (1/4)·n^2 + (1/8)·n - 6 | 6 | 3·n^-3 | write C[i6, i5] (i0=0) |
| n^1 | 2.5 | level | (1/4)·n^2 + (1/8)·n | 5 | 2.5·n^-3 | write C[i6, i5] (i0=0) |
| n^1 | 2 | level | (1/4)·n^2 + (1/8)·n | 4 | 2·n^-3 | write A[i1] (i0=0); write C[i5, i6] (i0=0, i5=0, i6=8) (+2) |
| n^1 | 1 | level | (1/4)·n^2 + (1/8)·n | 2 | 1·n^-3 | write C[i5, i6] (i0=0, i5=0, i6=0); write C[i5, i6] (i0=0, i5=8, i6=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.438·n^-2 | read C[i5, i6] (i0=0, i6=1) |
| n^1 | 0.875 | level | 1 | (7/8)·n - 7 | 0.438·n^-2 | read C[i5, i6] (i0=0, i6=8) |
| n^1 | 0.75 | level | 1 | (3/4)·n - 6 | 0.375·n^-2 | read C[i5, i6] (i0=0) |
| n^1 | 0.75 | level | 1 | (3/4)·n | 0.375·n^-2 | write C[i5, i6] (i0=0, i6=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/4)·n + 15 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8, i6=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n - 6 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + n + (35/4) | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (-49/8) | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/8)·n + 12 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/2)·n + 10 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/8)·n + 8 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 6 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 4 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/4)·n + 14 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8, i6=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n - 7 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + 2 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8, i6=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n - 7 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/4)·n + 14 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (13/8)·n + (3/4) | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + n + (7/4) | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/4)·n + 7 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write C[i5, i6] (i0=0, i6=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write C[i5, i6] (i0=0, i5=8) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write C[i5, i6] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write C[i5, i6] (i0=0, i6=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-5/8)·n + 6 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/2)·n + 5 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/8)·n + 4 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 3 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 2 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + 1 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0, i6=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write C[i5, i6] (i0=0, i6=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (7/8) | 1 | 0.5·n^-3 | write C[i5, i6] (i0=0, i6=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | write A[i1] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/4)·n + 7 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0, i6=7) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-3/4)·n + 8 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i5=0, i6=8) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n - 9 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 9 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i5=0, i6=8, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 1 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i5=0, i6=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 1 | 1 | 0.5·n^-3 | read A[i4] (i0=0, i3=0, i4=0) |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0625·n^-2 | read C[i5, i6] (i0=0, i6=1) |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0625·n^-2 | read C[i5, i6] (i0=0, i6=8) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.73 | level | 3·n + 10 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i6=8, i7=0) |
| n^0.5 | 1.5 | level | (9/4)·n + 5 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i5=0, i6=8) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=1) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 11 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i6=8) |
| n^0.5 | 1.12 | level | (5/4)·n - 1 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i5=0, i6=0) |
| n^0.5 | 1.12 | level | (5/4)·n - 1 | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0, i3=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 1 | level | n + 2 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=1) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 1 | level | n + 4 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0, i3=0); read A[i4] (i0=0, i3=0) |

The column-pair sweep re-reads data columns: `read B[i7,i6]` ramps 4n → (1/8)n^2 + (9/8)n (0.016·n^4 combined), and the fixed column reuse at 2n + 2 lines carries a heavy n^3.5 band (0.62 + 0.46). Headroom +1.0; the miss curve steps first at the two-column boundary (128n bytes), then at the matrix boundary.

## covariance — single-shot  [`exact`]

Accesses $A(n) = 2·n^3 + (21/2)·n^2 + (11/2)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0161·n^4  +  1.33·n^3.5  +  2.07·n^3  +  19.5·n^2.5  +  26.5·n^2  +  28.7·n^1.5  +  3·n^1  +  30.9·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0142 | ramp | 4·n + 22  →  (1/8)·n^2 + (9/8)·n - 8 | (7/128)·n^3 + (-35/16)·n^2 + 21·n | 0.0273 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^4 | 0.00193 | ramp | 5·n + 29  →  (1/8)·n^2 + (9/8)·n - 16 | (1/128)·n^3 + (-7/16)·n^2 + 6·n | 0.00391 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3.5 | 0.619 | level | 2·n + 2 | (7/16)·n^3 + (-63/8)·n^2 + 35·n | 0.219 | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^3.5 | 0.464 | level | 2·n + 2 | (21/64)·n^3 + (-357/64)·n^2 + (105/4)·n - 21 | 0.164 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3.5 | 0.0884 | level | 2·n + 3 | (1/16)·n^3 + (-19/8)·n^2 + 22·n | 0.0312 | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^3.5 | 0.0773 | level | 2·n + 2 | (7/128)·n^3 + (-7/16)·n^2 | 0.0273 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3.5 | 0.0773 | level | 2·n + 2 | (7/128)·n^3 + (-21/16)·n^2 + 7·n | 0.0273 | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3 | 0.866 | level | 3 | (1/2)·n^3 + (-17/2)·n^2 + 36·n | 0.25 | read A[i4] (i0=0); write C[i5, i6] (i0=0) |
| n^3 | 0.5 | level | 1 | (1/2)·n^3 + 3·n^2 + (11/2)·n | 0.25 | read A[i1] (i0=0); write A[i1] (i0=0) (+7) |
| n^3 | 0.21 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 8 | (7/8)·n^2 - 14·n | 0.438/n | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3 | 0.15 | ramp | 3·n + 14  →  (1/8)·n^2 + (9/8)·n - 8 | (5/8)·n^2 + (-79/8)·n - 2 | 0.312/n | read B[i7, i6] (i0=0, i7=0); read B[i7, i6] (i0=0) |
| n^3 | 0.109 | ramp | n + 5  →  (1/8)·n^2 + (9/8)·n - 1 | (49/128)·n^2 + (-33/16)·n + 2 | 0.191/n | write C[i6, i5] (i0=0) |
| n^3 | 0.0359 | ramp | (19/8)·n - 4  →  (1/8)·n^2 + (1/8)·n - 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0625/n | read B[i3, i4] (i0=0) |
| n^3 | 0.0354 | ramp | (27/8)·n + 12  →  (1/8)·n^2 + (5/4)·n - 19 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0625/n | read B[i7, i6] (i0=0, i5=0) |
| n^3 | 0.0299 | ramp | 3·n + 13  →  (1/8)·n^2 + (9/8)·n - 14 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i7, i6] (i0=0, i6=1, i7=0); read B[i7, i6] (i0=0, i6=1) |
| n^3 | 0.0299 | ramp | 3·n + 19  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0625/n | read B[i7, i6] (i0=0, i6=7) |
| n^3 | 0.0296 | ramp | 3·n + 12  →  (1/8)·n^2 + (9/8)·n - 15 | (1/8)·n^2 - 3·n + 1 | 0.0625/n | read B[i7, i5] (i0=0, i6=0, i7=0); read B[i7, i5] (i0=0, i6=0) |
| n^3 | 0.0296 | ramp | 4·n + 20  →  (1/8)·n^2 + (9/8)·n - 16 | (1/8)·n^2 - 3·n | 0.0625/n | read B[i7, i6] (i0=0, i6=8, i7=0); read B[i7, i6] (i0=0, i6=8) |
| n^3 | 0.029 | ramp | 3·n + 21  →  (1/8)·n^2 + (9/8)·n - 18 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0625/n | read B[i7, i6] (i0=0) |
| n^3 | 0.0154 | ramp | 2·n + 12  →  (1/8)·n^2 + (9/8)·n - 1 | (7/128)·n^2 + (-9/16)·n + 1 | 0.0273/n | write C[i6, i5] (i0=0) |
| n^2.5 | 4.95 | level | 2·n + 2 | (7/2)·n^2 - 28·n | 1.75/n | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^2.5 | 2.65 | level | 2·n + 2 | (15/8)·n^2 + (-57/4)·n - 6 | 0.938/n | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^2.5 | 2.62 | level | n + 2 | (21/8)·n^2 | 1.31/n | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i7, i5] (i0=0, i6=9, i7=0); read B[i7, i5] (i0=0, i6=9) |
| n^2.5 | 1.24 | level | 2·n + 2 | (7/8)·n^2 - 7·n | 0.438/n | read B[i7, i6] (i0=0, i6=8, i7=0); read B[i7, i6] (i0=0, i6=8) |
| n^2.5 | 1.06 | level | 2·n + 3 | (3/4)·n^2 - 12·n | 0.375/n | read B[i7, i5] (i0=0, i7=0); read B[i7, i5] (i0=0) |
| n^2.5 | 1.06 | level | 2·n + 2 | (3/4)·n^2 + (-27/4)·n + 6 | 0.375/n | read B[i7, i6] (i0=0, i6=7) |
| n^2.5 | 1.04 | ramp | n + 5  →  2·n + 2 | (7/8)·n^2 + (-35/4)·n + 14 | 0.438/n | read B[i7, i5] (i0=0, i6=0) |
| n^2.5 | 1.04 | ramp | n + 4  →  2·n + 1 | (7/8)·n^2 + (-35/4)·n + 14 | 0.438/n | read B[i7, i5] (i0=0) |
| n^2.5 | 0.875 | level | n + 1 | (7/8)·n^2 + (-7/8)·n | 0.438/n | read B[i7, i5] (i0=0, i6=1, i7=0); read B[i7, i5] (i0=0, i6=1) |
| n^2.5 | 0.875 | level | n + 1 | (7/8)·n^2 + (-7/8)·n | 0.438/n | read B[i2, i1] (i0=0, i2=0); read B[i2, i1] (i0=0) |
| n^2.5 | 0.464 | level | 2·n + 2 | (21/64)·n^2 + (-21/4)·n + 21 | 0.164/n | read B[i7, i6] (i0=0) |
| n^2.5 | 0.177 | level | 2·n + 3 | (1/8)·n^2 - 2·n | 0.0625/n | read B[i7, i5] (i0=0, i6=9, i7=0); read B[i7, i5] (i0=0, i6=9) |
| n^2.5 | 0.148 | ramp | n + 3  →  2·n | (1/8)·n^2 + (-5/4)·n + 2 | 0.0625/n | read B[i7, i5] (i0=0, i6=1) |
| n^2.5 | 0.0625 | level | (1/4)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0625/n | read A[i4] (i0=0) |
| n^2 | 7.79 | level | 3 | (9/2)·n^2 - 36·n | 2.25/n | write C[i5, i6] (i0=0) |
| n^2 | 4.95 | level | 2 | (7/2)·n^2 | 1.75/n | write C[i5, i6] (i0=0) |
| n^2 | 4.24 | level | 2 | 3·n^2 | 1.5/n | write A[i1] (i0=0); read B[i3, i4] (i0=0) (+4) |
| n^2 | 2.5 | level | 1 | (5/2)·n^2 + (1/8)·n | 1.25/n | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.438/n | read A[i4] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 + (7/8)·n + 1 | 0.438/n | write A[i1] (i0=0); write C[i5, i6] (i0=0, i6=0) (+1) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.438/n | read B[i7, i6] (i0=0, i6=1) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.438/n | read B[i3, i4] (i0=0) |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-7/8)·n | 0.219/n | write C[i5, i6] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n - 9 | n - 1 | 0.5·n^-2 | read B[i7, i6] (i0=0, i5=0, i7=0); read B[i7, i6] (i0=0, i5=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/8)·n | n - 1 | 0.5·n^-2 | read B[i3, i4] (i0=0, i3=0, i4=0); read B[i3, i4] (i0=0, i4=0) |
| n^2 | 0.246 | ramp | (19/8)·n + 5  →  (1/8)·n^2 + 11 | n - 2 | 0.5·n^-2 | read B[i7, i6] (i0=0, i5=0, i6=8) |
| n^2 | 0.241 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + 2 | n - 2 | 0.5·n^-2 | read B[i7, i5] (i0=0, i5=0, i6=0) |
| n^2 | 0.241 | ramp | (11/8)·n - 2  →  (1/8)·n^2 + 1 | n - 2 | 0.5·n^-2 | read B[i3, i4] (i0=0) |
| n^2 | 0.212 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 2 | (7/8)·n - 2 | 0.438·n^-2 | write C[i6, i5] (i0=0) |
| n^2 | 0.182 | ramp | n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n | 0.375·n^-2 | write C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.182 | ramp | n + 4  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n - 1 | 0.375·n^-2 | write C[i6, i5] (i0=0, i6=1) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 + (-1/8)·n | 0.0625/n | read B[i7, i6] (i0=0, i6=7) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 - n | 0.0625/n | read B[i7, i6] (i0=0, i6=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/8)·n | (1/8)·n - 2 | 0.0625·n^-2 | read B[i3, i4] (i0=0) |
| n^2 | 0.0412 | ramp | (1/8)·n^2 + (1/8)·n + 18  →  (1/8)·n^2 + (5/4)·n - 18 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0, i5=0, i7=0) |
| n^2 | 0.0302 | ramp | 2·n + 11  →  (1/8)·n^2 + (9/8)·n - 7 | (1/8)·n - 1 | 0.0625·n^-2 | write C[i6, i5] (i0=0, i6=1) |
| n^2 | 0.0302 | ramp | 2·n + 10  →  (1/8)·n^2 + (9/8)·n - 8 | (1/8)·n - 1 | 0.0625·n^-2 | write C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.0296 | ramp | 4·n + 19  →  (1/8)·n^2 + (9/8)·n - 17 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0, i7=0) |
| n^2 | 0.0289 | ramp | (13/4)·n + 12  →  (1/8)·n^2 + (1/8)·n - 16 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0, i5=0) |
| n^2 | 0.0289 | ramp | (9/4)·n - 2  →  (1/8)·n^2 + (-7/8)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read B[i3, i4] (i0=0, i3=0) |
| n^2 | 0.0288 | ramp | 3·n + 20  →  (1/8)·n^2 + (1/8)·n - 16 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i7, i6] (i0=0) |
| n^2 | 0.0279 | ramp | (5/4)·n - 1  →  (1/8)·n^2 + (-15/8)·n + 2 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i4] (i0=0, i3=0) |
| n^1.5 | 6 | level | n + 3 | 6·n - 12 | 3·n^-2 | read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 1.73 | level | 3·n + 12 | n - 1 | 0.5·n^-2 | read B[i7, i5] (i0=0, i6=0, i7=0); read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 1.58 | ramp | 2·n + 12  →  3·n + 9 | n - 2 | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=8) |
| n^1.5 | 1.41 | level | 2·n + 5 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 6 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 7 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 8 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 9 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^1.5 | 1.41 | level | 2·n + 10 | n - 1 | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=7) |
| n^1.5 | 1.41 | level | 2·n + 4 | n | 0.5·n^-2 | read B[i7, i6] (i0=0, i6=1, i7=0); read B[i7, i6] (i0=0, i6=1) |
| n^1.5 | 1.24 | level | 2·n + 2 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0) |
| n^1.5 | 1.24 | level | 2·n + 3 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^1.5 | 1.21 | ramp | n + 4  →  2·n + 1 | n - 2 | 0.5·n^-2 | read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 1 | level | n + 2 | n | 0.5·n^-2 | read B[i7, i5] (i0=0, i6=0, i7=0); read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 0.875 | level | n + 1 | (7/8)·n | 0.438·n^-2 | read B[i7, i5] (i0=0, i6=1) |
| n^1.5 | 0.875 | level | n + 3 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0, i7=0) |
| n^1.5 | 0.875 | level | n + 4 | (7/8)·n - 7 | 0.438·n^-2 | read B[i7, i5] (i0=0, i6=0) |
| n^1.5 | 0.875 | level | n + 1 | (7/8)·n | 0.438·n^-2 | read B[i2, i1] (i0=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 1 | 0.5·n^-2 | read A[i4] (i0=0) |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 1 | 0.5·n^-2 | read A[i4] (i0=0, i4=0) |
| n^1.5 | 0.177 | level | 2·n + 1 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i7, i5] (i0=0, i6=1) |
| n^1.5 | 0.125 | level | n + 2 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i7, i5] (i0=0, i6=1, i7=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.438·n^-2 | write C[i5, i6] (i0=0, i6=1) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n - 9 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 9 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i5=0, i6=8, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 1 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i5=0, i6=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 1 | 1 | 0.5·n^-3 | read A[i4] (i0=0, i3=0, i4=0) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 6 | level | n + 3 | 6 | 3·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.73 | level | 3·n + 10 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i6=8, i7=0) |
| n^0.5 | 1.5 | level | (9/4)·n + 5 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i5=0, i6=8) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=1) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 2 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0, i7=0) |
| n^0.5 | 1.41 | level | 2·n + 10 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i6=7, i7=0); read B[i7, i6] (i0=0, i7=0) (+1) |
| n^0.5 | 1.41 | level | 2·n + 11 | 1 | 0.5·n^-3 | read B[i7, i6] (i0=0, i6=8) |
| n^0.5 | 1.12 | level | (5/4)·n - 1 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i5=0, i6=0) |
| n^0.5 | 1.12 | level | (5/4)·n - 1 | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0, i3=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.5·n^-3 | read B[i7, i5] (i0=0, i6=0) |
| n^0.5 | 1 | level | n + 2 | 1 | 0.5·n^-3 | write C[i6, i5] (i0=0, i6=1) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.5·n^-3 | read B[i3, i4] (i0=0, i3=0); read A[i4] (i0=0, i3=0) |

The column-pair sweep re-reads data columns: `read B[i7,i6]` ramps 4n → (1/8)n^2 + (9/8)n (0.016·n^4 combined), and the fixed column reuse at 2n + 2 lines carries a heavy n^3.5 band (0.62 + 0.46). Headroom +1.0; the miss curve steps first at the two-column boundary (128n bytes), then at the matrix boundary.

## doitgen — infinite-repeat  [`exact`]

Accesses $A(n) = 4·n^4 + 3·n^3$ (exact on n ≡ 0 mod 8); DMD order $n^{5}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^5  +  1.1·n^4.5  +  5.14·n^4  +  4.78·n^3.5  +  7.07·n^3  +  1.5·n^2.5  +  1.73·n^2

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^5 | 0.0387 | level | (1/8)·n^2 + (3/8)·n | (7/64)·n^4 + (-15/8)·n^3 + 2·n^2 | 0.0273 | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^5 | 0.00552 | level | (1/8)·n^2 + (3/8)·n | (1/64)·n^4 + (-3/8)·n^3 + 2·n^2 | 0.00391 | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^4.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^4 + (-7/8)·n^3 | 0.191 | read C[i4, i3] (i0=0) |
| n^4.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^4 + (-7/8)·n^3 | 0.0273 | read C[i4, i3] (i0=0) |
| n^4.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^4 + (-7/4)·n^3 | 0.0273 | read B[i1, i2, i4] (i0=0) |
| n^4.5 | 0.0442 | level | (1/8)·n^3 + (1/8)·n^2 + (1/8)·n | (1/8)·n^3 - 2·n^2 | 0.0312/n | read B[i1, i2, i4] (i0=0, i1=0, i2=0, i3=0); read B[i1, i2, i4] (i0=0, i1=0, i3=0) (+2) |
| n^4.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^4 + (-3/8)·n^3 + 2·n^2 | 0.00391 | read B[i1, i2, i4] (i0=0) |
| n^4 | 1.73 | level | 3 | n^4 - n^2 | 0.25 | read C[i4, i3] (i0=0, i1=0, i2=0, i3=0, i4=0); read A[i3] (i0=0, i1=0, i4=0) (+5) |
| n^4 | 1.52 | level | 3 | (7/8)·n^4 | 0.219 | read B[i1, i2, i4] (i0=0) |
| n^4 | 0.875 | level | 1 | (7/8)·n^4 + (7/8)·n^3 | 0.219 | write A[i3] (i0=0, i1=0, i2=0); write A[i3] (i0=0, i1=0) (+1) |
| n^4 | 0.309 | ramp | (1/8)·n^2 + (1/4)·n + 1  →  (1/8)·n^2 + (3/8)·n | (7/8)·n^3 - n^2 | 0.219/n | read C[i4, i3] (i0=0, i1=0, i2=0, i3=0); read C[i4, i3] (i0=0, i1=0, i3=0) (+2) |
| n^4 | 0.309 | level | (1/8)·n^2 + (3/8)·n | (7/8)·n^3 - 7·n^2 | 0.219/n | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^4 | 0.125 | level | 1 | (1/8)·n^4 | 0.0312 | write A[i3] (i0=0, i1=0, i2=0, i4=0); write A[i3] (i0=0, i1=0, i2=0) (+5) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (21/8) | (1/8)·n^3 + (-9/8)·n^2 | 0.0312/n | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - 2·n^2 | 0.0312/n | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - n^2 | 0.0312/n | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - 2·n^2 | 0.0312/n | read C[i4, i3] (i0=0, i1=0, i2=0, i4=0); read C[i4, i3] (i0=0, i1=0, i4=0) (+2) |
| n^4 | 0.0432 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - n^2 | 0.0312/n | read C[i4, i3] (i0=0, i1=0, i2=0, i3=0); read C[i4, i3] (i0=0, i1=0, i3=0) (+2) |
| n^4 | 0.0281 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-7/4)·n + 2 | (1/8)·n^3 - 2·n^2 | 0.0312/n | write A[i3] (i0=0, i1=0, i2=0); write A[i3] (i0=0, i1=0) (+3) |
| n^4 | 0.028 | ramp | (5/4)·n  →  (1/8)·n^2 + (-7/4)·n | (1/8)·n^3 - 2·n^2 | 0.0312/n | read A[i5] (i0=0, i1=0, i2=0); read A[i5] (i0=0, i1=0) (+1) |
| n^3.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^3 | 0.219/n | read C[i4, i3] (i0=0) |
| n^3.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^3 | 0.219/n | read C[i4, i3] (i0=0, i4=0) |
| n^3.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^3 | 0.219/n | read B[i1, i2, i4] (i0=0) |
| n^3.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^3 | 0.219/n | read B[i1, i2, i4] (i0=0, i4=0) |
| n^3.5 | 0.354 | level | (1/8)·n^3 + (1/8)·n^2 + (1/8)·n | n^2 | 0.25·n^-2 | read B[i1, i2, i4] (i0=0, i1=0, i2=0, i3=0, i4=0); read B[i1, i2, i4] (i0=0, i1=0, i3=0, i4=0) (+2) |
| n^3.5 | 0.354 | level | (1/8)·n^3 + (1/8)·n^2 + (1/8)·n | n^2 | 0.25·n^-2 | read B[i1, i2, i4] (i0=0, i1=0, i2=0, i3=0); read B[i1, i2, i4] (i0=0, i1=0, i3=0) (+2) |
| n^3.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^3 - n^2 | 0.0312/n | read B[i1, i2, i4] (i0=0) |
| n^3.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^3 | 0.0312/n | read B[i1, i2, i4] (i0=0, i4=0); write B[i1, i2, i5] (i0=0) |
| n^3.5 | 0.0964 | ramp | (1/4)·n + 9  →  (9/8)·n - 12 | (1/8)·n^3 - 2·n^2 | 0.0312/n | write B[i1, i2, i5] (i0=0) |
| n^3 | 2.12 | level | (1/8)·n^2 + (3/8)·n | 6·n^2 | 1.5·n^-2 | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^3 | 1.24 | level | 2 | (7/8)·n^3 | 0.219/n | write B[i1, i2, i5] (i0=0) |
| n^3 | 1.24 | level | 2 | (7/8)·n^3 | 0.219/n | read A[i5] (i0=0, i1=0, i2=0); read A[i5] (i0=0, i1=0) (+1) |
| n^3 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (21/8) | n^2 | 0.25·n^-2 | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n^2 | 0.25·n^-2 | read C[i4, i3] (i0=0, i1=0, i2=0); read C[i4, i3] (i0=0, i1=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n^2 | 0.25·n^-2 | read C[i4, i3] (i0=0, i1=0, i2=0, i4=0); read C[i4, i3] (i0=0, i1=0, i4=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n^2 | 0.25·n^-2 | read C[i4, i3] (i0=0, i1=0, i2=0, i3=0); read C[i4, i3] (i0=0, i1=0, i3=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n^2 | 0.25·n^-2 | read C[i4, i3] (i0=0, i1=0, i2=0, i3=0, i4=0); read C[i4, i3] (i0=0, i1=0, i3=0, i4=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 1 | n^2 | 0.25·n^-2 | write A[i3] (i0=0, i1=0, i2=0); write A[i3] (i0=0, i1=0) (+3) |
| n^3 | 0.354 | level | (1/8)·n^2 + (-3/4)·n | n^2 | 0.25·n^-2 | read A[i5] (i0=0, i1=0, i2=0, i5=0); read A[i5] (i0=0, i1=0, i5=0) (+1) |
| n^2.5 | 0.5 | level | (1/4)·n | n^2 | 0.25·n^-2 | write A[i3] (i0=0, i1=0, i2=0, i3=0); read B[i1, i2, i4] (i0=0, i1=0, i2=0, i3=0, i4=0) (+4) |
| n^2.5 | 0.5 | level | (1/4)·n + 1 | n^2 | 0.25·n^-2 | write B[i1, i2, i5] (i0=0) |
| n^2.5 | 0.5 | level | (1/4)·n - 1 | n^2 | 0.25·n^-2 | read A[i5] (i0=0, i1=0, i2=0); read A[i5] (i0=0, i1=0) (+1) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 | 0.219·n^-2 | read A[i3] (i0=0, i1=0, i2=0, i4=0); read A[i3] (i0=0, i1=0, i2=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 | 0.0312·n^-2 | read A[i3] (i0=0, i1=0, i2=0, i4=0); read A[i3] (i0=0, i1=0, i2=0) |

The contraction buffer `read C[i4,i3]` is re-read per (r,q) pair at (1/8)n^2 + (3/8)n lines — gemm's boundary — but the access count is n^4, so the term order is n^5 (0.0387 + 0.0055 ≈ 0.044, gemm's constant one dimension up). The n^4.5 band is the row-window reuse at (9/8)n + 1 lines.

## doitgen — single-shot  [`exact`]

Accesses $A(n) = 4·n^4 + 3·n^3$ (exact on n ≡ 0 mod 8); DMD order $n^{5}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^5  +  1.06·n^4.5  +  5.14·n^4  +  4.07·n^3.5  +  7.95·n^3  +  1.5·n^2.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^5 | 0.0387 | level | (1/8)·n^2 + (3/8)·n | (7/64)·n^4 + (-15/8)·n^3 + (121/64)·n^2 + (15/8)·n - 2 | 0.0273 | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^5 | 0.00552 | level | (1/8)·n^2 + (3/8)·n | (1/64)·n^4 + (-3/8)·n^3 + (127/64)·n^2 + (3/8)·n - 2 | 0.00391 | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^4.5 | 0.812 | level | (9/8)·n + 1 | (49/64)·n^4 + (-7/8)·n^3 | 0.191 | read C[i4, i3] (i0=0) |
| n^4.5 | 0.116 | level | (9/8)·n + 1 | (7/64)·n^4 + (-7/8)·n^3 | 0.0273 | read C[i4, i3] (i0=0) |
| n^4.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^4 + (-7/4)·n^3 | 0.0273 | read B[i1, i2, i4] (i0=0) |
| n^4.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^4 + (-3/8)·n^3 + 2·n^2 | 0.00391 | read B[i1, i2, i4] (i0=0) |
| n^4 | 1.73 | level | 3 | n^4 | 0.25 | read B[i1, i2, i4] (i0=0, i4=0); read A[i3] (i0=0, i4=0) (+2) |
| n^4 | 1.52 | level | 3 | (7/8)·n^4 | 0.219 | read B[i1, i2, i4] (i0=0) |
| n^4 | 1 | level | 1 | n^4 | 0.25 | write A[i3] (i0=0); read A[i5] (i0=0) |
| n^4 | 0.309 | ramp | (1/8)·n^2 + (1/4)·n + 1  →  (1/8)·n^2 + (3/8)·n | (7/8)·n^3 - n^2 + (-7/8)·n + 1 | 0.219/n | read C[i4, i3] (i0=0, i2=0, i3=0); read C[i4, i3] (i0=0, i3=0) |
| n^4 | 0.309 | level | (1/8)·n^2 + (3/8)·n | (7/8)·n^3 - 7·n^2 + (-7/8)·n + 7 | 0.219/n | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (5/4)·n + (21/8) | (1/8)·n^3 + (-9/8)·n^2 + (-1/8)·n + (9/8) | 0.0312/n | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - 2·n^2 + (-1/8)·n + 2 | 0.0312/n | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - n^2 + (-1/8)·n + 1 | 0.0312/n | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^4 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - 2·n^2 + (-1/8)·n + 2 | 0.0312/n | read C[i4, i3] (i0=0, i2=0, i4=0); read C[i4, i3] (i0=0, i4=0) |
| n^4 | 0.0432 | ramp | (1/8)·n^2 + (1/4)·n + 2  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^3 - n^2 + (-1/8)·n + 1 | 0.0312/n | read C[i4, i3] (i0=0, i2=0, i3=0); read C[i4, i3] (i0=0, i3=0) |
| n^4 | 0.0281 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-7/4)·n + 2 | (1/8)·n^3 - 2·n^2 + (-1/8)·n + 2 | 0.0312/n | write A[i3] (i0=0, i2=0); write A[i3] (i0=0) |
| n^4 | 0.028 | ramp | (5/4)·n  →  (1/8)·n^2 + (-7/4)·n | (1/8)·n^3 - 2·n^2 | 0.0312/n | read A[i5] (i0=0) |
| n^3.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^3 | 0.219/n | read C[i4, i3] (i0=0) |
| n^3.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^3 | 0.219/n | read C[i4, i3] (i0=0, i4=0) |
| n^3.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^3 | 0.219/n | read B[i1, i2, i4] (i0=0) |
| n^3.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n^3 | 0.219/n | read B[i1, i2, i4] (i0=0, i4=0) |
| n^3.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^3 - n^2 | 0.0312/n | read B[i1, i2, i4] (i0=0) |
| n^3.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n^3 | 0.0312/n | read B[i1, i2, i4] (i0=0, i4=0); write B[i1, i2, i5] (i0=0) |
| n^3.5 | 0.0964 | ramp | (1/4)·n + 9  →  (9/8)·n - 12 | (1/8)·n^3 - 2·n^2 | 0.0312/n | write B[i1, i2, i5] (i0=0) |
| n^3 | 2.12 | level | (1/8)·n^2 + (3/8)·n | 6·n^2 - 6 | 1.5·n^-2 | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^3 | 1.24 | level | 2 | (7/8)·n^3 | 0.219/n | write B[i1, i2, i5] (i0=0) |
| n^3 | 1.24 | level | 2 | (7/8)·n^3 | 0.219/n | read A[i5] (i0=0) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 | 0.219/n | write A[i3] (i0=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (21/8) | n^2 - 1 | 0.25·n^-2 | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n^2 - 1 | 0.25·n^-2 | read C[i4, i3] (i0=0, i2=0); read C[i4, i3] (i0=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n^2 - 1 | 0.25·n^-2 | read C[i4, i3] (i0=0, i2=0, i4=0); read C[i4, i3] (i0=0, i4=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/8)·n | n^2 - 1 | 0.25·n^-2 | read C[i4, i3] (i0=0, i2=0, i3=0); read C[i4, i3] (i0=0, i3=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n^2 - 1 | 0.25·n^-2 | read C[i4, i3] (i0=0, i2=0, i3=0, i4=0); read C[i4, i3] (i0=0, i3=0, i4=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 1 | n^2 - 1 | 0.25·n^-2 | write A[i3] (i0=0, i2=0); write A[i3] (i0=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (-3/4)·n | n^2 | 0.25·n^-2 | read A[i5] (i0=0, i5=0) |
| n^2.5 | 0.5 | level | (1/4)·n | n^2 - 1 | 0.25·n^-2 | write A[i3] (i0=0, i2=0, i3=0); write A[i3] (i0=0, i3=0) |
| n^2.5 | 0.5 | level | (1/4)·n + 1 | n^2 | 0.25·n^-2 | write B[i1, i2, i5] (i0=0) |
| n^2.5 | 0.5 | level | (1/4)·n - 1 | n^2 | 0.25·n^-2 | read A[i5] (i0=0) |

The contraction buffer `read C[i4,i3]` is re-read per (r,q) pair at (1/8)n^2 + (3/8)n lines — gemm's boundary — but the access count is n^4, so the term order is n^5 (0.0387 + 0.0055 ≈ 0.044, gemm's constant one dimension up). The n^4.5 band is the row-window reuse at (9/8)n + 1 lines.

## fdtd — infinite-repeat  [`exact`]

Accesses $A(n) = 14·n^3 - 18·n^2 + 6·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.617·n^4  +  0.253·n^3.5  +  34.1·n^3  +  3.35·n^2.5  +  41.5·n^2  +  4.34·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0765 | level | (3/8)·n^2 + (5/2)·n + (1/8) | (1/8)·n^3 + (-9/4)·n^2 + (17/8)·n | 0.00893 | read D[i5, i6] (i0=0, i1=0, i5=0); read D[i5, i6] (i0=0, i1=0) (+2) |
| n^4 | 0.0765 | level | (3/8)·n^2 + (-1/8)·n + 1 | (1/8)·n^3 + (-17/8)·n^2 + 2·n | 0.00893 | read D[i5, i6] (i0=0, i1=0, i5=0); read D[i5, i6] (i0=0, i1=0) (+2) |
| n^4 | 0.0765 | level | (3/8)·n^2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read B[i7 + 1, i8] (i0=0, i1=0); read B[i7 + 1, i8] (i0=0) |
| n^4 | 0.0658 | ramp | (1/4)·n^2 + (1/4)·n + 1  →  (3/8)·n^2 - 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read C[i7, i8] (i0=0, i1=0); read C[i7, i8] (i0=0) |
| n^4 | 0.0658 | ramp | (1/4)·n^2 + (1/4)·n + 1  →  (3/8)·n^2 - 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read D[i7, i8 + 1] (i0=0, i1=0); read D[i7, i8 + 1] (i0=0) |
| n^4 | 0.0653 | ramp | (1/4)·n^2 + 2  →  (3/8)·n^2 + (-1/2)·n - 1 | (1/8)·n^3 + (-5/2)·n^2 + 8·n | 0.00893 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^4 | 0.0653 | ramp | (1/4)·n^2 + 2  →  (3/8)·n^2 + (-1/2)·n - 1 | (1/8)·n^3 + (-5/2)·n^2 + 8·n | 0.00893 | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^4 | 0.0625 | level | (1/4)·n^2 + (13/8)·n + (1/8) | (1/8)·n^3 + (-3/2)·n^2 + (27/8)·n | 0.00893 | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^4 | 0.0625 | level | (1/4)·n^2 + (-1/8)·n + 1 | (1/8)·n^3 + (-19/8)·n^2 + 6·n | 0.00893 | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^3.5 | 0.0884 | level | (1/2)·n + (5/2) | (1/8)·n^3 + (-19/8)·n^2 + (17/4)·n | 0.00893 | read B[i7, i8] (i0=0, i1=0); read B[i7, i8] (i0=0) |
| n^3.5 | 0.0884 | level | (1/2)·n + 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read B[i7, i8] (i0=0, i1=0); read B[i7, i8] (i0=0) |
| n^3.5 | 0.0765 | level | (3/8)·n + 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read C[i3 - 1, i4] (i0=0, i1=0); read C[i3 - 1, i4] (i0=0) |
| n^3 | 3.03 | level | 3 | (7/4)·n^3 + (-7/4)·n^2 | 0.125 | read C[i3 - 1, i4] (i0=0, i1=0); write B[i3, i4] (i0=0, i1=0) (+2) |
| n^3 | 3 | level | 4 | (3/2)·n^3 + (-3/2)·n^2 | 0.107 | read B[i7, i8] (i0=0, i1=0); write C[i7, i8] (i0=0, i1=0) (+2) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (-11/4)·n^2 + n | 0.125 | read B[i3, i4] (i0=0); read C[i7, i8] (i0=0) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (-15/8)·n^2 - n + 1 | 0.125 | read D[i7, i8] (i0=0, i1=0); read D[i5, i6] (i0=0) (+1) |
| n^3 | 1.75 | level | 4 | (7/8)·n^3 + (-7/8)·n^2 | 0.0625 | read B[i7 + 1, i8] (i0=0, i1=0); write C[i7, i8] (i0=0, i1=0) (+2) |
| n^3 | 1.75 | level | 4 | (7/8)·n^3 + (-7/4)·n^2 | 0.0625 | read C[i5, i6 - 1] (i0=0, i1=0); read D[i7, i8 + 1] (i0=0, i1=0) (+2) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 + (-21/8)·n^2 + (21/8)·n + (-7/8) | 0.0625 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 + (-7/4)·n^2 + (7/8)·n | 0.0625 | read C[i3, i4] (i0=0) |
| n^3 | 1.24 | level | 2 | (7/8)·n^3 - n^2 | 0.0625 | write D[i5, i6] (i0=0, i1=0); write D[i5, i6] (i0=0) |
| n^3 | 1.1 | ramp | (1/4)·n^2 + (1/8)·n  →  (3/8)·n^2 + (-1/2)·n | 2·n^2 - 8·n | 0.143/n | read B[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i1=0, i4=0) (+4) |
| n^3 | 1.06 | level | 2 | (3/4)·n^3 | 0.0536 | read C[i5, i6] (i0=0, i1=0, i5=0); read C[i5, i6] (i0=0, i1=0) (+2) |
| n^3 | 0.839 | level | 5 | (3/8)·n^3 + (-27/8)·n^2 + 3·n | 0.0268 | read B[i7 + 1, i8] (i0=0, i1=0); read B[i7, i8] (i0=0, i1=0) (+4) |
| n^3 | 0.75 | level | 1 | (3/4)·n^3 | 0.0536 | read C[i5, i6 - 1] (i0=0, i1=0); read C[i5, i6 - 1] (i0=0) |
| n^3 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (1/8) | n^2 - n | 0.0714/n | read D[i5, i6] (i0=0, i1=0, i5=0, i6=0); read D[i5, i6] (i0=0, i1=0, i6=0) (+2) |
| n^3 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | n^2 - n | 0.0714/n | read D[i5, i6] (i0=0, i1=0, i5=0, i6=0); read D[i5, i6] (i0=0, i1=0, i6=0) (+2) |
| n^3 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (-7/8) | n^2 - n | 0.0714/n | read D[i5, i6] (i0=0, i1=0, i5=0); read D[i5, i6] (i0=0, i1=0) (+2) |
| n^3 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | n^2 - n | 0.0714/n | read D[i5, i6] (i0=0, i1=0, i5=0); read D[i5, i6] (i0=0, i1=0) (+2) |
| n^3 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (1/8) | n^2 - n | 0.0714/n | read D[i5, i6] (i0=0, i1=0, i5=0); read D[i5, i6] (i0=0, i1=0) (+4) |
| n^3 | 0.612 | level | (3/8)·n^2 + (21/8)·n + 1 | n^2 | 0.0714/n | write B[0, i2] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=0, i3=0) (+7) |
| n^3 | 0.612 | level | (3/8)·n^2 | n^2 - n | 0.0714/n | read B[i7, i8] (i0=0, i1=0, i7=0); read B[i7 + 1, i8] (i0=0, i1=0) (+2) |
| n^3 | 0.612 | level | (3/8)·n^2 | n^2 - 2·n | 0.0714/n | read B[i7 + 1, i8] (i0=0, i1=0, i8=0); read B[i7 + 1, i8] (i0=0, i8=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (3/8)·n - 1  →  (3/8)·n^2 - 1 | n^2 - 2·n | 0.0714/n | read C[i7, i8] (i0=0, i1=0); read C[i7, i8] (i0=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (3/8)·n - 2  →  (3/8)·n^2 - 2 | n^2 - 2·n | 0.0714/n | read D[i7, i8 + 1] (i0=0, i1=0); read D[i7, i8 + 1] (i0=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (1/4)·n  →  (3/8)·n^2 + (-1/8)·n | n^2 - 2·n | 0.0714/n | read D[i7, i8 + 1] (i0=0, i1=0, i8=0); read D[i7, i8 + 1] (i0=0, i8=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (1/4)·n - 1  →  (3/8)·n^2 + (-1/8)·n - 1 | n^2 - 2·n | 0.0714/n | read C[i7, i8] (i0=0, i1=0, i8=0); read C[i7, i8] (i0=0, i8=0) |
| n^3 | 0.55 | ramp | (1/4)·n^2 + 1  →  (3/8)·n^2 + (-5/8)·n + 1 | n^2 - 4·n | 0.0714/n | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^3 | 0.55 | ramp | (1/4)·n^2 + 1  →  (3/8)·n^2 + (-5/8)·n + 1 | n^2 - 4·n | 0.0714/n | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^3 | 0.5 | level | (1/4)·n^2 + (13/8)·n + (1/8) | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0, i1=0, i6=0); read C[i5, i6] (i0=0, i6=0) |
| n^3 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0, i1=0, i6=0); read C[i5, i6] (i0=0, i6=0) |
| n^3 | 0.5 | level | (1/4)·n^2 + (13/8)·n + (1/8) | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^3 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^3 | 0.306 | level | 6 | (1/8)·n^3 + (-9/8)·n^2 + n | 0.00893 | read D[i7, i8 + 1] (i0=0, i1=0); read D[i7, i8 + 1] (i0=0) |
| n^3 | 0.28 | level | 5 | (1/8)·n^3 + (-9/8)·n^2 + n | 0.00893 | read D[i7, i8] (i0=0, i1=0); read D[i7, i8] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 - n^2 | 0.00893 | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 - n^2 | 0.00893 | write D[i5, i6] (i0=0, i1=0); write D[i5, i6] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 + (-1/8)·n^2 | 0.00893 | write B[i3, i4] (i0=0, i1=0); write B[i3, i4] (i0=0) |
| n^3 | 0.125 | level | 1 | (1/8)·n^3 - n | 0.00893 | read C[i5, i6 - 1] (i0=0, i1=0); read C[i5, i6 - 1] (i0=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (1/8)·n + (-1/2) | (1/8)·n^2 + (-17/8)·n | 0.00893/n | read C[i3, i4] (i0=0, i1=0, i3=0); read C[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (-1/2)·n | (1/8)·n^2 - 2·n | 0.00893/n | read C[i3, i4] (i0=0, i1=0, i3=0); read C[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (1/8)·n + (-1/2) | (1/8)·n^2 + (-17/8)·n | 0.00893/n | read B[i3, i4] (i0=0, i1=0, i3=0); read B[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (-1/2)·n | (1/8)·n^2 - 2·n | 0.00893/n | read B[i3, i4] (i0=0, i1=0, i3=0); read B[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (21/8)·n + 1 | (1/8)·n^2 + (-9/8)·n | 0.00893/n | read D[i5, i6] (i0=0, i1=0); read D[i5, i6] (i0=0, i1=1) (+1) |
| n^3 | 0.0765 | level | (3/8)·n^2 + 1 | (1/8)·n^2 - 2·n | 0.00893/n | read D[i5, i6] (i0=0, i1=0); read D[i5, i6] (i0=0, i1=1) (+1) |
| n^3 | 0.0765 | level | (3/8)·n^2 + 1 | (1/8)·n^2 - 2·n | 0.00893/n | read B[i7, i8] (i0=0, i1=0, i7=0); read B[i7, i8] (i0=0, i7=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (13/8)·n + 1 | (1/8)·n^2 + (-9/8)·n | 0.00893/n | read B[i7 + 1, i8] (i0=0, i1=0); read B[i7 + 1, i8] (i0=0) |
| n^3 | 0.0726 | ramp | (3/8)·n^2 + (-1/8)·n + 2  →  (3/8)·n^2 - 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^3 | 0.0726 | ramp | (3/8)·n^2 + (-1/8)·n + 1  →  (3/8)·n^2 - 2 | (1/8)·n^2 - 2·n | 0.00893/n | read B[i7 + 1, i8] (i0=0, i1=0, i7=0); read B[i7 + 1, i8] (i0=0, i7=0) |
| n^3 | 0.0725 | ramp | (3/8)·n^2 + (-3/8)·n + 2  →  (3/8)·n^2 + (-1/4)·n - 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i3 - 1, i4] (i0=0, i1=0, i3=0); read C[i3 - 1, i4] (i0=0, i1=1, i3=0) (+1) |
| n^3 | 0.0725 | ramp | (3/8)·n^2 + (-1/2)·n + 4  →  (3/8)·n^2 + (-1/4)·n - 2 | (1/8)·n^2 - 2·n | 0.00893/n | write B[0, i2] (i0=0, i1=0); write B[0, i2] (i0=0, i1=1) (+1) |
| n^3 | 0.0625 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/8)·n^2 + (-17/8)·n | 0.00893/n | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^3 | 0.0625 | level | (1/4)·n^2 | (1/8)·n^2 - 2·n | 0.00893/n | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^3 | 0.0625 | level | (1/4)·n^2 + (-1/8)·n + 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i5, i6] (i0=0, i1=0, i5=0); read C[i5, i6] (i0=0, i5=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 + (7/4)·n + 1 | (1/8)·n^2 + (-9/8)·n | 0.00893/n | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 + 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^3 | 0.0593 | ramp | (1/4)·n^2 + 2  →  (1/4)·n^2 + (1/4)·n - 4 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i7, i8] (i0=0, i1=0, i7=0); read C[i7, i8] (i0=0, i7=0) |
| n^3 | 0.0593 | ramp | (1/4)·n^2 + 2  →  (1/4)·n^2 + (1/4)·n - 4 | (1/8)·n^2 - 2·n | 0.00893/n | read D[i7, i8 + 1] (i0=0, i1=0, i7=0); read D[i7, i8 + 1] (i0=0, i7=0) |
| n^3 | 0.0592 | ramp | (1/4)·n^2 + (-1/4)·n + 3  →  (1/4)·n^2 - 3 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^3 | 0.0592 | ramp | (1/4)·n^2 + (-1/4)·n + 3  →  (1/4)·n^2 - 3 | (1/8)·n^2 - 2·n | 0.00893/n | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^3 | 0.0592 | ramp | (1/4)·n^2 + (-1/4)·n + 2  →  (1/4)·n^2 + (-1/8)·n - 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^2.5 | 0.707 | level | (1/2)·n + 2 | n^2 - 2·n | 0.0714/n | read B[i7, i8] (i0=0, i1=0); read B[i7, i8] (i0=0) |
| n^2.5 | 0.707 | level | (1/2)·n + (5/2) | n^2 - 2·n | 0.0714/n | read B[i7, i8] (i0=0, i1=0, i8=0); read B[i7, i8] (i0=0, i8=0) |
| n^2.5 | 0.707 | level | (1/2)·n + 2 | n^2 - 2·n | 0.0714/n | read B[i7, i8] (i0=0, i1=0, i8=0); read B[i7, i8] (i0=0, i8=0) |
| n^2.5 | 0.612 | level | (3/8)·n + 2 | n^2 - 2·n | 0.0714/n | read C[i3 - 1, i4] (i0=0, i1=0); read C[i3 - 1, i4] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n + 2 | n^2 - 2·n | 0.0714/n | read C[i3 - 1, i4] (i0=0, i1=0, i4=0); read C[i3 - 1, i4] (i0=0, i4=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-7/8)·n | 0.0625/n | read C[i3, i4] (i0=0, i1=0) |
| n^2 | 1.41 | level | 2 | n^2 - n | 0.0714/n | read A[i1] (i0=0, i2=0); read A[i1] (i0=0) |
| n^2 | 1.41 | level | 2 | n^2 | 0.0714/n | write B[0, i2] (i0=0, i1=0, i2=0); write D[i5, i6] (i0=0, i1=0) (+2) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.0625/n | write B[0, i2] (i0=0, i1=0); write B[0, i2] (i0=0) |
| n^2 | 1.22 | level | (3/8)·n^2 + (1/8)·n + (-1/2) | 2·n | 0.143·n^-2 | read B[i3, i4] (i0=0, i1=0, i3=0, i4=0); read C[i3, i4] (i0=0, i1=0, i3=0, i4=0) (+4) |
| n^2 | 1.22 | level | (3/8)·n^2 + (-1/2)·n | 2·n | 0.143·n^-2 | read B[i3, i4] (i0=0, i1=0, i3=0, i4=0); read C[i3, i4] (i0=0, i1=0, i3=0, i4=0) (+4) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 4 | 2·n | 0.143·n^-2 | read B[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i1=0, i4=0) (+4) |
| n^2 | 1 | level | (1/4)·n^2 - 1 | 2·n | 0.143·n^-2 | read B[i3, i4] (i0=0, i1=0, i4=0); read C[i7, i8] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 1 | 2·n | 0.143·n^-2 | read B[i3, i4] (i0=0, i1=0); read C[i7, i8] (i0=0, i1=0, i7=0, i8=0) (+3) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n + 1 | n | 0.0714·n^-2 | read D[i5, i6] (i0=0, i1=0); read D[i5, i6] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + 1 | n | 0.0714·n^-2 | read D[i5, i6] (i0=0, i1=0); read D[i5, i6] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/8)·n + (-3/2) | n | 0.0714·n^-2 | read B[i3, i4] (i0=0, i1=0, i3=0); read B[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/2)·n | n | 0.0714·n^-2 | read B[i3, i4] (i0=0, i1=0, i3=0); read B[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/8)·n + (-3/2) | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0, i3=0); read C[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/2)·n | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0, i3=0); read C[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n + 1 | n | 0.0714·n^-2 | read A[i1] (i0=0, i1=0); read D[i5, i6] (i0=0, i1=0, i6=0) (+2) |
| n^2 | 0.612 | level | (3/8)·n^2 + 1 | n | 0.0714·n^-2 | read A[i1] (i0=0, i1=0); read D[i5, i6] (i0=0, i1=0, i6=0) (+2) |
| n^2 | 0.612 | level | (3/8)·n^2 - 1 | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i1=0, i7=0); read B[i7 + 1, i8] (i0=0, i7=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/4)·n + (3/8) | n | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i1=0, i3=0); read C[i3 - 1, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-3/8)·n + 1 | n | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i1=0, i3=0); read C[i3 - 1, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (13/8)·n + 3 | n | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i1=0, i3=0); read C[i3 - 1, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (-7/8) | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 2 | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (9/8) | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (13/8)·n + 2 | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0, i3=0); read C[i3, i4] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (-7/8) | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i1=0, i7=0, i8=0); read B[i7 + 1, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/8)·n | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i1=0, i7=0, i8=0); read B[i7 + 1, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n - 1 | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i1=1, i4=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i1=1, i4=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (3/8)·n + (1/4) | n | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i1=0, i3=0, i4=0); read C[i3 - 1, i4] (i0=0, i1=1, i3=0, i4=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/4)·n | n | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i1=0, i3=0, i4=0); read C[i3 - 1, i4] (i0=0, i1=1, i3=0, i4=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (13/8)·n + 1 | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i1=0, i8=0); read B[i7 + 1, i8] (i0=0, i8=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/8)·n + (1/2) | n | 0.0714·n^-2 | write B[0, i2] (i0=0, i1=0); write B[0, i2] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/2)·n + 2 | n | 0.0714·n^-2 | write B[0, i2] (i0=0, i1=0); write B[0, i2] (i0=0, i1=1) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (3/8)·n + (-3/4) | n | 0.0714·n^-2 | write B[0, i2] (i0=0, i1=0, i2=0); write B[0, i2] (i0=0, i1=1, i2=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/4)·n | n | 0.0714·n^-2 | write B[0, i2] (i0=0, i1=0, i2=0); write B[0, i2] (i0=0, i1=1, i2=0) (+1) |
| n^2 | 0.612 | level | (3/8)·n^2 + 1 | n | 0.0714·n^-2 | read B[i7, i8] (i0=0, i1=0, i7=0, i8=0); read B[i7, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.536 | level | (3/8)·n^2 + (21/8)·n + 1 | (7/8)·n + (-7/8) | 0.0625·n^-2 | read A[i1] (i0=0, i2=0) |
| n^2 | 0.536 | level | (3/8)·n^2 + 1 | (7/8)·n | 0.0625·n^-2 | read A[i1] (i0=0, i2=0) |
| n^2 | 0.5 | level | (1/4)·n^2 - 1 | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i1=1, i4=0) (+1) |
| n^2 | 0.5 | level | (1/4)·n^2 + 2·n + (-13/4) | n | 0.0714·n^-2 | read C[i7, i8] (i0=0, i1=0, i7=0); read C[i7, i8] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (1/4)·n - 2 | n | 0.0714·n^-2 | read C[i7, i8] (i0=0, i1=0, i7=0); read C[i7, i8] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 1 | n | 0.0714·n^-2 | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/2)·n + (-11/4) | n | 0.0714·n^-2 | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.5 | level | (1/4)·n^2 | n | 0.0714·n^-2 | read B[i3, i4] (i0=0, i1=0); read B[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/2)·n + (-11/4) | n | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0, i1=1) (+1) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 1 | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (7/4)·n | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i1=0, i7=0, i8=0); read D[i7, i8 + 1] (i0=0, i7=0, i8=0) |
| n^2 | 0.5 | level | (1/4)·n^2 | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i1=0, i7=0, i8=0); read D[i7, i8 + 1] (i0=0, i7=0, i8=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0, i5=0, i6=0); read C[i5, i6] (i0=0, i5=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (13/8)·n + (-7/8) | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0, i6=0); read C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/8)·n | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0, i6=0); read C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + 2·n + (-5/4) | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i1=0, i7=0); read D[i7, i8 + 1] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (1/4)·n - 3 | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i1=0, i7=0); read D[i7, i8 + 1] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0, i5=0); read C[i5, i6] (i0=0, i5=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/2)·n + (-3/4) | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (7/4)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0); read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 | n | 0.0714·n^-2 | read B[i3, i4] (i0=0, i1=0, i4=0); read B[i3, i4] (i0=0, i1=1, i4=0) (+1) |
| n^2 | 0.5 | level | (1/4)·n^2 + (7/4)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0, i6=0); read C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i1=0, i6=0); read C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.0765 | level | (3/8)·n^2 + (11/4)·n + (7/8) | (1/8)·n + (-9/8) | 0.00893·n^-2 | read A[i1] (i0=0, i2=0) |
| n^2 | 0.0765 | level | (3/8)·n^2 + (1/8)·n | (1/8)·n - 2 | 0.00893·n^-2 | read A[i1] (i0=0, i2=0) |
| n^1 | 2 | level | 1 | 2·n - 1 | 0.143·n^-2 | read C[i5, i6 - 1] (i0=0, i1=0); read D[i7, i8] (i0=0, i1=0, i8=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (11/4)·n + (7/8) | 1 | 0.0714·n^-3 | read A[i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (1/8)·n | 1 | 0.0714·n^-3 | read A[i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.612 | level | (3/8)·n^2 + (1/8)·n | 1 | 0.0714·n^-3 | read A[i1] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 1 | 1 | 0.0714·n^-3 | read C[i3, i4] (i0=0, i1=1) |

Multi-field 2-D stencil (B, C, D fields): cross-sweep re-reads at (3/8)n^2 + O(n) lines — three field planes — with 0.077-coefficient families per field pair and a (1/4)n^2 → (3/8)n^2 ramp on `read D[i7,i8+1]`. Headroom +1.0; the boundary is 3× a single plane, so fdtd crosses the cache one third the size of a same-n single-field stencil.

## fdtd — single-shot  [`exact`]

Accesses $A(n) = 14·n^3 - 18·n^2 + 6·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.616·n^4  +  0.253·n^3.5  +  32.6·n^3  +  3.35·n^2.5  +  36.5·n^2

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0765 | level | (3/8)·n^2 + (5/2)·n + (1/8) | (1/8)·n^3 + (-19/8)·n^2 + (35/8)·n + (-17/8) | 0.00893 | read D[i5, i6] (i0=0, i5=0); read D[i5, i6] (i0=0) |
| n^4 | 0.0765 | level | (3/8)·n^2 + (-1/8)·n + 1 | (1/8)·n^3 + (-9/4)·n^2 + (33/8)·n - 2 | 0.00893 | read D[i5, i6] (i0=0, i5=0); read D[i5, i6] (i0=0) |
| n^4 | 0.0765 | level | (3/8)·n^2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read B[i7 + 1, i8] (i0=0) |
| n^4 | 0.0658 | ramp | (1/4)·n^2 + (1/4)·n + 1  →  (3/8)·n^2 - 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read D[i7, i8 + 1] (i0=0) |
| n^4 | 0.0658 | ramp | (1/4)·n^2 + (1/4)·n + 1  →  (3/8)·n^2 - 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read C[i7, i8] (i0=0) |
| n^4 | 0.065 | ramp | (1/4)·n^2 + 2  →  (3/8)·n^2 + (-1/2)·n - 1 | (1/8)·n^3 + (-21/8)·n^2 + (21/2)·n - 8 | 0.00893 | read C[i3, i4] (i0=0) |
| n^4 | 0.065 | ramp | (1/4)·n^2 + 2  →  (3/8)·n^2 + (-1/2)·n - 1 | (1/8)·n^3 + (-21/8)·n^2 + (21/2)·n - 8 | 0.00893 | read B[i3, i4] (i0=0) |
| n^4 | 0.0625 | level | (1/4)·n^2 + (13/8)·n + (1/8) | (1/8)·n^3 + (-3/2)·n^2 + (27/8)·n | 0.00893 | read C[i5, i6] (i0=0) |
| n^4 | 0.0625 | level | (1/4)·n^2 + (-1/8)·n + 1 | (1/8)·n^3 + (-19/8)·n^2 + 6·n | 0.00893 | read C[i5, i6] (i0=0) |
| n^3.5 | 0.0884 | level | (1/2)·n + (5/2) | (1/8)·n^3 + (-19/8)·n^2 + (17/4)·n | 0.00893 | read B[i7, i8] (i0=0) |
| n^3.5 | 0.0884 | level | (1/2)·n + 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read B[i7, i8] (i0=0) |
| n^3.5 | 0.0765 | level | (3/8)·n + 2 | (1/8)·n^3 + (-9/4)·n^2 + 4·n | 0.00893 | read C[i3 - 1, i4] (i0=0) |
| n^3 | 2.62 | level | 1 | (21/8)·n^3 + (-15/8)·n^2 | 0.188 | read D[i5, i6] (i0=0); read C[i5, i6 - 1] (i0=0) (+1) |
| n^3 | 1.95 | level | 3 | (9/8)·n^3 - 2·n^2 | 0.0804 | write B[0, i2] (i0=0); read B[i3, i4] (i0=0) (+4) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (-11/4)·n^2 + n | 0.125 | read B[i3, i4] (i0=0); read C[i7, i8] (i0=0) |
| n^3 | 1.75 | level | 4 | (7/8)·n^3 + (-7/8)·n^2 | 0.0625 | write C[i7, i8] (i0=0) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 + (-7/8)·n^2 | 0.0625 | read C[i3 - 1, i4] (i0=0) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 + (-7/8)·n^2 | 0.0625 | read C[i3, i4] (i0=0) |
| n^3 | 1.5 | level | 4 | (3/4)·n^3 + (-3/4)·n^2 | 0.0536 | read B[i7, i8] (i0=0) |
| n^3 | 1.5 | level | 4 | (3/4)·n^3 + (-3/4)·n^2 | 0.0536 | read B[i7 + 1, i8] (i0=0) |
| n^3 | 1.5 | level | 4 | (3/4)·n^3 + (-3/4)·n^2 | 0.0536 | read D[i7, i8 + 1] (i0=0) |
| n^3 | 1.24 | level | 2 | (7/8)·n^3 + n^2 - n | 0.0625 | write B[0, i2] (i0=0, i2=0); read A[i1] (i0=0) (+1) |
| n^3 | 1.1 | ramp | (1/4)·n^2 + (1/8)·n  →  (3/8)·n^2 + (-1/2)·n | 2·n^2 - 10·n + 8 | 0.143/n | read B[i3, i4] (i0=0, i4=0); read C[i3, i4] (i0=0, i4=0) |
| n^3 | 1.06 | level | 2 | (3/4)·n^3 | 0.0536 | read C[i5, i6] (i0=0, i5=0); read C[i5, i6] (i0=0) |
| n^3 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (1/8) | n^2 - 2·n + 1 | 0.0714/n | read D[i5, i6] (i0=0, i5=0, i6=0); read D[i5, i6] (i0=0, i6=0) |
| n^3 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | n^2 - 2·n + 1 | 0.0714/n | read D[i5, i6] (i0=0, i5=0, i6=0); read D[i5, i6] (i0=0, i6=0) |
| n^3 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (-7/8) | n^2 - 2·n + 1 | 0.0714/n | read D[i5, i6] (i0=0, i5=0); read D[i5, i6] (i0=0) |
| n^3 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | n^2 - 2·n + 1 | 0.0714/n | read D[i5, i6] (i0=0, i5=0); read D[i5, i6] (i0=0) |
| n^3 | 0.612 | level | (3/8)·n^2 | n^2 - n | 0.0714/n | read B[i7, i8] (i0=0, i7=0); read B[i7 + 1, i8] (i0=0) |
| n^3 | 0.612 | level | (3/8)·n^2 | n^2 - 2·n | 0.0714/n | read B[i7 + 1, i8] (i0=0, i8=0) |
| n^3 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (1/8) | n^2 - 2·n + 1 | 0.0714/n | read D[i5, i6] (i0=0, i5=0); read D[i5, i6] (i0=0) |
| n^3 | 0.612 | level | (3/8)·n^2 + (21/8)·n + 1 | n^2 - n | 0.0714/n | write B[0, i2] (i0=0); read B[i3, i4] (i0=0, i3=0) (+1) |
| n^3 | 0.559 | level | 5 | (1/4)·n^3 + (-9/4)·n^2 + 2·n | 0.0179 | read B[i7 + 1, i8] (i0=0); read B[i7, i8] (i0=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (3/8)·n - 1  →  (3/8)·n^2 - 1 | n^2 - 2·n | 0.0714/n | read C[i7, i8] (i0=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (3/8)·n - 2  →  (3/8)·n^2 - 2 | n^2 - 2·n | 0.0714/n | read D[i7, i8 + 1] (i0=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (1/4)·n  →  (3/8)·n^2 + (-1/8)·n | n^2 - 2·n | 0.0714/n | read D[i7, i8 + 1] (i0=0, i8=0) |
| n^3 | 0.555 | ramp | (1/4)·n^2 + (1/4)·n - 1  →  (3/8)·n^2 + (-1/8)·n - 1 | n^2 - 2·n | 0.0714/n | read C[i7, i8] (i0=0, i8=0) |
| n^3 | 0.548 | ramp | (1/4)·n^2 + 1  →  (3/8)·n^2 + (-5/8)·n + 1 | n^2 - 5·n + 4 | 0.0714/n | read C[i3, i4] (i0=0) |
| n^3 | 0.548 | ramp | (1/4)·n^2 + 1  →  (3/8)·n^2 + (-5/8)·n + 1 | n^2 - 5·n + 4 | 0.0714/n | read B[i3, i4] (i0=0) |
| n^3 | 0.5 | level | (1/4)·n^2 + (13/8)·n + (1/8) | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0, i6=0) |
| n^3 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0, i6=0) |
| n^3 | 0.5 | level | (1/4)·n^2 + (13/8)·n + (1/8) | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0) |
| n^3 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n^2 - 3·n | 0.0714/n | read C[i5, i6] (i0=0) |
| n^3 | 0.306 | level | 6 | (1/8)·n^3 + (-9/8)·n^2 + n | 0.00893 | read D[i7, i8 + 1] (i0=0) |
| n^3 | 0.28 | level | 5 | (1/8)·n^3 + (-9/8)·n^2 + n | 0.00893 | read B[i3, i4] (i0=0, i3=0, i4=0); read C[i3, i4] (i0=0, i3=0, i4=0) (+3) |
| n^3 | 0.28 | level | 5 | (1/8)·n^3 + (-9/8)·n^2 + n | 0.00893 | read D[i7, i8] (i0=0) |
| n^3 | 0.25 | level | 4 | (1/8)·n^3 - n^2 | 0.00893 | read C[i5, i6 - 1] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 - n^2 | 0.00893 | read C[i5, i6] (i0=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (1/8)·n + (-1/2) | (1/8)·n^2 + (-9/4)·n + (17/8) | 0.00893/n | read C[i3, i4] (i0=0, i3=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (-1/2)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read C[i3, i4] (i0=0, i3=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (1/8)·n + (-1/2) | (1/8)·n^2 + (-9/4)·n + (17/8) | 0.00893/n | read B[i3, i4] (i0=0, i3=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (-1/2)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read B[i3, i4] (i0=0, i3=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + 1 | (1/8)·n^2 - 2·n | 0.00893/n | read B[i7, i8] (i0=0, i7=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (13/8)·n + 1 | (1/8)·n^2 + (-9/8)·n | 0.00893/n | read B[i7 + 1, i8] (i0=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (21/8)·n + 1 | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.00893/n | read D[i5, i6] (i0=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read D[i5, i6] (i0=0) |
| n^3 | 0.0726 | ramp | (3/8)·n^2 + (-1/8)·n + 1  →  (3/8)·n^2 - 2 | (1/8)·n^2 - 2·n | 0.00893/n | read B[i7 + 1, i8] (i0=0, i7=0) |
| n^3 | 0.0724 | ramp | (3/8)·n^2 + (-1/8)·n + 2  →  (3/8)·n^2 - 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read C[i3, i4] (i0=0) |
| n^3 | 0.0723 | ramp | (3/8)·n^2 + (-3/8)·n + 2  →  (3/8)·n^2 + (-1/4)·n - 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read C[i3 - 1, i4] (i0=0, i3=0) |
| n^3 | 0.0723 | ramp | (3/8)·n^2 + (-1/2)·n + 4  →  (3/8)·n^2 + (-1/4)·n - 2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | write B[0, i2] (i0=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 + (-1/8)·n + 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i5, i6] (i0=0, i5=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 + (7/4)·n + 1 | (1/8)·n^2 + (-9/8)·n | 0.00893/n | read C[i5, i6] (i0=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 + 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i5, i6] (i0=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/8)·n^2 + (-9/4)·n + (17/8) | 0.00893/n | read B[i3, i4] (i0=0) |
| n^3 | 0.0625 | level | (1/4)·n^2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read B[i3, i4] (i0=0) |
| n^3 | 0.0593 | ramp | (1/4)·n^2 + 2  →  (1/4)·n^2 + (1/4)·n - 4 | (1/8)·n^2 - 2·n | 0.00893/n | read D[i7, i8 + 1] (i0=0, i7=0) |
| n^3 | 0.0593 | ramp | (1/4)·n^2 + 2  →  (1/4)·n^2 + (1/4)·n - 4 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i7, i8] (i0=0, i7=0) |
| n^3 | 0.0592 | ramp | (1/4)·n^2 + (-1/4)·n + 2  →  (1/4)·n^2 + (-1/8)·n - 1 | (1/8)·n^2 - 2·n | 0.00893/n | read C[i5, i6] (i0=0) |
| n^3 | 0.0591 | ramp | (1/4)·n^2 + (-1/4)·n + 3  →  (1/4)·n^2 - 3 | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read C[i3, i4] (i0=0) |
| n^3 | 0.0591 | ramp | (1/4)·n^2 + (-1/4)·n + 3  →  (1/4)·n^2 - 3 | (1/8)·n^2 + (-17/8)·n + 2 | 0.00893/n | read B[i3, i4] (i0=0) |
| n^2.5 | 0.707 | level | (1/2)·n + 2 | n^2 - 2·n | 0.0714/n | read B[i7, i8] (i0=0) |
| n^2.5 | 0.707 | level | (1/2)·n + (5/2) | n^2 - 2·n | 0.0714/n | read B[i7, i8] (i0=0, i8=0) |
| n^2.5 | 0.707 | level | (1/2)·n + 2 | n^2 - 2·n | 0.0714/n | read B[i7, i8] (i0=0, i8=0) |
| n^2.5 | 0.612 | level | (3/8)·n + 2 | n^2 - 2·n | 0.0714/n | read C[i3 - 1, i4] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n + 2 | n^2 - 2·n | 0.0714/n | read C[i3 - 1, i4] (i0=0, i4=0) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.0625/n | write B[0, i2] (i0=0) |
| n^2 | 1.22 | level | (3/8)·n^2 + (1/8)·n + (-1/2) | 2·n - 2 | 0.143·n^-2 | read B[i3, i4] (i0=0, i3=0, i4=0); read C[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 1.22 | level | (3/8)·n^2 + (-1/2)·n | 2·n - 2 | 0.143·n^-2 | read B[i3, i4] (i0=0, i3=0, i4=0); read C[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 4 | 2·n - 2 | 0.143·n^-2 | read B[i3, i4] (i0=0, i4=0); read C[i3, i4] (i0=0, i4=0) |
| n^2 | 1 | level | (1/4)·n^2 - 1 | 2·n - 1 | 0.143·n^-2 | read B[i3, i4] (i0=0, i4=0); read C[i7, i8] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 1 | 2·n - 1 | 0.143·n^-2 | read B[i3, i4] (i0=0); read C[i7, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/8)·n + (-3/2) | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/2)·n | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/8)·n + (-3/2) | n - 1 | 0.0714·n^-2 | read B[i3, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/2)·n | n - 1 | 0.0714·n^-2 | read B[i3, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n + 1 | n - 1 | 0.0714·n^-2 | read D[i5, i6] (i0=0, i6=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + 1 | n - 1 | 0.0714·n^-2 | read D[i5, i6] (i0=0, i6=0) |
| n^2 | 0.612 | level | (3/8)·n^2 - 1 | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i7=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/4)·n + (3/8) | n - 1 | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-3/8)·n + 1 | n - 1 | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (13/8)·n + 3 | n - 1 | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (-7/8) | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 2 | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (9/8) | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (13/8)·n + 2 | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0, i3=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (5/2)·n + (-7/8) | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/8)·n | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n - 1 | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0, i4=0) |
| n^2 | 0.612 | level | (3/8)·n^2 | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0, i4=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (3/8)·n + (1/4) | n - 1 | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/4)·n | n - 1 | 0.0714·n^-2 | read C[i3 - 1, i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (21/8)·n + 1 | n - 1 | 0.0714·n^-2 | read D[i5, i6] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + 1 | n - 1 | 0.0714·n^-2 | read D[i5, i6] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (13/8)·n + 1 | n | 0.0714·n^-2 | read B[i7 + 1, i8] (i0=0, i8=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (1/8)·n + (1/2) | n - 1 | 0.0714·n^-2 | write B[0, i2] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/2)·n + 2 | n - 1 | 0.0714·n^-2 | write B[0, i2] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (3/8)·n + (-3/4) | n - 1 | 0.0714·n^-2 | write B[0, i2] (i0=0, i2=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/4)·n | n - 1 | 0.0714·n^-2 | write B[0, i2] (i0=0, i2=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + 1 | n | 0.0714·n^-2 | read B[i7, i8] (i0=0, i7=0, i8=0) |
| n^2 | 0.536 | level | (3/8)·n^2 + 1 | (7/8)·n | 0.0625·n^-2 | read A[i1] (i0=0, i2=0) |
| n^2 | 0.5 | level | (1/4)·n^2 - 1 | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0, i4=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/2)·n + (-11/4) | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 1 | n - 1 | 0.0714·n^-2 | read C[i3, i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 1 | n - 1 | 0.0714·n^-2 | read B[i3, i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/2)·n + (-11/4) | n - 1 | 0.0714·n^-2 | read B[i3, i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (7/4)·n | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i7=0, i8=0) |
| n^2 | 0.5 | level | (1/4)·n^2 | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i7=0, i8=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i5=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (13/8)·n + (-7/8) | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/8)·n | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + 2·n + (-5/4) | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (1/4)·n - 3 | n | 0.0714·n^-2 | read D[i7, i8 + 1] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/8)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i5=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/2)·n + (-3/4) | n | 0.0714·n^-2 | read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (-1/4)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (7/4)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + 2·n + (-13/4) | n | 0.0714·n^-2 | read C[i7, i8] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (1/4)·n - 2 | n | 0.0714·n^-2 | read C[i7, i8] (i0=0, i7=0) |
| n^2 | 0.5 | level | (1/4)·n^2 | n - 1 | 0.0714·n^-2 | read B[i3, i4] (i0=0, i4=0) |
| n^2 | 0.5 | level | (1/4)·n^2 | n - 1 | 0.0714·n^-2 | read B[i3, i4] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (7/4)·n + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i6=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + 1 | n | 0.0714·n^-2 | read C[i5, i6] (i0=0, i6=0) |

Multi-field 2-D stencil (B, C, D fields): cross-sweep re-reads at (3/8)n^2 + O(n) lines — three field planes — with 0.077-coefficient families per field pair and a (1/4)n^2 → (3/8)n^2 ramp on `read D[i7,i8+1]`. Headroom +1.0; the boundary is 3× a single plane, so fdtd crosses the cache one third the size of a same-n single-field stencil.

## floyd_warshall — infinite-repeat  [`exact`]

Accesses $A(n) = 4·n^3$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^4  +  0.0625·n^3.5  +  7.39·n^3  +  1.91·n^2.5  +  82·n^2  +  11·n^1.5  +  90.8·n^1  +  48.2·n^0.5  +  89.8·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0221 | level | (1/8)·n^2 | (1/16)·n^3 + (-11/4)·n^2 + (129/4)·n - 54 | 0.0156 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^4 | 0.0221 | level | (1/8)·n^2 | (1/16)·n^3 + (-13/8)·n^2 + (23/2)·n - 26 | 0.0156 | read A[i1, i2] (i0=0, i1=0, i2=8); read A[i1, i2] (i0=0, i1=0) (+4) |
| n^3.5 | 0.0417 | level | (1/4)·n + 2 | (1/12)·n^3 + (-91/32)·n^2 + (847/24)·n - 192 | 0.0208 | read A[i0, i2] (i0=0); read A[i0, i2] |
| n^3.5 | 0.0182 | level | (1/4)·n + 2 | (7/192)·n^3 + (-57/64)·n^2 + (61/24)·n + 38 | 0.00911 | read A[i0, i2] |
| n^3.5 | 0.0026 | level | (1/4)·n + 2 | (1/192)·n^3 + (-17/64)·n^2 + (61/24)·n + 20 | 0.0013 | read A[i0, i2] |
| n^3 | 1.77 | level | 3 | (49/48)·n^3 + (-1983/128)·n^2 + (2861/48)·n - 8 | 0.255 | read A[i0, i2]; write A[i1, i2] |
| n^3 | 1.01 | level | 3 | (7/12)·n^3 + (-287/32)·n^2 + (259/6)·n - 70 | 0.146 | read A[i0, i2] (i0=0); write A[i1, i2] (i0=0) (+1) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 + (7/16)·n^2 - 7 | 0.219 | write A[i1, i2] (i0=0, i1=0); read A[i1, i2] (+1) |
| n^3 | 0.785 | level | 3 | (29/64)·n^3 + (-509/64)·n^2 + (83/2)·n - 26 | 0.113 | read A[i0, i2] (i0=0, i1=1, i2=0); read A[i0, i2] (i0=0, i2=0) (+9) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (-115/16)·n^2 + 31·n - 6 | 0.109 | read A[i1, i0] (i0=0, i1=0); write A[i1, i2] (i0=0) (+3) |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 9·n + 17 | 0.25/n | read A[i1, i2] (i0=0, i1=0, i2=0); read A[i1, i2] (i0=0, i2=0) (+4) |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 11·n + 27 | 0.25/n | read A[i1, i2] (i0=0, i1=0, i2=8); read A[i1, i2] (i0=0, i1=0) (+3) |
| n^3 | 0.309 | level | (1/8)·n^2 | (7/8)·n^2 + (-67/4)·n + 30 | 0.219/n | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^3 | 0.309 | level | (1/8)·n^2 + (-1/8)·n + 3 | (7/8)·n^2 + (-63/4)·n + 28 | 0.219/n | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^3 | 0.253 | level | 3 | (7/48)·n^3 + (-469/128)·n^2 + (1085/48)·n - 21 | 0.0365 | read A[i0, i2]; write A[i1, i2] |
| n^3 | 0.125 | level | 4 | (1/16)·n^3 + (-25/16)·n^2 + (19/2)·n - 8 | 0.0156 | read A[i0, i2] (i0=0, i1=1, i2=8); read A[i1, i0] (i0=0) (+2) |
| n^3 | 0.125 | level | 4 | (1/16)·n^3 + (-27/16)·n^2 + (93/8)·n - 10 | 0.0156 | read A[i1, i0] |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-175/128)·n^2 + (133/16)·n - 7 | 0.0137 | write A[i1, i2] |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-159/128)·n^2 + (115/16)·n - 6 | 0.0137 | write A[i1, i2] |
| n^3 | 0.0442 | level | (1/8)·n^2 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0312/n | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^3 | 0.0405 | ramp | (1/8)·n^2 + (-1/8)·n + 4  →  (1/8)·n^2 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0312/n | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^3 | 0.0394 | ramp | (1/8)·n^2 + (-1/8)·n + 4  →  (1/8)·n^2 - 1 | (1/8)·n^2 + (-17/4)·n + 8 | 0.0312/n | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^3 | 0.0164 | ramp | (19/8)·n - 1  →  (1/8)·n^2 - 1 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i1, i2] |
| n^3 | 0.0157 | ramp | (13/4)·n - 1  →  (1/8)·n^2 - 1 | (1/16)·n^2 + (-21/8)·n + 27 | 0.0156/n | read A[i0, i2] (i1=0) |
| n^3 | 0.0107 | ramp | (3/8)·n + 3  →  (1/8)·n^2 - 2·n + 3 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i1, i2] |
| n^3 | 0.0107 | ramp | (3/8)·n + 3  →  (1/8)·n^2 - 2·n + 3 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i0, i2] (i1=0) |
| n^2.5 | 0.5 | level | (1/4)·n + 2 | n^2 - 10·n + 23 | 0.25/n | read A[i0, i2] (i0=0, i2=8); read A[i0, i2] (i2=0) |
| n^2.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n^2 + (-133/8)·n + 42 | 0.219/n | read A[i0, i2] |
| n^2.5 | 0.25 | level | (1/4)·n + 1 | (1/2)·n^2 + (-31/2)·n + 134 | 0.125/n | read A[i0, i2] (i0=0); read A[i0, i2] |
| n^2.5 | 0.219 | level | (1/4)·n + 1 | (7/16)·n^2 + (-7/4)·n - 34 | 0.109/n | read A[i0, i2] |
| n^2.5 | 0.156 | level | (1/4)·n + 2 | (5/16)·n^2 + (-85/8)·n + 90 | 0.0781/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 1 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 1 | (1/16)·n^2 + (25/4)·n - 24 | 0.0156/n | read A[i0, i2] (i1=0, i2=0); read A[i0, i2] (i2=0) (+1) |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (-5/8)·n - 6 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (1/8)·n - 18 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (-3/4)·n - 18 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 1 | (1/16)·n^2 + (-3/4)·n - 18 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0261 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i1, i2] |
| n^2.5 | 0.0238 | ramp | (1/8)·n + 3  →  (1/4)·n | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i1, i2] (i2=8); read A[i1, i2] |
| n^2.5 | 0.0238 | ramp | (1/8)·n + 3  →  (1/4)·n | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0219 | ramp | (1/8)·n + 3  →  (1/4)·n - 1 | (7/128)·n^2 + (-35/16)·n + 21 | 0.0137/n | read A[i0, i2] |
| n^2.5 | 0.00296 | ramp | (1/8)·n + 4  →  (1/4)·n - 1 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00195/n | read A[i0, i2] |
| n^2 | 6.06 | level | 3 | (7/2)·n^2 + (-63/2)·n + 28 | 0.875/n | read A[i0, i2] |
| n^2 | 6.06 | level | 3 | (7/2)·n^2 + (-63/2)·n + 28 | 0.875/n | read A[i1, i0] |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 + (-189/8)·n + 21 | 0.656/n | read A[i1, i0] |
| n^2 | 3.71 | level | 2 | (21/8)·n^2 + (-21/8)·n | 0.656/n | read A[i0, i2] |
| n^2 | 3.71 | level | 2 | (21/8)·n^2 + (-21/8)·n | 0.656/n | write A[i1, i2] |
| n^2 | 3.5 | level | 1 | (7/2)·n^2 + (-21/2)·n + 7 | 0.875/n | read A[i1, i0] |
| n^2 | 3.25 | level | 3 | (15/8)·n^2 + (-135/8)·n + 15 | 0.469/n | write A[i1, i2] |
| n^2 | 3.25 | level | 3 | (15/8)·n^2 + (-135/8)·n + 15 | 0.469/n | write A[i1, i2] |
| n^2 | 3.09 | level | 2 | (35/16)·n^2 - 14·n + 49 | 0.547/n | read A[i0, i2]; write A[i1, i2] |
| n^2 | 2.83 | level | (1/8)·n^2 + (-1/8)·n + 2 | 8·n - 16 | 2·n^-2 | read A[i1, i2] (i0=1, i2=0); read A[i1, i2] (i1=0, i2=0) (+1) |
| n^2 | 2.81 | level | 3 | (13/8)·n^2 + (-117/8)·n + 13 | 0.406/n | write A[i1, i2] |
| n^2 | 2.65 | level | 2 | (15/8)·n^2 + (-15/8)·n | 0.469/n | write A[i1, i2] |
| n^2 | 2.62 | level | 1 | (21/8)·n^2 + (-69/8)·n + 6 | 0.656/n | read A[i1, i0] |
| n^2 | 2.39 | level | 2 | (27/16)·n^2 + (-21/2)·n + 11 | 0.422/n | read A[i0, i2]; write A[i1, i2] |
| n^2 | 2.12 | level | (1/8)·n^2 + (-1/8)·n + 2 | 6·n - 12 | 1.5·n^-2 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^2 | 1.88 | level | 1 | (15/8)·n^2 + (-23/4)·n + 4 | 0.469/n | read A[i1, i0] |
| n^2 | 1.77 | level | 2 | (5/4)·n^2 + (-45/8)·n | 0.312/n | read A[i0, i2] |
| n^2 | 1.62 | level | 3 | (15/16)·n^2 + (-95/8)·n + 35 | 0.234/n | read A[i0, i2] |
| n^2 | 1.62 | level | 3 | (15/16)·n^2 - 5·n - 20 | 0.234/n | read A[i0, i2] |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-31/4)·n + 5 | 0.219/n | read A[i1, i0] |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-63/8)·n + 7 | 0.219/n | write A[i1, i2] |
| n^2 | 1.33 | level | 2 | (15/16)·n^2 + (5/4)·n | 0.234/n | read A[i0, i2] |
| n^2 | 1.3 | level | 3 | (3/4)·n^2 + (-27/4)·n + 6 | 0.188/n | write A[i1, i2] |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 - 14·n + 56 | 0.219/n | read A[i1, i0] (i0=0, i1=0); read A[i0, i2] (i0=0, i1=0) (+1) |
| n^2 | 1.08 | level | 3 | (5/8)·n^2 + (-45/8)·n + 5 | 0.156/n | write A[i1, i2] |
| n^2 | 1.06 | level | 2 | (3/4)·n^2 - 6·n | 0.188/n | read A[i1, i0] |
| n^2 | 1.06 | level | 2 | (3/4)·n^2 + (-3/4)·n | 0.188/n | write A[i1, i2] |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 - 7·n + 28 | 0.109/n | read A[i1, i0] |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-7/8)·n - 21 | 0.109/n | read A[i1, i0] |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-45/8)·n + 21 | 0.0938/n | read A[i0, i2] |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-9/8)·n - 15 | 0.0938/n | read A[i0, i2] |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-7/8)·n | 0.109/n | read A[i0, i2] |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-7/8)·n | 0.109/n | read A[i0, i2] |
| n^2 | 0.53 | level | 2 | (3/8)·n^2 + (-3/8)·n | 0.0938/n | read A[i0, i2] |
| n^2 | 0.383 | level | 1 | (49/128)·n^2 + (-79/16)·n + 15 | 0.0957/n | write A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 3 | 0.25·n^-2 | read A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 2 | 0.25·n^-2 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 9 | 0.25·n^-2 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 9 | 0.25·n^-2 | read A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | n - 3 | 0.25·n^-2 | read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 2 | n - 1 | 0.25·n^-2 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (17/8) | n - 2 | 0.25·n^-2 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 2 | n - 10 | 0.25·n^-2 | read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | 2 | (1/4)·n^2 + (-33/8)·n + 17 | 0.0625/n | read A[i0, i2] (i2=0); read A[i0, i2] (+1) |
| n^2 | 0.265 | level | 2 | (3/16)·n^2 + (-7/4)·n + 2 | 0.0469/n | read A[i0, i2]; write A[i1, i2] |
| n^2 | 0.25 | level | 4 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0312/n | read A[i1, i0] |
| n^2 | 0.233 | ramp | (9/4)·n  →  (1/8)·n^2 | n - 17 | 0.25·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.227 | ramp | (3/8)·n + 2  →  (1/8)·n^2 - n + 2 | n - 10 | 0.25·n^-2 | read A[i0, i2] (i1=0, i2=0) |
| n^2 | 0.227 | ramp | (3/8)·n + 1  →  (1/8)·n^2 - n + 1 | n - 10 | 0.25·n^-2 | read A[i1, i2] |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0312/n | read A[i1, i2] (i1=0); read A[i1, i2] (i2=8) (+1) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0312/n | write A[i1, i2] (i2=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0312/n | write A[i1, i2] |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0312/n | write A[i1, i2] |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0312/n | write A[i1, i2] |
| n^2 | 0.207 | ramp | (1/2)·n  →  (1/8)·n^2 | (7/8)·n - 2 | 0.219·n^-2 | read A[i1, i2] (i2=0) |
| n^2 | 0.199 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-9/8)·n + 5 | (7/8)·n - 14 | 0.219·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.199 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-9/8)·n + 5 | (7/8)·n - 14 | 0.219·n^-2 | read A[i1, i0] (i2=0) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 + (-1/8)·n | 0.0312/n | write A[i1, i2] |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 - n | 0.0312/n | read A[i1, i0] |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 + (-1/8)·n | 0.0312/n | write A[i1, i2] |
| n^2 | 0.146 | ramp | (5/2)·n - 1  →  (1/8)·n^2 + (-1/8)·n + 2 | (5/8)·n - 10 | 0.156·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-1/2)·n | 0.0156/n | read A[i0, i2] |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-3/2)·n + 8 | 0.0156/n | read A[i1, i0] |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-1/2)·n - 8 | 0.0156/n | read A[i1, i0] |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0156/n | read A[i0, i2] |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (1/2)·n - 1 | 0.0156/n | write A[i1, i2] (i0=0, i1=0); write A[i1, i2] |
| n^2 | 0.0547 | level | 1 | (7/128)·n^2 + (5/16)·n - 6 | 0.0137/n | write A[i1, i2] |
| n^2 | 0.0547 | level | 1 | (7/128)·n^2 + (-21/16)·n + 7 | 0.0137/n | write A[i1, i2] |
| n^2 | 0.029 | ramp | (19/8)·n - 1  →  (1/8)·n^2 + (-3/4)·n + 2 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0289 | ramp | (9/4)·n  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 2 | 0.0312·n^-2 | read A[i1, i2] (i2=0) |
| n^2 | 0.0289 | ramp | (9/4)·n - 1  →  (1/8)·n^2 + (-7/8)·n + 2 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i1, i2] |
| n^2 | 0.0289 | ramp | (9/4)·n - 1  →  (1/8)·n^2 + (-7/8)·n + 2 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0285 | ramp | (25/8)·n - 1  →  (1/8)·n^2 - n + 3 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0279 | ramp | (5/4)·n  →  (1/8)·n^2 - 2·n + 6 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0277 | ramp | (9/4)·n - 2  →  (1/8)·n^2 - 2·n + 6 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i1, i0] (i2=0) |
| n^2 | 0.00781 | level | 1 | (1/128)·n^2 + (127/16)·n + 30 | 0.00195/n | read A[i1, i2] (i0=0, i1=0, i2=0); read A[i1, i0] (i0=0, i1=0, i2=0) (+9) |
| n^1.5 | 3.78 | ramp | 5  →  (1/4)·n + 1 | (63/8)·n - 34 | 1.97·n^-2 | read A[i0, i2] |
| n^1.5 | 2.5 | level | (1/4)·n + 1 | 5·n - 65 | 1.25·n^-2 | read A[i0, i2] |
| n^1.5 | 0.53 | ramp | 5  →  (1/4)·n + 1 | (9/8)·n - 11 | 0.281·n^-2 | read A[i0, i2] |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 10 | 0.25·n^-2 | read A[i0, i2] |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 16 | 0.25·n^-2 | read A[i0, i2] |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 2 | 0.25·n^-2 | read A[i0, i2] (i0=0, i2=0) |
| n^1.5 | 0.5 | level | (1/4)·n | n | 0.25·n^-2 | read A[i1, i2] (i0=1, i1=0, i2=0); read A[i1, i2] (i2=0) |
| n^1.5 | 0.438 | level | (1/4)·n | (7/8)·n - 14 | 0.219·n^-2 | read A[i0, i2] |
| n^1.5 | 0.354 | level | (1/8)·n + 1 | n - 10 | 0.25·n^-2 | read A[i1, i2] |
| n^1.5 | 0.354 | level | (1/8)·n + 2 | n - 8 | 0.25·n^-2 | read A[i0, i2] (i0=0, i1=1, i2=8); read A[i0, i2] (i2=0) |
| n^1.5 | 0.288 | ramp | 5  →  (1/4)·n - 1 | (7/8)·n - 14 | 0.219·n^-2 | read A[i1, i0] (i2=0) |
| n^1.5 | 0.257 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (5/8)·n - 10 | 0.156·n^-2 | read A[i0, i2] |
| n^1.5 | 0.125 | level | (1/4)·n + 1 | (1/4)·n - 4 | 0.0625·n^-2 | read A[i1, i2] (i0=1, i1=0); read A[i0, i2] (i1=0) |
| n^1.5 | 0.0625 | level | (1/4)·n | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0514 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0514 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0502 | ramp | (1/8)·n + 3  →  (1/4)·n - 1 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0502 | ramp | (1/8)·n + 3  →  (1/4)·n - 1 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] (i0=0, i1=1) |
| n^1.5 | 0.0502 | ramp | (1/8)·n + 3  →  (1/4)·n - 1 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i1, i2] |
| n^1.5 | 0.0412 | ramp | 5  →  (1/4)·n - 1 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i1, i0] (i2=0) |
| n^1 | 22.6 | level | 2 | 16·n - 13 | 4·n^-2 | read A[i1, i0] (i0=0, i1=0, i2=8); read A[i0, i2] (i0=0, i1=1, i2=0) (+10) |
| n^1 | 7 | level | 1 | 7·n - 14 | 1.75·n^-2 | read A[i1, i0] (i2=0) |
| n^1 | 4.95 | level | 2 | (7/2)·n - 28 | 0.875·n^-2 | read A[i1, i0] |
| n^1 | 4.5 | level | 1 | (9/2)·n - 15 | 1.12·n^-2 | write A[i1, i2] |
| n^1 | 3.71 | level | 2 | (21/8)·n - 21 | 0.656·n^-2 | read A[i1, i0] |
| n^1 | 3.71 | level | 2 | (21/8)·n - 21 | 0.656·n^-2 | read A[i0, i2] |
| n^1 | 3.71 | level | 2 | (21/8)·n - 21 | 0.656·n^-2 | read A[i0, i2] |
| n^1 | 3.5 | level | 1 | (7/2)·n - 7 | 0.875·n^-2 | read A[i1, i0] |
| n^1 | 2.83 | level | (1/8)·n^2 + (-1/8)·n + 2 | 8 | 2·n^-3 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^1 | 2.62 | level | 1 | (21/8)·n - 6 | 0.656·n^-2 | read A[i1, i0] |
| n^1 | 2.62 | level | 1 | (21/8)·n | 0.656·n^-2 | read A[i0, i2] |
| n^1 | 2.62 | level | 1 | (21/8)·n | 0.656·n^-2 | read A[i0, i2] |
| n^1 | 2.12 | level | (1/8)·n^2 | 6 | 1.5·n^-3 | read A[i1, i2] |
| n^1 | 1.88 | level | 1 | (15/8)·n - 2 | 0.469·n^-2 | read A[i1, i0] |
| n^1 | 1.88 | level | 1 | (15/8)·n - 2 | 0.469·n^-2 | read A[i1, i0] |
| n^1 | 1.75 | level | 1 | (7/4)·n | 0.438·n^-2 | read A[i0, i2]; write A[i1, i2] |
| n^1 | 1.59 | level | 2 | (9/8)·n - 17 | 0.281·n^-2 | read A[i0, i2] (i0=0, i1=0); read A[i1, i2] |
| n^1 | 1.52 | level | 3 | (7/8)·n - 14 | 0.219·n^-2 | read A[i1, i0] |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 0.219·n^-2 | read A[i1, i0] |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 0.219·n^-2 | read A[i1, i0] |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 0.219·n^-2 | read A[i0, i2] |
| n^1 | 1.06 | level | 2 | (3/4)·n - 6 | 0.188·n^-2 | read A[i0, i2] |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.219·n^-2 | read A[i0, i2] |
| n^1 | 0.75 | level | 1 | (3/4)·n - 6 | 0.188·n^-2 | read A[i0, i2] |
| n^1 | 0.707 | level | (1/8)·n^2 | 2 | 0.5·n^-3 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.25·n^-3 | read A[i1, i2] (i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.25·n^-3 | read A[i1, i2] |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.25·n^-3 | read A[i0, i2] (i0=1, i1=0, i2=8) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.25·n^-3 | read A[i0, i2] (i0=1, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/8)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/8)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n - 2 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 4 | 1 | 0.25·n^-3 | read A[i0, i2] (i0=8, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/8)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/8)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 2 | 1 | 0.25·n^-3 | read A[i0, i2] (i0=1, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.25·n^-3 | read A[i1, i0] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.25 | level | 1 | (1/4)·n - 2 | 0.0625·n^-2 | read A[i0, i2] |
| n^1 | 0.177 | level | 2 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] |
| n^1 | 0.177 | level | 2 | (1/8)·n - 1 | 0.0312·n^-2 | read A[i0, i2] |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0312·n^-2 | write A[i1, i2] |
| n^0.5 | 7 | level | (1/4)·n + 1 | 14 | 3.5·n^-3 | read A[i0, i2] |
| n^0.5 | 6.5 | level | (1/4)·n + 1 | 13 | 3.25·n^-3 | read A[i0, i2] |
| n^0.5 | 3.5 | level | (1/4)·n | 7 | 1.75·n^-3 | read A[i0, i2] (i1=0); read A[i0, i2] |
| n^0.5 | 3.5 | level | (1/4)·n + 1 | 7 | 1.75·n^-3 | read A[i0, i2] |
| n^0.5 | 3 | level | (1/4)·n + 1 | 6 | 1.5·n^-3 | read A[i0, i2] |
| n^0.5 | 2.5 | level | (1/4)·n | 5 | 1.25·n^-3 | read A[i0, i2] |
| n^0.5 | 1.46 | level | (17/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.37 | level | (15/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.32 | level | (7/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.27 | level | (13/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.22 | level | (3/2)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.25·n^-3 | read A[i1, i0] (i0=8, i1=7, i2=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.25·n^-3 | read A[i1, i2] (i0=8, i1=7, i2=0) |
| n^0.5 | 1.06 | level | (9/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1 | level | (1/4)·n | 2 | 0.5·n^-3 | read A[i0, i2] (i0=0, i1=1); read A[i1, i2] (i0=1, i1=0) |
| n^0.5 | 1 | level | n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.935 | level | (7/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.791 | level | (5/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.707 | level | (1/2)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0); read A[i0, i2] |
| n^0.5 | 0.5 | level | (1/4)·n + (11/4) | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] |
| n^0.5 | 0.354 | level | (1/8)·n + 2 | 1 | 0.25·n^-3 | read A[i1, i2] |
| n^0.5 | 0.354 | level | (1/8)·n + 1 | 1 | 0.25·n^-3 | read A[i1, i2] (i0=8, i1=8) |
| n^0 | 48.5 | level | 3 | 28 | 7·n^-3 | read A[i1, i2] |
| n^0 | 12.1 | level | 3 | 7 | 1.75·n^-3 | read A[i1, i2] |
| n^0 | 10.4 | level | 3 | 6 | 1.5·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 8.49 | level | 2 | 6 | 1.5·n^-3 | read A[i1, i2] |
| n^0 | 2.24 | level | 5 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.25·n^-3 | read A[i1, i2] |
| n^0 | 1.41 | level | 2 | 1 | 0.25·n^-3 | read A[i1, i2] |

Each k-sweep re-reads the whole distance matrix: two n^4 level terms of `read A[i1,i2]`/`read A[i0,i2]` at exactly (1/8)n^2 lines (no row additive term — the k-row is already resident), combined coefficient 0.0442, identical to gemm's constant. The n^3.5 terms are the k-row reuses at (1/4)n + 2 lines.

## floyd_warshall — single-shot  [`exact`]

Accesses $A(n) = 4·n^3$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^4  +  0.0625·n^3.5  +  7.39·n^3  +  1.91·n^2.5  +  81.6·n^2  +  10.2·n^1.5  +  66.2·n^1  +  47.8·n^0.5  +  88.3·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0221 | level | (1/8)·n^2 | (1/16)·n^3 + (-11/4)·n^2 + (129/4)·n - 54 | 0.0156 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^4 | 0.0221 | level | (1/8)·n^2 | (1/16)·n^3 + (-7/4)·n^2 + (27/2)·n - 26 | 0.0156 | read A[i0, i2] (i0=1, i1=0); read A[i1, i2] (i1=0) (+1) |
| n^3.5 | 0.0417 | level | (1/4)·n + 2 | (1/12)·n^3 + (-95/32)·n^2 + (925/24)·n - 198 | 0.0208 | read A[i0, i2] |
| n^3.5 | 0.0182 | level | (1/4)·n + 2 | (7/192)·n^3 + (-57/64)·n^2 + (61/24)·n + 38 | 0.00911 | read A[i0, i2] |
| n^3.5 | 0.0026 | level | (1/4)·n + 2 | (1/192)·n^3 + (-9/64)·n^2 + (-17/24)·n + 26 | 0.0013 | read A[i0, i2] |
| n^3 | 1.62 | level | 3 | (15/16)·n^3 + (-263/16)·n^2 + (161/2)·n - 57 | 0.234 | read A[i0, i2] (i0=0, i1=1, i2=0); read A[i0, i2] (i0=0, i2=0) (+8) |
| n^3 | 1.01 | level | 3 | (7/12)·n^3 + (-343/32)·n^2 + (707/12)·n - 84 | 0.146 | read A[i0, i2] |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 | 0.219 | read A[i1, i2] |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (-117/16)·n^2 + (257/8)·n | 0.109 | read A[i1, i2] (i1=0); read A[i1, i2] (+1) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (-105/16)·n^2 + (217/8)·n - 21 | 0.109 | write A[i1, i2] |
| n^3 | 0.442 | level | 3 | (49/192)·n^3 + (-469/128)·n^2 + (287/48)·n + 56 | 0.0638 | read A[i0, i2] |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 11·n + 26 | 0.25/n | read A[i1, i0] (i0=8, i1=0, i2=0); read A[i1, i0] (i0=8, i2=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 12·n + 27 | 0.25/n | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^3 | 0.309 | level | (1/8)·n^2 | (7/8)·n^2 + (-67/4)·n + 30 | 0.219/n | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^3 | 0.309 | level | (1/8)·n^2 + (-1/8)·n + 3 | (7/8)·n^2 + (-63/4)·n + 28 | 0.219/n | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^3 | 0.125 | level | 4 | (1/16)·n^3 + (-25/16)·n^2 + (19/2)·n - 8 | 0.0156 | read A[i0, i2] (i0=0, i1=1, i2=8); read A[i1, i0] |
| n^3 | 0.125 | level | 4 | (1/16)·n^3 + (-27/16)·n^2 + (93/8)·n - 10 | 0.0156 | read A[i1, i0] |
| n^3 | 0.108 | level | 3 | (1/16)·n^3 + (-7/16)·n^2 + (-11/4)·n + 6 | 0.0156 | read A[i1, i2] (i1=0); read A[i1, i2] (+1) |
| n^3 | 0.0631 | level | 3 | (7/192)·n^3 + (-7/128)·n^2 + (-91/48)·n | 0.00911 | read A[i0, i2] |
| n^3 | 0.0442 | level | (1/8)·n^2 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0312/n | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^3 | 0.0405 | ramp | (1/8)·n^2 + (-1/8)·n + 4  →  (1/8)·n^2 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0312/n | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^3 | 0.0394 | ramp | (1/8)·n^2 + (-1/8)·n + 4  →  (1/8)·n^2 - 1 | (1/8)·n^2 + (-17/4)·n + 8 | 0.0312/n | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^3 | 0.0164 | ramp | (19/8)·n - 1  →  (1/8)·n^2 - 1 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i1, i2] |
| n^3 | 0.0157 | ramp | (13/4)·n - 1  →  (1/8)·n^2 - 1 | (1/16)·n^2 + (-21/8)·n + 27 | 0.0156/n | read A[i0, i2] (i1=0) |
| n^3 | 0.0107 | ramp | (3/8)·n + 3  →  (1/8)·n^2 - 2·n + 3 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i1, i2] |
| n^3 | 0.0107 | ramp | (3/8)·n + 3  →  (1/8)·n^2 - 2·n + 3 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i0, i2] (i1=0) |
| n^2.5 | 0.5 | level | (1/4)·n + 2 | n^2 - 11·n + 25 | 0.25/n | read A[i0, i2] (i2=0) |
| n^2.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n^2 + (-133/8)·n + 42 | 0.219/n | read A[i0, i2] |
| n^2.5 | 0.25 | level | (1/4)·n + 1 | (1/2)·n^2 + (-33/2)·n + 136 | 0.125/n | read A[i0, i2] |
| n^2.5 | 0.219 | level | (1/4)·n + 1 | (7/16)·n^2 + (-7/4)·n - 34 | 0.109/n | read A[i0, i2] |
| n^2.5 | 0.156 | level | (1/4)·n + 2 | (5/16)·n^2 + (-85/8)·n + 90 | 0.0781/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 1 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 1 | (1/16)·n^2 + (29/4)·n - 26 | 0.0156/n | read A[i0, i2] (i0=0, i2=0); read A[i0, i2] (i1=0, i2=0) (+2) |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (-5/8)·n - 6 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (1/8)·n - 18 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 2 | (1/16)·n^2 + (1/4)·n - 20 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0312 | level | (1/4)·n + 1 | (1/16)·n^2 + (1/4)·n - 20 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0261 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i1, i2] |
| n^2.5 | 0.0238 | ramp | (1/8)·n + 3  →  (1/4)·n | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i1, i2] (i2=8); read A[i1, i2] |
| n^2.5 | 0.0238 | ramp | (1/8)·n + 3  →  (1/4)·n | (1/16)·n^2 + (-13/8)·n + 10 | 0.0156/n | read A[i0, i2] |
| n^2.5 | 0.0227 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (7/128)·n^2 + (-25/16)·n + 11 | 0.0137/n | read A[i0, i2] |
| n^2.5 | 0.00296 | ramp | (1/8)·n + 4  →  (1/4)·n - 1 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00195/n | read A[i0, i2] |
| n^2 | 6.06 | level | 3 | (7/2)·n^2 + (-63/2)·n + 28 | 0.875/n | read A[i0, i2] |
| n^2 | 5.85 | level | 3 | (27/8)·n^2 + (-243/8)·n + 27 | 0.844/n | write A[i1, i2] |
| n^2 | 5.3 | level | 3 | (49/16)·n^2 + (-245/8)·n + 49 | 0.766/n | read A[i1, i0] |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 + (-189/8)·n + 21 | 0.656/n | read A[i1, i0] |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 + (-189/8)·n + 21 | 0.656/n | write A[i1, i2] |
| n^2 | 3.71 | level | 2 | (21/8)·n^2 + (-21/8)·n | 0.656/n | read A[i0, i2] |
| n^2 | 3.71 | level | 2 | (21/8)·n^2 + (-21/8)·n | 0.656/n | write A[i1, i2] |
| n^2 | 3.71 | level | 2 | (21/8)·n^2 + (-21/8)·n | 0.656/n | write A[i1, i2] |
| n^2 | 3.44 | level | 1 | (55/16)·n^2 + (-87/8)·n + 14 | 0.859/n | read A[i1, i0] |
| n^2 | 3.09 | level | 2 | (35/16)·n^2 + (-63/8)·n + 21 | 0.547/n | read A[i0, i2]; write A[i1, i2] |
| n^2 | 2.83 | level | (1/8)·n^2 + (-1/8)·n + 2 | 8·n - 16 | 2·n^-2 | read A[i1, i2] (i1=0, i2=0); read A[i1, i2] (i2=0) |
| n^2 | 2.62 | level | 1 | (21/8)·n^2 + (-21/8)·n | 0.656/n | read A[i1, i0] |
| n^2 | 2.12 | level | (1/8)·n^2 + (-1/8)·n + 2 | 6·n - 12 | 1.5·n^-2 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^2 | 1.77 | level | 2 | (5/4)·n^2 + (-45/8)·n | 0.312/n | read A[i0, i2] |
| n^2 | 1.73 | level | 3 | n^2 - 9·n + 8 | 0.25/n | write A[i1, i2] |
| n^2 | 1.62 | level | 3 | (15/16)·n^2 + (-95/8)·n + 35 | 0.234/n | read A[i0, i2] |
| n^2 | 1.62 | level | 3 | (15/16)·n^2 - 5·n - 20 | 0.234/n | read A[i0, i2] |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-63/8)·n + 7 | 0.219/n | read A[i1, i0] |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-31/4)·n + 5 | 0.219/n | read A[i1, i0] |
| n^2 | 1.41 | level | 3 | (13/16)·n^2 - 2·n - 36 | 0.203/n | read A[i0, i2] |
| n^2 | 1.33 | level | 2 | (15/16)·n^2 + (5/4)·n | 0.234/n | read A[i0, i2] |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 + (-7/8)·n | 0.219/n | write A[i1, i2] |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 + (-119/8)·n + 63 | 0.219/n | read A[i1, i0] |
| n^2 | 1.06 | level | 2 | (3/4)·n^2 - 6·n | 0.188/n | read A[i1, i0] |
| n^2 | 1.06 | level | 2 | (3/4)·n^2 + (-3/4)·n | 0.188/n | write A[i1, i2] |
| n^2 | 1 | level | 1 | n^2 + (5/8)·n + 1 | 0.25/n | read A[i1, i0]; read A[i0, i2] |
| n^2 | 0.873 | ramp | 1  →  2 | (7/8)·n^2 + (-7/8)·n - 6 | 0.219/n | read A[i1, i0]; read A[i0, i2] |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-7/8)·n - 21 | 0.109/n | read A[i1, i0] |
| n^2 | 0.707 | level | 2 | (1/2)·n^2 + (-15/2)·n + 35 | 0.125/n | read A[i0, i2] (i0=0, i1=1, i2=0); write A[i1, i2] (i0=0, i2=0) (+6) |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-45/8)·n + 21 | 0.0938/n | read A[i0, i2] |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-7/8)·n | 0.109/n | read A[i0, i2] |
| n^2 | 0.53 | level | 2 | (3/8)·n^2 + (-3/8)·n | 0.0938/n | read A[i0, i2] |
| n^2 | 0.5 | level | 1 | (1/2)·n^2 + (-3/2)·n + 18 | 0.125/n | read A[i0, i2] (i0=0, i1=0, i2=0); write A[i1, i2] (i0=0, i1=0, i2=0) (+2) |
| n^2 | 0.442 | level | 2 | (5/16)·n^2 + (15/8)·n | 0.0781/n | read A[i0, i2] |
| n^2 | 0.442 | level | 2 | (5/16)·n^2 + (-7/4)·n + 2 | 0.0781/n | read A[i1, i2]; read A[i0, i2] (+1) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 + (-49/8)·n + 21 | 0.109/n | write A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 3 | 0.25·n^-2 | read A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | n - 3 | 0.25·n^-2 | read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 2 | n - 1 | 0.25·n^-2 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (17/8) | n - 2 | 0.25·n^-2 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 2 | n - 10 | 0.25·n^-2 | read A[i1, i0] (i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 9 | 0.25·n^-2 | read A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 8 | 0.25·n^-2 | read A[i0, i2] (i0=1, i1=0, i2=8); read A[i1, i0] (i0=8, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 2 | 0.25·n^-2 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 9 | 0.25·n^-2 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^2 | 0.25 | level | 4 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0312/n | read A[i1, i0] |
| n^2 | 0.233 | ramp | (9/4)·n  →  (1/8)·n^2 | n - 17 | 0.25·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.227 | ramp | (3/8)·n + 2  →  (1/8)·n^2 - n + 2 | n - 10 | 0.25·n^-2 | read A[i0, i2] (i1=0, i2=0) |
| n^2 | 0.227 | ramp | (3/8)·n + 1  →  (1/8)·n^2 - n + 1 | n - 10 | 0.25·n^-2 | read A[i1, i2] |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0312/n | write A[i1, i2] |
| n^2 | 0.207 | ramp | (1/2)·n  →  (1/8)·n^2 | (7/8)·n - 2 | 0.219·n^-2 | read A[i1, i2] (i2=0) |
| n^2 | 0.199 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-9/8)·n + 5 | (7/8)·n - 14 | 0.219·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.199 | ramp | (11/8)·n - 1  →  (1/8)·n^2 + (-9/8)·n + 5 | (7/8)·n - 14 | 0.219·n^-2 | read A[i1, i0] (i2=0) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0312/n | read A[i1, i0] |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 - n | 0.0312/n | read A[i1, i0] |
| n^2 | 0.146 | ramp | (5/2)·n - 1  →  (1/8)·n^2 + (-1/8)·n + 2 | (5/8)·n - 10 | 0.156·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-1/2)·n | 0.0156/n | read A[i0, i2] |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (3/8)·n | 0.0156/n | read A[i0, i2] |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (3/8)·n | 0.0156/n | read A[i0, i2] |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (3/8)·n - 7 | 0.0156/n | read A[i1, i0] |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-1/8)·n - 1 | 0.0156/n | read A[i0, i2]; write A[i1, i2] |
| n^2 | 0.029 | ramp | (19/8)·n - 1  →  (1/8)·n^2 + (-3/4)·n + 2 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0289 | ramp | (9/4)·n  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 2 | 0.0312·n^-2 | read A[i1, i2] (i2=0) |
| n^2 | 0.0289 | ramp | (9/4)·n - 1  →  (1/8)·n^2 + (-7/8)·n + 2 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i1, i2] |
| n^2 | 0.0289 | ramp | (9/4)·n - 1  →  (1/8)·n^2 + (-7/8)·n + 2 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0285 | ramp | (25/8)·n - 1  →  (1/8)·n^2 - n + 3 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0279 | ramp | (5/4)·n  →  (1/8)·n^2 - 2·n + 6 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] (i1=0) |
| n^2 | 0.0277 | ramp | (9/4)·n - 2  →  (1/8)·n^2 - 2·n + 6 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i1, i0] (i2=0) |
| n^1.5 | 3.78 | ramp | 5  →  (1/4)·n + 1 | (63/8)·n - 34 | 1.97·n^-2 | read A[i0, i2] |
| n^1.5 | 2.5 | level | (1/4)·n + 1 | 5·n - 65 | 1.25·n^-2 | read A[i0, i2] |
| n^1.5 | 0.53 | ramp | 5  →  (1/4)·n + 1 | (9/8)·n - 11 | 0.281·n^-2 | read A[i0, i2] |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 10 | 0.25·n^-2 | read A[i0, i2] |
| n^1.5 | 0.5 | level | (1/4)·n + 1 | n - 16 | 0.25·n^-2 | read A[i0, i2] |
| n^1.5 | 0.5 | level | (1/4)·n | n | 0.25·n^-2 | read A[i1, i2] (i0=1, i1=0, i2=0); read A[i1, i2] (i2=0) |
| n^1.5 | 0.438 | level | (1/4)·n | (7/8)·n - 14 | 0.219·n^-2 | read A[i0, i2] |
| n^1.5 | 0.354 | level | (1/8)·n + 2 | n - 8 | 0.25·n^-2 | read A[i0, i2] (i0=0, i1=1, i2=8); read A[i0, i2] (i2=0) |
| n^1.5 | 0.354 | level | (1/8)·n + 1 | n - 9 | 0.25·n^-2 | read A[i1, i2] |
| n^1.5 | 0.288 | ramp | 5  →  (1/4)·n - 1 | (7/8)·n - 14 | 0.219·n^-2 | read A[i1, i0] (i2=0) |
| n^1.5 | 0.125 | level | (1/4)·n + 1 | (1/4)·n - 4 | 0.0625·n^-2 | read A[i1, i2] (i0=1, i1=0); read A[i0, i2] (i1=0) |
| n^1.5 | 0.0625 | level | (1/4)·n | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0514 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0514 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0502 | ramp | (1/8)·n + 3  →  (1/4)·n - 1 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] (i0=0, i1=1) |
| n^1.5 | 0.0502 | ramp | (1/8)·n + 3  →  (1/4)·n - 1 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i0, i2] |
| n^1.5 | 0.0502 | ramp | (1/8)·n + 3  →  (1/4)·n - 1 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i1, i2] |
| n^1.5 | 0.0412 | ramp | 5  →  (1/4)·n - 1 | (1/8)·n - 2 | 0.0312·n^-2 | read A[i1, i0] (i2=0) |
| n^1 | 7 | level | 1 | 7·n - 14 | 1.75·n^-2 | read A[i1, i0] (i0=7, i2=0); read A[i1, i0] (i2=0) |
| n^1 | 6.12 | level | 1 | (49/8)·n - 21 | 1.53·n^-2 | write A[i1, i2] |
| n^1 | 4.95 | level | 2 | (7/2)·n - 28 | 0.875·n^-2 | read A[i1, i0] |
| n^1 | 3.71 | level | 2 | (21/8)·n - 21 | 0.656·n^-2 | read A[i1, i0] |
| n^1 | 3.71 | level | 2 | (21/8)·n - 21 | 0.656·n^-2 | read A[i0, i2] |
| n^1 | 3.54 | level | 2 | (5/2)·n - 20 | 0.625·n^-2 | read A[i1, i0]; read A[i0, i2] |
| n^1 | 3.5 | level | 1 | (7/2)·n - 7 | 0.875·n^-2 | read A[i1, i0] |
| n^1 | 2.83 | level | (1/8)·n^2 + (-1/8)·n + 2 | 8 | 2·n^-3 | read A[i1, i0] (i1=0, i2=0); read A[i1, i0] (i2=0) |
| n^1 | 2.62 | level | 1 | (21/8)·n | 0.656·n^-2 | read A[i1, i0] |
| n^1 | 2.62 | level | 1 | (21/8)·n | 0.656·n^-2 | read A[i0, i2] |
| n^1 | 2.12 | level | (1/8)·n^2 | 6 | 1.5·n^-3 | read A[i1, i2] |
| n^1 | 1.88 | level | 1 | (15/8)·n - 2 | 0.469·n^-2 | read A[i1, i0] |
| n^1 | 1.88 | level | 1 | (15/8)·n - 1 | 0.469·n^-2 | read A[i0, i2]; write A[i1, i2] |
| n^1 | 1.88 | level | 1 | (15/8)·n - 1 | 0.469·n^-2 | read A[i1, i0] |
| n^1 | 1.62 | level | 1 | (13/8)·n - 1 | 0.406·n^-2 | read A[i0, i2]; write A[i1, i2] |
| n^1 | 1.52 | level | 3 | (7/8)·n - 14 | 0.219·n^-2 | read A[i1, i0] |
| n^1 | 1.41 | level | 2 | n - 8 | 0.25·n^-2 | read A[i0, i2] |
| n^1 | 1.24 | level | 2 | (7/8)·n - 13 | 0.219·n^-2 | read A[i1, i2] |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 0.219·n^-2 | read A[i1, i0] |
| n^1 | 1.24 | level | 2 | (7/8)·n - 7 | 0.219·n^-2 | read A[i1, i0] |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.219·n^-2 | read A[i0, i2] |
| n^1 | 0.707 | level | (1/8)·n^2 | 2 | 0.5·n^-3 | read A[i1, i2] (i1=0); read A[i1, i2] |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.25·n^-3 | read A[i0, i2] (i0=1, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.25·n^-3 | read A[i1, i2] (i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.25·n^-3 | read A[i1, i2] |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/8)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/8)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n - 2 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 4 | 1 | 0.25·n^-3 | read A[i0, i2] (i0=8, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/8)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/8)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 3 | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 2 | 1 | 0.25·n^-3 | read A[i0, i2] (i0=1, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.25·n^-3 | read A[i1, i0] (i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^1 | 0.177 | level | 2 | (1/8)·n - 3 | 0.0312·n^-2 | read A[i1, i2] |
| n^1 | 0.177 | level | 2 | (1/8)·n - 1 | 0.0312·n^-2 | read A[i1, i0] |
| n^1 | 0.177 | level | 2 | (1/8)·n - 1 | 0.0312·n^-2 | read A[i0, i2] |
| n^0.5 | 7 | level | (1/4)·n + 1 | 14 | 3.5·n^-3 | read A[i0, i2] |
| n^0.5 | 6.5 | level | (1/4)·n + 1 | 13 | 3.25·n^-3 | read A[i0, i2] |
| n^0.5 | 3.5 | level | (1/4)·n | 7 | 1.75·n^-3 | read A[i0, i2] (i1=0); read A[i0, i2] |
| n^0.5 | 3.5 | level | (1/4)·n + 1 | 7 | 1.75·n^-3 | read A[i0, i2] |
| n^0.5 | 3 | level | (1/4)·n + 1 | 6 | 1.5·n^-3 | read A[i0, i2] |
| n^0.5 | 2.5 | level | (1/4)·n | 5 | 1.25·n^-3 | read A[i0, i2] |
| n^0.5 | 1.46 | level | (17/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.37 | level | (15/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.32 | level | (7/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.27 | level | (13/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.22 | level | (3/2)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.25·n^-3 | read A[i1, i0] (i0=8, i1=7, i2=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.25·n^-3 | read A[i1, i2] (i0=8, i1=7, i2=0) |
| n^0.5 | 1.06 | level | (9/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 1 | level | n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.935 | level | (7/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.791 | level | (5/8)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.707 | level | (1/2)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i1, i2] (i0=1, i1=0) |
| n^0.5 | 0.5 | level | (1/4)·n + (11/4) | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i1=0); read A[i0, i2] |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] (i0=0, i1=1) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.25·n^-3 | read A[i0, i2] |
| n^0.5 | 0.354 | level | (1/8)·n + 2 | 1 | 0.25·n^-3 | read A[i1, i2] |
| n^0 | 48.5 | level | 3 | 28 | 7·n^-3 | read A[i1, i2] |
| n^0 | 12.1 | level | 3 | 7 | 1.75·n^-3 | read A[i1, i2] |
| n^0 | 10.4 | level | 3 | 6 | 1.5·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 8.49 | level | 2 | 6 | 1.5·n^-3 | read A[i1, i2] |
| n^0 | 2.24 | level | 5 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.25·n^-3 | read A[i1, i0] (i2=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.25·n^-3 | read A[i1, i2] (i0=8, i1=8, i2=8) |

Each k-sweep re-reads the whole distance matrix: two n^4 level terms of `read A[i1,i2]`/`read A[i0,i2]` at exactly (1/8)n^2 lines (no row additive term — the k-row is already resident), combined coefficient 0.0442, identical to gemm's constant. The n^3.5 terms are the k-row reuses at (1/4)n + 2 lines.

## gemm — infinite-repeat  [`exact`]

Accesses $A(n) = 4·n^3 + 2·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^4  +  0.0625·n^3.5  +  6.83·n^3  +  1.05·n^2.5  +  7.84·n^2  +  0.354·n^1.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0331 | level | (1/8)·n^2 + (3/8)·n + 1 | (3/32)·n^3 + (-3/2)·n^2 | 0.0234 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^4 | 0.011 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/32)·n^3 + (-3/4)·n^2 + 4·n | 0.00781 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^3.5 | 0.0547 | level | (1/4)·n | (7/64)·n^3 + (-7/4)·n^2 | 0.0273 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00781 | level | (1/4)·n + 1 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.00391 | read C[i3, i4] (i0=0, i1=0, i3=0); read A[i1, i4] (i0=0) |
| n^3 | 3.03 | level | 3 | (7/4)·n^3 | 0.438 | write A[i1, i4] (i0=0, i3=0); read C[i3, i4] (i0=0) (+1) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 | 0.219 | read B[i1, i3] (i0=0, i1=0, i3=0, i4=0); read C[i3, i4] (i0=0, i1=0, i3=0, i4=0) (+1) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 | 0.219 | read A[i1, i4] (i0=0) |
| n^3 | 0.265 | level | (1/8)·n^2 + (3/8)·n + 1 | (3/4)·n^2 | 0.188/n | read C[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i4=0) |
| n^3 | 0.265 | level | (1/8)·n^2 + (3/8)·n + 1 | (3/4)·n^2 | 0.188/n | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^3 | 0.25 | level | 4 | (1/8)·n^3 - n^2 | 0.0312 | read C[i3, i4] (i0=0, i1=0, i3=0, i4=0); read C[i3, i4] (i0=0, i3=0, i4=0) (+1) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 | 0.0312 | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^3 | 0.0884 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/4)·n^2 - 2·n | 0.0625/n | read C[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i4=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 | (1/8)·n^2 - 2·n | 0.0312/n | read B[i1, i3] (i0=0, i1=0, i4=0); read B[i1, i3] (i0=0, i4=0) |
| n^3 | 0.0765 | level | (3/8)·n^2 + (-1/8)·n + 1 | (1/8)·n^2 - 2·n | 0.0312/n | read A[i1, i2] (i0=0, i1=0); read A[i1, i2] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 - 2·n | 0.0312/n | read C[i3, i4] (i0=0, i1=0, i3=0); read C[i3, i4] (i0=0, i3=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 - n | 0.0312/n | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^2 - n | 0.0312/n | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^3 | 0.0421 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 - 2·n | 0.0312/n | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^2.5 | 0.5 | level | (1/4)·n | n^2 | 0.25/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.5 | level | (1/4)·n | n^2 - n | 0.25/n | read A[i1, i2] (i0=0, i1=0); read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.0514 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/8)·n^2 - 2·n | 0.0312/n | read A[i1, i4] (i0=0, i3=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 | 0.219/n | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 | 0.438/n | read A[i1, i2] (i0=0); write A[i1, i2] (i0=0) |
| n^2 | 0.612 | level | (3/8)·n^2 | n | 0.25·n^-2 | read B[i1, i3] (i0=0, i1=0, i4=0); read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.612 | level | (3/8)·n^2 | n | 0.25·n^-2 | read B[i1, i3] (i0=0, i1=0, i3=0, i4=0); read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/8)·n + 1 | n | 0.25·n^-2 | read A[i1, i2] (i0=0, i1=0, i2=0); read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.612 | level | (3/8)·n^2 + (-1/8)·n | n | 0.25·n^-2 | read A[i1, i2] (i0=0, i1=0); read A[i1, i2] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n | 0.25·n^-2 | read C[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i4=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n | 0.25·n^-2 | read C[i3, i4] (i0=0, i1=0, i4=0); read C[i3, i4] (i0=0, i4=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n | 0.25·n^-2 | read C[i3, i4] (i0=0, i1=0); read C[i3, i4] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n | 0.25·n^-2 | read C[i3, i4] (i0=0, i1=0, i3=0); read C[i3, i4] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n | 0.25·n^-2 | read C[i3, i4] (i0=0, i1=0, i3=0, i4=0); read C[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 | 0.0312/n | write A[i1, i2] (i0=0) |
| n^1.5 | 0.354 | level | (1/8)·n | n | 0.25·n^-2 | read A[i1, i2] (i0=0, i1=0); read A[i1, i4] (i0=0, i3=0, i4=0) |

The n^4 order is carried by two bins of `read C[i3,i4]`: each C element is re-touched on the next outer iteration at distance (1/8)n^2 + (3/8)n + 1 lines — the streamed matrix plus one row — with combined population n^3/8 (one reuse per line element per sweep). Their coefficients sum to 0.0442 = (1/8)*sqrt(1/8), the constant the earlier study measured. Below it: A-row reuse at (1/4)n lines (order n^3.5) and the line-reuse bulk at distance 1–3 (order n^3, >90% of accesses). Under infinite repeat the same terms persist (the wraparound consumers are the first C reads of the next pass, annotated i1=0); no new order arises because all matrices are already re-touched within a pass. Miss reading: the miss ratio steps at 64·((1/4)n) bytes (row) and 64·((1/8)n^2) bytes (matrix) — the second is the tiling cliff.

## gemm — single-shot  [`exact`]

Accesses $A(n) = 4·n^3 + 2·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0442·n^4  +  0.0625·n^3.5  +  6.68·n^3  +  1.05·n^2.5  +  5.39·n^2  +  0.354·n^1.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0331 | level | (1/8)·n^2 + (3/8)·n + 1 | (3/32)·n^3 + (-51/32)·n^2 + (3/2)·n | 0.0234 | read C[i3, i4] (i0=0) |
| n^4 | 0.011 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/32)·n^3 + (-25/32)·n^2 + (19/4)·n - 4 | 0.00781 | read C[i3, i4] (i0=0) |
| n^3.5 | 0.0547 | level | (1/4)·n | (7/64)·n^3 + (-7/4)·n^2 | 0.0273 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00781 | level | (1/4)·n + 1 | (1/64)·n^3 + (-3/8)·n^2 + 2·n | 0.00391 | read A[i1, i4] (i0=0) |
| n^3 | 3.25 | level | 3 | (15/8)·n^3 | 0.469 | read B[i1, i3] (i0=0); write A[i1, i4] (i0=0) |
| n^3 | 1.52 | level | 3 | (7/8)·n^3 | 0.219 | read C[i3, i4] (i0=0) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 | 0.219 | read A[i1, i4] (i0=0) |
| n^3 | 0.265 | level | (1/8)·n^2 + (3/8)·n + 1 | (3/4)·n^2 + (-3/4)·n | 0.188/n | read C[i3, i4] (i0=0, i4=0) |
| n^3 | 0.265 | level | (1/8)·n^2 + (3/8)·n + 1 | (3/4)·n^2 + (-3/4)·n | 0.188/n | read C[i3, i4] (i0=0) |
| n^3 | 0.25 | level | 4 | (1/8)·n^3 - n^2 | 0.0312 | read C[i3, i4] (i0=0, i3=0, i4=0); read B[i1, i3] (i0=0) |
| n^3 | 0.0884 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/4)·n^2 + (-9/4)·n + 2 | 0.0625/n | read C[i3, i4] (i0=0, i4=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0312/n | read C[i3, i4] (i0=0, i3=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0312/n | read C[i3, i4] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-9/8)·n + 1 | 0.0312/n | read C[i3, i4] (i0=0) |
| n^3 | 0.042 | ramp | (1/8)·n^2 + (1/4)·n + 3  →  (1/8)·n^2 + (3/8)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.0312/n | read C[i3, i4] (i0=0) |
| n^2.5 | 0.5 | level | (1/4)·n | n^2 | 0.25/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.5 | level | (1/4)·n | n^2 - n | 0.25/n | read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.0514 | ramp | (1/8)·n + 2  →  (1/4)·n - 1 | (1/8)·n^2 - 2·n | 0.0312/n | read A[i1, i4] (i0=0, i3=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 | 0.219/n | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 1 | level | 1 | n^2 | 0.25/n | write A[i1, i2] (i0=0); read A[i1, i4] (i0=0, i3=0, i4=0) (+1) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.219/n | read A[i1, i2] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | n - 1 | 0.25·n^-2 | read C[i3, i4] (i0=0, i4=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.25·n^-2 | read C[i3, i4] (i0=0, i4=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | n - 1 | 0.25·n^-2 | read C[i3, i4] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.25·n^-2 | read C[i3, i4] (i0=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | n - 1 | 0.25·n^-2 | read C[i3, i4] (i0=0, i3=0, i4=0) |
| n^1.5 | 0.354 | level | (1/8)·n | n | 0.25·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=0) |

The n^4 order is carried by two bins of `read C[i3,i4]`: each C element is re-touched on the next outer iteration at distance (1/8)n^2 + (3/8)n + 1 lines — the streamed matrix plus one row — with combined population n^3/8 (one reuse per line element per sweep). Their coefficients sum to 0.0442 = (1/8)*sqrt(1/8), the constant the earlier study measured. Below it: A-row reuse at (1/4)n lines (order n^3.5) and the line-reuse bulk at distance 1–3 (order n^3, >90% of accesses). Under infinite repeat the same terms persist (the wraparound consumers are the first C reads of the next pass, annotated i1=0); no new order arises because all matrices are already re-touched within a pass. Miss reading: the miss ratio steps at 64·((1/4)n) bytes (row) and 64·((1/8)n^2) bytes (matrix) — the second is the tiling cliff.

## gemver — infinite-repeat  [`exact`]

Accesses $A(n) = 14·n^2 + 3·n$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.114·n^3  +  1.28·n^2.5  +  24.1·n^2  +  8.36·n^1.5  +  21.6·n^1  +  2.95·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0317 | ramp | (21/8)·n  →  (1/8)·n^2 + (3/4)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.00781 | read A[i4, i3] (i0=0) |
| n^3 | 0.0317 | ramp | (1/8)·n^2 + (1/2)·n + 2  →  (1/8)·n^2 + (5/8)·n + 1 | (3/32)·n^2 + (-3/2)·n | 0.0067 | read A[i1, i2] (i0=0) |
| n^3 | 0.0317 | ramp | (21/8)·n - 3  →  (1/8)·n^2 + (1/2)·n - 1 | (7/64)·n^2 + (-15/8)·n + 2 | 0.00781 | read A[i6, i7] (i0=0) |
| n^3 | 0.00514 | ramp | (1/8)·n^2 + (1/2)·n + 3  →  (1/8)·n^2 + (5/8)·n + 1 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read A[i1, i2] (i0=0) |
| n^3 | 0.00514 | ramp | (1/8)·n^2 + (1/2)·n + 2  →  (1/8)·n^2 + (5/8)·n | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read A[i1, i2] (i0=0) |
| n^3 | 0.00444 | ramp | (7/2)·n - 12  →  (1/8)·n^2 + (3/4)·n - 8 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read A[i4, i3] (i0=0) |
| n^3 | 0.00443 | ramp | (7/2)·n - 16  →  (1/8)·n^2 + (1/2)·n - 7 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read A[i6, i7] (i0=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 + (-7/4)·n | 0.0625 | read A[i4, i3] (i0=0) |
| n^2.5 | 0.134 | level | (3/8)·n + 3 | (7/32)·n^2 + (-7/2)·n | 0.0156 | read C[i2] (i0=0); read E[i2] (i0=0) |
| n^2.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^2 + (-7/4)·n | 0.00781 | read G[i4] (i0=0) |
| n^2.5 | 0.0547 | level | (1/4)·n + 1 | (7/64)·n^2 + (-13/8)·n - 2 | 0.00781 | read F[i7] (i0=0, i6=0); read F[i7] (i0=0) |
| n^2.5 | 0.0191 | level | (3/8)·n + 5 | (1/32)·n^2 + (-3/4)·n + 4 | 0.00223 | read C[i2] (i0=0); read E[i2] (i0=0) |
| n^2.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read G[i4] (i0=0) |
| n^2.5 | 0.00781 | level | (1/4)·n + 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read F[i7] (i0=0) |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 + (-7/4)·n | 0.188 | read G[i4] (i0=0); read A[i6, i7] (i0=0) (+1) |
| n^2 | 4.19 | level | 5 | (15/8)·n^2 + (-7/8)·n | 0.134 | read C[i2] (i0=0, i1=0); read C[i2] (i0=0) (+2) |
| n^2 | 3.91 | level | 5 | (7/4)·n^2 | 0.125 | read B[i1] (i0=0); read D[i1] (i0=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 | 0.125 | write F[i3] (i0=0); write I[i6] (i0=0) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 + (7/8)·n | 0.0625 | read E[i2] (i0=0, i1=0); write A[i1, i2] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 + (7/8)·n | 0.125 | read F[i3] (i0=0); read F[i5] (i0=0) (+2) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.0625 | read A[i1, i2] (i0=0) |
| n^2 | 0.612 | level | 6 | (1/4)·n^2 - 2·n | 0.0179 | read B[i1] (i0=0); read D[i1] (i0=0) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 | 0.0179 | write F[i3] (i0=0); write I[i6] (i0=0, i7=0) (+1) |
| n^2 | 0.354 | ramp | (1/8)·n^2 + (1/2)·n + 2  →  (1/8)·n^2 + (3/4)·n | n - 2 | 0.0714/n | read A[i4, i3] (i0=0) |
| n^2 | 0.31 | ramp | (1/8)·n^2 + (1/2)·n + 1  →  (1/8)·n^2 + (5/8)·n + 1 | (7/8)·n - 1 | 0.0625/n | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.31 | ramp | (1/8)·n^2 + (3/8)·n + 1  →  (1/8)·n^2 + (1/2)·n | (7/8)·n - 1 | 0.0625/n | read A[i6, i7] (i0=0, i7=0) |
| n^2 | 0.267 | ramp | (1/8)·n^2 + (1/2)·n + 2  →  (1/8)·n^2 + (5/8)·n + 1 | (3/4)·n | 0.0536/n | read A[i1, i2] (i0=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (-1/4)·n | 0.0179 | read F[i3] (i0=0); read I[i6] (i0=0) |
| n^2 | 0.213 | ramp | (13/8)·n + 1  →  (1/8)·n^2 + (3/8)·n + 3 | (7/8)·n - 1 | 0.0625/n | read A[i4, i3] (i0=0, i3=0) |
| n^2 | 0.183 | ramp | (13/8)·n - 1  →  (1/8)·n^2 + (1/4)·n + 2 | (3/4)·n | 0.0536/n | read A[i6, i7] (i0=0) |
| n^2 | 0.177 | level | (1/8)·n^2 + n | (1/2)·n - 8 | 0.0357/n | read B[i1] (i0=0, i2=0); read D[i1] (i0=0, i2=0) (+2) |
| n^2 | 0.0884 | level | (1/8)·n^2 + (3/4)·n + 4 | (1/4)·n - 4 | 0.0179/n | read C[i2] (i0=0, i1=0); read E[i2] (i0=0, i1=0) |
| n^2 | 0.0597 | ramp | (19/8)·n - 7  →  (1/8)·n^2 + (-1/2)·n + 8 | (1/4)·n - 2 | 0.0179/n | read A[i6, i7] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.00893/n | read A[i6, i7] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (3/4)·n + 1 | (1/8)·n - 2 | 0.00893/n | read F[i3] (i0=0, i4=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + n | (1/8)·n - 2 | 0.00893/n | read G[i4] (i0=0, i3=0) |
| n^2 | 0.0434 | ramp | (1/8)·n^2 + (1/2)·n + 3  →  (1/8)·n^2 + (5/8)·n + 1 | (1/8)·n - 1 | 0.00893/n | read A[i1, i2] (i0=0) |
| n^2 | 0.0434 | ramp | (1/8)·n^2 + (1/2)·n + 2  →  (1/8)·n^2 + (5/8)·n | (1/8)·n - 1 | 0.00893/n | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0434 | ramp | (1/8)·n^2 + (1/2)·n + 2  →  (1/8)·n^2 + (5/8)·n | (1/8)·n - 1 | 0.00893/n | read A[i1, i2] (i0=0) |
| n^2 | 0.0433 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n | (1/8)·n - 1 | 0.00893/n | read A[i6, i7] (i0=0, i7=0) |
| n^2 | 0.0423 | ramp | (1/8)·n^2 + (5/8)·n + 2  →  (1/8)·n^2 + (3/4)·n - 1 | (1/8)·n - 2 | 0.00893/n | read A[i4, i3] (i0=0, i4=0) |
| n^2 | 0.0422 | ramp | (1/8)·n^2 + (1/2)·n + 3  →  (1/8)·n^2 + (5/8)·n | (1/8)·n - 2 | 0.00893/n | read A[i1, i2] (i0=0) |
| n^2 | 0.0421 | ramp | (1/8)·n^2 + (1/4)·n + 4  →  (1/8)·n^2 + (1/2)·n - 2 | (1/8)·n - 2 | 0.00893/n | read A[i1, i2] (i0=0, i1=0) |
| n^2 | 0.03 | ramp | (5/2)·n - 5  →  (1/8)·n^2 + (-3/8)·n + 9 | (1/8)·n - 1 | 0.00893/n | read A[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0291 | ramp | (5/2)·n - 1  →  (1/8)·n^2 + (-5/8)·n + 2 | (1/8)·n - 2 | 0.00893/n | read A[i6, i7] (i0=0, i6=0) |
| n^2 | 0.029 | ramp | (5/2)·n - 2  →  (1/8)·n^2 + (-3/4)·n + 4 | (1/8)·n - 2 | 0.00893/n | read A[i4, i3] (i0=0) |
| n^2 | 0.0281 | ramp | (11/8)·n - 2  →  (1/8)·n^2 + (-7/4)·n + 1 | (1/8)·n - 2 | 0.00893/n | read F[i5] (i0=0) |
| n^1.5 | 1.15 | level | (3/8)·n + 3 | (15/8)·n - 1 | 0.134/n | read C[i2] (i0=0); read E[i2] (i0=0) |
| n^1.5 | 1.07 | level | (3/8)·n + 3 | (7/4)·n | 0.125/n | read C[i2] (i0=0, i2=0); read E[i2] (i0=0, i2=0) |
| n^1.5 | 0.928 | level | (9/8)·n + (15/8) | (7/8)·n + (-7/8) | 0.0625/n | read A[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.0625/n | read A[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.0625/n | read A[i4, i3] (i0=0, i4=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.0625/n | read G[i4] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.0625/n | read G[i4] (i0=0, i4=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n + 1 | 0.0625/n | read F[i7] (i0=0, i6=0); read F[i7] (i0=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n + 1 | 0.0625/n | read F[i7] (i0=0, i6=0, i7=0); read F[i7] (i0=0, i7=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.00893/n | read G[i4] (i0=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.00893/n | read G[i4] (i0=0, i4=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 4 | (1/8)·n - 1 | 0.00893/n | read C[i2] (i0=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 5 | (1/8)·n - 1 | 0.00893/n | read E[i2] (i0=0, i2=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 4 | (1/8)·n - 1 | 0.00893/n | read C[i2] (i0=0, i2=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.00893/n | read F[i7] (i0=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.00893/n | read F[i7] (i0=0, i7=0) |
| n^1 | 4.29 | level | 6 | (7/4)·n | 0.125/n | read B[i1] (i0=0, i2=0); read D[i1] (i0=0, i2=0) |
| n^1 | 3.03 | level | 3 | (7/4)·n | 0.125/n | read G[i4] (i0=0); read F[i7] (i0=0, i6=0) |
| n^1 | 2.47 | level | 2 | (7/4)·n | 0.125/n | read H[i5] (i0=0); write F[i5] (i0=0) |
| n^1 | 1.41 | level | (1/8)·n^2 + n | 4 | 0.286·n^-2 | read B[i1] (i0=0, i2=0); read D[i1] (i0=0, i2=0) (+2) |
| n^1 | 1.41 | level | (1/8)·n^2 + n | 4 | 0.286·n^-2 | read B[i1] (i0=0, i1=0, i2=0); read D[i1] (i0=0, i1=0, i2=0) (+2) |
| n^1 | 1.06 | level | (1/8)·n^2 + (15/8)·n + 7 | 3 | 0.214·n^-2 | read B[i1] (i0=0, i2=0); read D[i1] (i0=0, i2=0) (+1) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n + (7/4) | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n + 1 | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.0714·n^-2 | read A[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.0714·n^-2 | read A[i1, i2] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0, i6=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + 1 | 1 | 0.0714·n^-2 | read F[i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (11/8)·n + (11/2) | 1 | 0.0714·n^-2 | read A[i1, i2] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n + 2 | 1 | 0.0714·n^-2 | read A[i1, i2] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n + 1 | 1 | 0.0714·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + 2 | 1 | 0.0714·n^-2 | read E[i2] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + n | 1 | 0.0714·n^-2 | read G[i4] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + 3 | 1 | 0.0714·n^-2 | read C[i2] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n | 1 | 0.0714·n^-2 | read F[i5] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + n | 1 | 0.0714·n^-2 | read H[i5] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n + 1 | 1 | 0.0714·n^-2 | read F[i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + 4 | 1 | 0.0714·n^-2 | read E[i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + 3 | 1 | 0.0714·n^-2 | read C[i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.177 | level | 2 | (1/8)·n | 0.00893/n | write F[i5] (i0=0) |
| n^0.5 | 1.22 | level | (3/2)·n | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0, i6=0) |
| n^0.5 | 1.22 | level | (3/2)·n | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n - 1 | 1 | 0.0714·n^-2 | read F[i5] (i0=0) |

The rank-1-update matrix A is walked twice (by rows, then transposed): ramps from ~(21/8)n up to (1/8)n^2 + (3/4)n lines on both `read A[i4,i3]` and `read A[i6,i7]` give d = 3.0, headroom +1.0 single-shot, same mechanism as mvt.

## gemver — single-shot  [`exact`]

Accesses $A(n) = 14·n^2 + 3·n$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0722·n^3  +  1.28·n^2.5  +  22.9·n^2  +  8.36·n^1.5  +  12.7·n^1  +  2.95·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0317 | ramp | (21/8)·n  →  (1/8)·n^2 + (3/4)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.00781 | read A[i4, i3] (i0=0) |
| n^3 | 0.0317 | ramp | (21/8)·n - 3  →  (1/8)·n^2 + (1/2)·n - 1 | (7/64)·n^2 + (-15/8)·n + 2 | 0.00781 | read A[i6, i7] (i0=0) |
| n^3 | 0.00444 | ramp | (7/2)·n - 12  →  (1/8)·n^2 + (3/4)·n - 8 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read A[i4, i3] (i0=0) |
| n^3 | 0.00443 | ramp | (7/2)·n - 16  →  (1/8)·n^2 + (1/2)·n - 7 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read A[i6, i7] (i0=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 + (-7/4)·n | 0.0625 | read A[i4, i3] (i0=0) |
| n^2.5 | 0.134 | level | (3/8)·n + 3 | (7/32)·n^2 + (-7/2)·n | 0.0156 | read C[i2] (i0=0); read E[i2] (i0=0) |
| n^2.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^2 + (-7/4)·n | 0.00781 | read G[i4] (i0=0) |
| n^2.5 | 0.0547 | level | (1/4)·n + 1 | (7/64)·n^2 + (-13/8)·n - 2 | 0.00781 | read F[i7] (i0=0, i6=0); read F[i7] (i0=0) |
| n^2.5 | 0.0191 | level | (3/8)·n + 5 | (1/32)·n^2 + (-3/4)·n + 4 | 0.00223 | read C[i2] (i0=0); read E[i2] (i0=0) |
| n^2.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read G[i4] (i0=0) |
| n^2.5 | 0.00781 | level | (1/4)·n + 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00112 | read F[i7] (i0=0) |
| n^2 | 6.15 | level | 5 | (11/4)·n^2 | 0.196 | read B[i1] (i0=0); read D[i1] (i0=0) (+2) |
| n^2 | 3.91 | level | 5 | (7/4)·n^2 | 0.125 | read C[i2] (i0=0); read E[i2] (i0=0) |
| n^2 | 3.46 | level | 3 | 2·n^2 | 0.143 | read G[i4] (i0=0); write F[i3] (i0=0) (+3) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 | 0.125 | read G[i4] (i0=0); read F[i7] (i0=0) |
| n^2 | 2 | level | 1 | 2·n^2 - 2·n | 0.143 | read F[i3] (i0=0); read F[i5] (i0=0) (+1) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 | 0.0625 | read A[i6, i7] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.0625 | read A[i1, i2] (i0=0) |
| n^2 | 0.612 | level | 6 | (1/4)·n^2 - 2·n | 0.0179 | read C[i2] (i0=0, i2=0); read E[i2] (i0=0, i2=0) (+2) |
| n^2 | 0.354 | ramp | (1/8)·n^2 + (1/2)·n + 2  →  (1/8)·n^2 + (3/4)·n | n - 2 | 0.0714/n | read A[i4, i3] (i0=0) |
| n^2 | 0.31 | ramp | (1/8)·n^2 + (3/8)·n + 1  →  (1/8)·n^2 + (1/2)·n | (7/8)·n - 1 | 0.0625/n | read A[i6, i7] (i0=0, i7=0) |
| n^2 | 0.213 | ramp | (13/8)·n + 1  →  (1/8)·n^2 + (3/8)·n + 3 | (7/8)·n - 1 | 0.0625/n | read A[i4, i3] (i0=0, i3=0) |
| n^2 | 0.183 | ramp | (13/8)·n - 1  →  (1/8)·n^2 + (1/4)·n + 2 | (3/4)·n | 0.0536/n | read A[i6, i7] (i0=0) |
| n^2 | 0.0597 | ramp | (19/8)·n - 7  →  (1/8)·n^2 + (-1/2)·n + 8 | (1/4)·n - 2 | 0.0179/n | read A[i6, i7] (i0=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.00893/n | read A[i6, i7] (i0=0) |
| n^2 | 0.0433 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n | (1/8)·n - 1 | 0.00893/n | read A[i6, i7] (i0=0, i7=0) |
| n^2 | 0.0423 | ramp | (1/8)·n^2 + (5/8)·n + 2  →  (1/8)·n^2 + (3/4)·n - 1 | (1/8)·n - 2 | 0.00893/n | read A[i4, i3] (i0=0, i4=0) |
| n^2 | 0.03 | ramp | (5/2)·n - 5  →  (1/8)·n^2 + (-3/8)·n + 9 | (1/8)·n - 1 | 0.00893/n | read A[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0291 | ramp | (5/2)·n - 1  →  (1/8)·n^2 + (-5/8)·n + 2 | (1/8)·n - 2 | 0.00893/n | read A[i6, i7] (i0=0, i6=0) |
| n^2 | 0.029 | ramp | (5/2)·n - 2  →  (1/8)·n^2 + (-3/4)·n + 4 | (1/8)·n - 2 | 0.00893/n | read A[i4, i3] (i0=0) |
| n^2 | 0.0281 | ramp | (11/8)·n - 2  →  (1/8)·n^2 + (-7/4)·n + 1 | (1/8)·n - 2 | 0.00893/n | read F[i5] (i0=0) |
| n^1.5 | 1.15 | level | (3/8)·n + 3 | (15/8)·n - 1 | 0.134/n | read C[i2] (i0=0); read E[i2] (i0=0) |
| n^1.5 | 1.07 | level | (3/8)·n + 3 | (7/4)·n | 0.125/n | read C[i2] (i0=0, i2=0); read E[i2] (i0=0, i2=0) |
| n^1.5 | 0.928 | level | (9/8)·n + (15/8) | (7/8)·n + (-7/8) | 0.0625/n | read A[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.0625/n | read A[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.0625/n | read A[i4, i3] (i0=0, i4=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.0625/n | read G[i4] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.0625/n | read G[i4] (i0=0, i4=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n + 1 | 0.0625/n | read F[i7] (i0=0, i6=0); read F[i7] (i0=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n + 1 | 0.0625/n | read F[i7] (i0=0, i6=0, i7=0); read F[i7] (i0=0, i7=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.00893/n | read G[i4] (i0=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.00893/n | read G[i4] (i0=0, i4=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 4 | (1/8)·n - 1 | 0.00893/n | read C[i2] (i0=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 5 | (1/8)·n - 1 | 0.00893/n | read E[i2] (i0=0, i2=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 4 | (1/8)·n - 1 | 0.00893/n | read C[i2] (i0=0, i2=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.00893/n | read F[i7] (i0=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.00893/n | read F[i7] (i0=0, i7=0) |
| n^1 | 4.29 | level | 6 | (7/4)·n | 0.125/n | read B[i1] (i0=0, i2=0); read D[i1] (i0=0, i2=0) |
| n^1 | 2.62 | level | 1 | (21/8)·n | 0.188/n | read F[i3] (i0=0); read F[i5] (i0=0) (+1) |
| n^1 | 1.41 | level | 2 | n | 0.0714/n | read A[i4, i3] (i0=0, i3=0, i4=0); write F[i5] (i0=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n | 0.0625/n | read H[i5] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n + (7/4) | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (29/8) | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n + 1 | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 1 | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0, i6=0, i7=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n | 1 | 0.0714·n^-2 | read F[i5] (i0=0, i5=0) |
| n^0.5 | 1.22 | level | (3/2)·n | 1 | 0.0714·n^-2 | read A[i6, i7] (i0=0, i6=0) |
| n^0.5 | 1.22 | level | (3/2)·n | 1 | 0.0714·n^-2 | read A[i4, i3] (i0=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n - 1 | 1 | 0.0714·n^-2 | read F[i5] (i0=0) |

The rank-1-update matrix A is walked twice (by rows, then transposed): ramps from ~(21/8)n up to (1/8)n^2 + (3/4)n lines on both `read A[i4,i3]` and `read A[i6,i7]` give d = 3.0, headroom +1.0 single-shot, same mechanism as mvt.

## gesummv — infinite-repeat  [`exact`]

Accesses $A(n) = 8·n^2 + 5·n$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.125·n^3  +  0.0765·n^2.5  +  17.6·n^2  +  1.22·n^1.5  +  14.3·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.125 | level | (1/4)·n^2 + (3/8)·n | (1/4)·n^2 + (-9/2)·n + 8 | 0.0312 | read C[i1, i2] (i0=0); read E[i1, i2] (i0=0) |
| n^2.5 | 0.067 | level | (3/8)·n + 1 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read D[i2] (i0=0, i1=1); read D[i2] (i0=0) |
| n^2.5 | 0.00957 | level | (3/8)·n + 3 | (1/64)·n^2 + (-1/4)·n | 0.00195 | read D[i2] (i0=0, i1=0); read D[i2] (i0=0) |
| n^2 | 3.5 | level | 4 | (7/4)·n^2 + (-7/8)·n | 0.219 | read D[i2] (i0=0, i1=0); read D[i2] (i0=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 | 0.219 | write A[i1] (i0=0); write B[i1] (i0=0) |
| n^2 | 2.25 | level | 4 | (9/8)·n^2 + (-7/8)·n | 0.141 | read B[i1] (i0=0, i2=0); read A[i1] (i0=0) (+1) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 | 0.109 | read E[i1, i2] (i0=0) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 | 0.109 | read C[i1, i2] (i0=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 + (7/8)·n | 0.109 | read B[i1] (i0=0, i2=0); read B[i1] (i0=0) (+1) |
| n^2 | 1 | level | (1/4)·n^2 + (3/8)·n | 2·n - 4 | 0.25/n | read C[i1, i2] (i0=0); read E[i1, i2] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/8)·n | n - 2 | 0.125/n | read E[i1, i2] (i0=0, i2=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/8)·n | n - 2 | 0.125/n | read C[i1, i2] (i0=0, i2=0) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 | 0.0312 | write A[i1] (i0=0, i2=0); write B[i1] (i0=0, i2=0) (+2) |
| n^2 | 0.25 | level | 4 | (1/8)·n^2 + (7/8)·n | 0.0156 | read D[i2] (i0=0, i1=0); read D[i2] (i0=0) |
| n^2 | 0.125 | level | (1/4)·n^2 + (3/8)·n | (1/4)·n - 4 | 0.0312/n | read C[i1, i2] (i0=0, i1=0); read E[i1, i2] (i0=0, i1=0) |
| n^2 | 0.125 | level | (1/4)·n^2 + (17/8)·n + (21/8) | (1/4)·n + (-9/4) | 0.0312/n | read C[i1, i2] (i0=0); read E[i1, i2] (i0=0) |
| n^2 | 0.125 | level | (1/4)·n^2 + (3/8)·n | (1/4)·n - 4 | 0.0312/n | read C[i1, i2] (i0=0); read E[i1, i2] (i0=0) |
| n^2 | 0.125 | level | (1/4)·n^2 + (-13/8)·n | (1/4)·n - 4 | 0.0312/n | write A[i1] (i0=0); write B[i1] (i0=0) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.109/n | read D[i2] (i0=0, i1=1); read D[i2] (i0=0) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.109/n | read D[i2] (i0=0, i1=1, i2=0); read D[i2] (i0=0, i2=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n | 0.0156/n | read D[i2] (i0=0, i1=0); read D[i2] (i0=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n | 0.0156/n | read D[i2] (i0=0, i1=0, i2=0); read D[i2] (i0=0, i2=0) |
| n^1 | 2.83 | level | 2 | 2·n | 0.25/n | write A[i1] (i0=0); read A[i1] (i0=0, i2=0) (+1) |
| n^1 | 2.47 | level | 2 | (7/4)·n | 0.219/n | write B[i1] (i0=0); read B[i1] (i0=0) |
| n^1 | 1 | level | (1/4)·n^2 + (3/8)·n | 2 | 0.25·n^-2 | read C[i1, i2] (i0=0, i1=0); read E[i1, i2] (i0=0, i1=0) |
| n^1 | 1 | level | (1/4)·n^2 + (17/8)·n + (21/8) | 2 | 0.25·n^-2 | read C[i1, i2] (i0=0); read E[i1, i2] (i0=0) |
| n^1 | 1 | level | (1/4)·n^2 + (3/8)·n | 2 | 0.25·n^-2 | read C[i1, i2] (i0=0); read E[i1, i2] (i0=0) |
| n^1 | 1 | level | (1/4)·n^2 + (-13/8)·n | 2 | 0.25·n^-2 | write A[i1] (i0=0, i1=0); write B[i1] (i0=0, i1=0) |
| n^1 | 1 | level | (1/4)·n^2 + (-13/8)·n | 2 | 0.25·n^-2 | write A[i1] (i0=0); write B[i1] (i0=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.109/n | write B[i1] (i0=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read E[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read C[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (17/8)·n + (21/8) | 1 | 0.125·n^-2 | read E[i1, i2] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read E[i1, i2] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (17/8)·n + (21/8) | 1 | 0.125·n^-2 | read C[i1, i2] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (3/8)·n | 1 | 0.125·n^-2 | read C[i1, i2] (i0=0, i2=0) |
| n^1 | 0.125 | level | 1 | (1/8)·n | 0.0156/n | write B[i1] (i0=0) |

Both matrices return as wraparound terms: coefficient 0.125 = 2 × 0.0625 at the two-matrix footprint boundary — twice atax's jump because gesummv streams two matrices.

## gesummv — single-shot  [`exact`]

Accesses $A(n) = 8·n^2 + 5·n$ (exact on n ≡ 0 mod 8); DMD order $n^{2.5}$, headroom **+0.5**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0765·n^2.5  +  15.1·n^2  +  1.22·n^1.5  +  6.3·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^2.5 | 0.067 | level | (3/8)·n + 1 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read D[i2] (i0=0) |
| n^2.5 | 0.00957 | level | (3/8)·n + 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read D[i2] (i0=0) |
| n^2 | 6 | level | 4 | 3·n^2 | 0.375 | read D[i2] (i0=0, i2=0); read B[i1] (i0=0, i2=0) (+3) |
| n^2 | 3.91 | level | 5 | (7/4)·n^2 | 0.219 | read C[i1, i2] (i0=0); read E[i1, i2] (i0=0) |
| n^2 | 3.46 | level | 3 | 2·n^2 | 0.25 | write A[i1] (i0=0); write B[i1] (i0=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 | 0.109 | read D[i2] (i0=0) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.109/n | read D[i2] (i0=0) |
| n^1.5 | 0.536 | level | (3/8)·n + 1 | (7/8)·n | 0.109/n | read D[i2] (i0=0, i2=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n - 1 | 0.0156/n | read D[i2] (i0=0) |
| n^1.5 | 0.0765 | level | (3/8)·n + 3 | (1/8)·n - 1 | 0.0156/n | read D[i2] (i0=0, i2=0) |
| n^1 | 2.83 | level | 2 | 2·n | 0.25/n | read A[i1] (i0=0, i2=0); read B[i1] (i0=0) |
| n^1 | 2.47 | level | 2 | (7/4)·n | 0.219/n | write A[i1] (i0=0); write B[i1] (i0=0) |
| n^1 | 1 | level | 1 | n | 0.125/n | write B[i1] (i0=0) |

Two matrices are streamed once per pass; single-shot sees only vector reuse (d = 2.5).

## gramschmidt — infinite-repeat  [`exact`]

Accesses $A(n) = 3·n^3 + (1/2)·n^2 + (1/2)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.016·n^4  +  2.74·n^3.5  +  2.81·n^3  +  23.9·n^2.5  +  26.8·n^2  +  64.4·n^1.5  +  29.2·n^1  +  77.8·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0101 | ramp | 5·n + 4  →  (1/8)·n^2 + (9/8)·n | (5/128)·n^3 + (-205/128)·n^2 + (265/16)·n - 15 | 0.013 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^4 | 0.00203 | ramp | 5·n + 4  →  (1/8)·n^2 + (9/8)·n | (1/128)·n^3 + (-41/128)·n^2 + (53/16)·n - 3 | 0.0026 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^4 | 0.00193 | ramp | 6·n + 4  →  (1/8)·n^2 + (9/8)·n - 1 | (1/128)·n^3 + (-57/128)·n^2 + (103/16)·n - 6 | 0.0026 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^4 | 0.0019 | ramp | 5·n + 4  →  (1/8)·n^2 + (1/8)·n - 1 | (1/128)·n^3 + (-57/128)·n^2 + (103/16)·n - 6 | 0.0026 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3.5 | 0.795 | level | 2·n + 1 | (9/16)·n^3 + (-9/8)·n^2 + n | 0.188 | read C[i6, i1] (i0=0, i4=0); read C[i5, i1] (i0=0, i5=0) (+2) |
| n^3.5 | 0.619 | level | 2·n | (7/16)·n^3 + (-119/16)·n^2 + (287/8)·n - 35 | 0.146 | read A[i6, i4] (i0=0, i4=6, i6=0); read A[i6, i4] (i0=0) |
| n^3.5 | 0.619 | level | 2·n + 1 | (7/16)·n^3 + (-7/8)·n^2 | 0.146 | read C[i6, i1] (i0=0, i6=0); read C[i6, i1] (i0=0) |
| n^3.5 | 0.475 | level | 2·n + 1 | (43/128)·n^3 + (-547/128)·n^2 + (223/16)·n - 10 | 0.112 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3.5 | 0.0884 | level | 2·n | (1/16)·n^3 + (1/16)·n^2 - 6·n + 27 | 0.0208 | read A[i3, i1] (i0=0); read A[i6, i4] (i0=0, i4=6, i6=0) (+3) |
| n^3.5 | 0.0663 | level | 2·n + 1 | (3/64)·n^3 + (-75/64)·n^2 + (57/8)·n - 6 | 0.0156 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3.5 | 0.0663 | level | 2·n + 1 | (3/64)·n^3 + (-75/64)·n^2 + (57/8)·n - 6 | 0.0156 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3.5 | 0.011 | level | 2·n + 1 | (1/128)·n^3 + (-25/128)·n^2 + (19/16)·n - 1 | 0.0026 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 1 | level | 4 | (1/2)·n^3 - 2·n^2 + (5/2)·n - 1 | 0.167 | read B[i1, i4] (i0=0) |
| n^3 | 0.663 | level | 3 | (49/128)·n^3 + (-105/16)·n^2 + 28·n | 0.128 | write A[i6, i4] (i0=0) |
| n^3 | 0.213 | ramp | 3·n + 2  →  (1/8)·n^2 + (9/8)·n | (7/8)·n^2 + (-71/8)·n + 8 | 0.292/n | read A[i2, i1] (i0=0, i2=0); read A[i2, i1] (i0=0) |
| n^3 | 0.152 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (5/8)·n^2 + (-85/8)·n + 10 | 0.208/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 0.152 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (5/8)·n^2 + (-85/8)·n + 10 | 0.208/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-21/16)·n^2 + 7·n | 0.0182 | write A[i6, i4] (i0=0) |
| n^3 | 0.0947 | level | 3 | (7/128)·n^3 + (-7/16)·n^2 | 0.0182 | write A[i6, i4] (i0=0) |
| n^3 | 0.0645 | ramp | (5/16)·n^2 + (-1/2)·n + 10  →  (5/16)·n^2 + (1/2)·n - 22 | (1/8)·n^2 - 3·n | 0.0417/n | write C[i3, i1] (i0=0, i3=0); write C[i3, i1] (i0=0) |
| n^3 | 0.0514 | ramp | (1/8)·n^2 + (33/8)·n + 23  →  (5/16)·n^2 + (-27/8)·n + 23 | (1/8)·n^2 - 4·n | 0.0417/n | read A[i5, i4] (i0=0, i1=0, i5=0); read A[i5, i4] (i0=0, i1=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (9/8)·n | (1/8)·n^2 + (-25/8)·n + 3 | 0.0417/n | read A[i5, i4] (i0=0, i1=1, i5=0); read A[i5, i4] (i0=0, i1=1) |
| n^3 | 0.0303 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.0417/n | read A[i5, i4] (i0=0, i4=0, i5=0); read A[i5, i4] (i0=0, i4=0) |
| n^3 | 0.0303 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.0417/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 0.03 | ramp | 5·n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (1/8)·n^2 + (-25/8)·n + 3 | 0.0417/n | read A[i5, i4] (i0=0, i4=7, i5=0); read A[i5, i4] (i0=0, i4=7) |
| n^3 | 0.0295 | ramp | 3·n + 2  →  (1/8)·n^2 + (1/8)·n - 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0417/n | read A[i2, i1] (i0=0, i2=0); read A[i2, i1] (i0=0) |
| n^3 | 0.0295 | ramp | 4·n + 4  →  (1/8)·n^2 + (9/8)·n - 3 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0417/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0292 | ramp | 4·n + 3  →  (1/8)·n^2 + (1/8)·n - 1 | (1/8)·n^2 + (-25/8)·n + 3 | 0.0417/n | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^3 | 0.0292 | ramp | 4·n + 3  →  (1/8)·n^2 + (1/8)·n - 1 | (1/8)·n^2 + (-25/8)·n + 3 | 0.0417/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 0.0262 | level | (5/16)·n^2 + (1/2)·n | (3/64)·n^2 + (-15/8)·n + 18 | 0.0156/n | write B[i1, i4] (i0=0) |
| n^3 | 0.0135 | level | 3 | (1/128)·n^3 + (-3/16)·n^2 + 8·n | 0.0026 | write C[i3, i1] (i0=0, i1=0, i3=0); write A[i6, i4] (i0=0, i1=0) (+1) |
| n^3 | 0.0122 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (3/64)·n^2 + (-15/8)·n + 18 | 0.0156/n | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0) |
| n^3 | 0.00437 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/128)·n^2 + (-13/64)·n + (153/128) | 0.0026/n | write B[i1, i4] (i0=0) |
| n^3 | 0.00437 | level | (5/16)·n^2 + (1/2)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.0026/n | write B[i1, i4] (i0=0) |
| n^3 | 0.00437 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/128)·n^2 + (-21/64)·n + (425/128) | 0.0026/n | write B[i1, i4] (i0=0) |
| n^3 | 0.00437 | level | (5/16)·n^2 + (1/2)·n | (1/128)·n^2 + (-7/16)·n + 6 | 0.0026/n | write B[i1, i4] (i0=0) |
| n^3 | 0.00204 | ramp | 5·n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0026/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00194 | ramp | 6·n + 3  →  (1/8)·n^2 + (9/8)·n - 2 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0026/n | read A[i5, i4] (i0=0) |
| n^2.5 | 7.42 | level | 2·n | (21/4)·n^2 + (-105/4)·n + 21 | 1.75/n | read A[i6, i4] (i0=0) |
| n^2.5 | 5.3 | level | 2·n + 1 | (15/4)·n^2 + (-75/4)·n + 15 | 1.25/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^2.5 | 1.41 | level | 2·n - 1 | n^2 - 2·n + 1 | 0.333/n | read C[i5, i1] (i0=0, i4=0, i5=0); read C[i5, i1] (i0=0, i4=0) |
| n^2.5 | 1.34 | ramp | 2·n + 4  →  3·n + 1 | (7/8)·n^2 + (-35/4)·n + 14 | 0.292/n | write C[i3, i1] (i0=0) |
| n^2.5 | 1.24 | level | 2·n | (7/8)·n^2 + (-63/8)·n + 7 | 0.292/n | read A[i6, i4] (i0=0, i4=6) |
| n^2.5 | 1.24 | level | 2·n | (7/8)·n^2 + (-15/8)·n + 1 | 0.292/n | read A[i5, i4] (i0=0, i4=0, i5=0); read A[i5, i4] (i0=0, i4=0) |
| n^2.5 | 1.24 | level | 2·n + 1 | (7/8)·n^2 + (-63/8)·n + 7 | 0.292/n | read A[i5, i4] (i0=0, i1=0, i5=0); read A[i5, i4] (i0=0, i1=0) |
| n^2.5 | 1.21 | ramp | n + 2  →  2·n - 1 | n^2 - 3·n + 2 | 0.333/n | read A[i3, i1] (i0=0) |
| n^2.5 | 1.06 | level | 2·n + 1 | (3/4)·n^2 + (-27/4)·n + 6 | 0.25/n | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 1.06 | level | 2·n | (3/4)·n^2 + (21/4)·n - 21 | 0.25/n | read A[i6, i4] (i0=0, i6=0); read A[i6, i4] (i0=0) |
| n^2.5 | 0.541 | level | 2·n | (49/128)·n^2 + (-79/16)·n + 15 | 0.128/n | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0) |
| n^2.5 | 0.541 | level | 2·n + 1 | (49/128)·n^2 + (-7/16)·n | 0.128/n | write B[i1, i4] (i0=0) |
| n^2.5 | 0.177 | level | 2·n + 1 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0417/n | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 0.0773 | level | 2·n | (7/128)·n^2 + (-21/16)·n + 7 | 0.0182/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.0773 | level | 2·n + 1 | (7/128)·n^2 + (-7/16)·n | 0.0182/n | write B[i1, i4] (i0=0) |
| n^2 | 7.58 | level | 3 | (35/8)·n^2 - 20·n | 1.46/n | write A[i6, i4] (i0=0) |
| n^2 | 2 | level | 4 | n^2 - 2·n + 1 | 0.333/n | read B[i1, i4] (i0=0, i4=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 - 7·n | 0.292/n | write A[i6, i4] (i0=0, i4=6); read C[i6, i1] (i0=0, i6=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 - 7·n | 0.292/n | write A[i6, i4] (i0=0, i1=0) |
| n^2 | 1.3 | level | 3 | (3/4)·n^2 | 0.25/n | write A[i6, i4] (i0=0, i4=0) |
| n^2 | 1.08 | level | 3 | (5/8)·n^2 - 5·n | 0.208/n | write A[i6, i4] (i0=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-7/8)·n | 0.146/n | read B[i1, i4] (i0=0, i6=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (9/4)·n + (-121/16) | n | 0.333·n^-2 | read A[i2, i1] (i0=0, i1=0, i2=0); read A[i2, i1] (i0=0, i1=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (-3/8)·n - 1 | n | 0.333·n^-2 | read A[i2, i1] (i0=0, i1=0, i2=0); read A[i2, i1] (i0=0, i1=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (5/4)·n + (-105/16) | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i1=0, i4=7, i5=0); read A[i5, i4] (i0=0, i1=0, i4=7) |
| n^2 | 0.559 | level | (5/16)·n^2 + (-11/8)·n + 7 | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i1=0, i4=7, i5=0); read A[i5, i4] (i0=0, i1=0, i4=7) |
| n^2 | 0.559 | level | (5/16)·n^2 + (1/2)·n - 14 | n | 0.333·n^-2 | write C[i3, i1] (i0=0, i3=0); write C[i3, i1] (i0=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (17/8)·n + (-87/16) | n | 0.333·n^-2 | write C[i3, i1] (i0=0, i1=0, i3=0); write C[i3, i1] (i0=0, i1=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (-1/2)·n + 2 | n | 0.333·n^-2 | write C[i3, i1] (i0=0, i1=0, i3=0); write C[i3, i1] (i0=0, i1=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-231/16) | n | 0.333·n^-2 | write C[i3, i1] (i0=0, i3=0); write C[i3, i1] (i0=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (1/2)·n - 7 | n | 0.333·n^-2 | write C[i3, i1] (i0=0, i3=0); write C[i3, i1] (i0=0) |
| n^2 | 0.489 | level | (5/16)·n^2 + (1/2)·n | (7/8)·n - 14 | 0.292·n^-2 | write B[i1, i1] (i0=0) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 - 2·n | 0.0833/n | write A[i6, i4] (i0=0, i4=0) |
| n^2 | 0.419 | level | (5/16)·n^2 + (1/2)·n | (3/4)·n - 12 | 0.25·n^-2 | write B[i1, i4] (i0=0) |
| n^2 | 0.419 | level | (5/16)·n^2 + (1/2)·n | (3/4)·n - 12 | 0.25·n^-2 | write B[i1, i4] (i0=0) |
| n^2 | 0.359 | ramp | (1/8)·n^2 + (9/8)·n + 1  →  (1/8)·n^2 + (17/8)·n - 2 | n - 2 | 0.333·n^-2 | read A[i5, i4] (i0=0, i1=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (9/8)·n | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i1=1, i4=6, i5=0); read A[i5, i4] (i0=0, i1=1, i4=6) |
| n^2 | 0.354 | level | (1/8)·n^2 + (9/8)·n | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i1=1, i5=0); read A[i5, i4] (i0=0, i1=1) |
| n^2 | 0.354 | level | (1/8)·n^2 + (25/8)·n + 7 | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i1=0, i5=0); read A[i5, i4] (i0=0, i1=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + 4·n + (7/8) | n | 0.333·n^-2 | read A[i5, i4] (i0=0, i1=0, i5=0); read A[i5, i4] (i0=0, i1=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (9/8)·n | n - 1 | 0.333·n^-2 | read A[i2, i1] (i0=0, i1=1, i2=0); read A[i2, i1] (i0=0, i1=1) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 - n | 0.0417/n | read C[i6, i1] (i0=0, i4=0); write A[i6, i4] (i0=0, i4=6) (+2) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 - n | 0.0417/n | write A[i6, i4] (i0=0, i1=0) |
| n^2 | 0.214 | ramp | 3·n + 1  →  (1/8)·n^2 + (9/8)·n - 1 | (7/8)·n - 8 | 0.292·n^-2 | read A[i2, i1] (i0=0) |
| n^2 | 0.182 | ramp | 3·n + 1  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n - 12 | 0.25·n^-2 | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0, i4=7) (+1) |
| n^2 | 0.152 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (5/8)·n - 10 | 0.208·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0208/n | read B[i1, i4] (i0=0, i6=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/8)·n + (-17/8) | 0.0417·n^-2 | write B[i1, i4] (i0=0, i1=7) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 3 | 0.0417·n^-2 | write B[i1, i4] (i0=0, i1=7) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/8)·n + (-9/8) | 0.0417·n^-2 | write B[i1, i4] (i0=0, i1=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.0417·n^-2 | write B[i1, i4] (i0=0, i1=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/8)·n + (-17/8) | 0.0417·n^-2 | write B[i1, i4] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.0417·n^-2 | write B[i1, i4] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/8)·n + (-17/8) | 0.0417·n^-2 | write B[i1, i4] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 3 | 0.0417·n^-2 | write B[i1, i4] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/8)·n + (-17/8) | 0.0417·n^-2 | write B[i1, i1] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 3 | 0.0417·n^-2 | write B[i1, i1] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | (1/8)·n + (-17/8) | 0.0417·n^-2 | write B[i1, i4] (i0=0, i4=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 3 | 0.0417·n^-2 | write B[i1, i4] (i0=0, i4=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (9/8)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i1=1) |
| n^2 | 0.0304 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0304 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0301 | ramp | 5·n + 2  →  (1/8)·n^2 + (9/8)·n - 2 | (1/8)·n - 3 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0301 | ramp | 5·n + 2  →  (1/8)·n^2 + (9/8)·n - 2 | (1/8)·n - 3 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^2 | 0.0296 | ramp | 3·n + 1  →  (1/8)·n^2 + (1/8)·n - 2 | (1/8)·n - 2 | 0.0417·n^-2 | read A[i2, i1] (i0=0) |
| n^2 | 0.0293 | ramp | 4·n + 2  →  (1/8)·n^2 + (1/8)·n - 2 | (1/8)·n - 3 | 0.0417·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 8.66 | level | 3·n + 2 | 5·n - 5 | 1.67·n^-2 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^1.5 | 8.49 | level | 2·n + 1 | 6·n - 5 | 2·n^-2 | write B[i1, i1] (i0=0, i1=0); write C[i3, i1] (i0=0, i1=0) (+3) |
| n^1.5 | 7.27 | ramp | n + 2  →  2·n - 1 | 6·n - 12 | 2·n^-2 | read A[i2, i1] (i0=0) |
| n^1.5 | 7.07 | level | 2·n + 2 | 5·n - 10 | 1.67·n^-2 | write C[i3, i1] (i0=0) |
| n^1.5 | 5.3 | level | 2·n | (15/4)·n - 15 | 1.25·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 2.83 | level | 2·n + 2 | 2·n | 0.667·n^-2 | write C[i3, i1] (i0=0, i3=0); write C[i3, i1] (i0=0) |
| n^1.5 | 1.86 | ramp | 3·n + 3  →  4·n | n - 2 | 0.333·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^1.5 | 1.73 | level | 3·n + 2 | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 1.73 | level | 3·n + 2 | n - 1 | 0.333·n^-2 | read A[i2, i1] (i0=0, i2=0); read A[i2, i1] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 2 | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0); read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 1.52 | level | 3·n + 2 | (7/8)·n + (-63/8) | 0.292·n^-2 | write C[i3, i1] (i0=0, i3=0) |
| n^1.5 | 1.52 | level | 3·n + 2 | (7/8)·n - 7 | 0.292·n^-2 | write C[i3, i1] (i0=0, i3=0) |
| n^1.5 | 1.41 | level | 2·n | n | 0.333·n^-2 | read A[i2, i1] (i0=0, i1=0); read A[i3, i1] (i0=0, i1=0) (+3) |
| n^1.5 | 1.24 | level | 2·n - 1 | (7/8)·n - 1 | 0.292·n^-2 | read C[i5, i1] (i0=0, i4=0) |
| n^1.5 | 1.24 | level | 2·n | (7/8)·n - 7 | 0.292·n^-2 | read A[i5, i4] (i0=0, i1=0) |
| n^1.5 | 1.24 | level | 2·n + 3 | (7/8)·n + (-63/8) | 0.292·n^-2 | write C[i3, i1] (i0=0) |
| n^1.5 | 1.24 | level | 2·n + 3 | (7/8)·n - 7 | 0.292·n^-2 | write C[i3, i1] (i0=0) |
| n^1.5 | 1.24 | level | 2·n + 1 | (7/8)·n - 8 | 0.292·n^-2 | write B[i1, i4] (i0=0, i4=0) |
| n^1.5 | 1.21 | ramp | n + 2  →  2·n - 1 | n - 2 | 0.333·n^-2 | read A[i2, i1] (i0=0) |
| n^1.5 | 1.21 | ramp | n + 2  →  2·n - 1 | n - 2 | 0.333·n^-2 | read A[i2, i1] (i0=0) |
| n^1.5 | 1.21 | ramp | n + 2  →  2·n - 1 | n - 2 | 0.333·n^-2 | read A[i3, i1] (i0=0, i1=0) |
| n^1.5 | 1.06 | level | 2·n | (3/4)·n | 0.25·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 1 | level | n + 1 | n - 1 | 0.333·n^-2 | read A[i3, i1] (i0=0, i3=0) |
| n^1.5 | 0.884 | level | 2·n | (5/8)·n - 5 | 0.208·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.177 | level | 2·n | (1/8)·n - 1 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 0.177 | level | 2·n | (1/8)·n - 1 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.177 | level | 2·n - 1 | (1/8)·n | 0.0417·n^-2 | read C[i5, i1] (i0=0, i4=0) |
| n^1 | 3.35 | level | (5/16)·n^2 + (1/2)·n | 6 | 2·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 3.35 | level | (5/16)·n^2 + (1/2)·n | 6 | 2·n^-3 | write B[i1, i4] (i0=0) |
| n^1 | 2.8 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 5 | 1.67·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 2.8 | level | (5/16)·n^2 + (1/2)·n | 5 | 1.67·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 1.68 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 3 | 1·n^-3 | write B[i1, i1] (i0=0, i1=0); write B[i1, i1] (i0=0, i1=6) (+1) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.292·n^-2 | read B[i1, i4] (i0=0, i4=0, i6=0) |
| n^1 | 1.12 | level | (5/16)·n^2 + (1/2)·n | 2 | 0.667·n^-3 | write B[i1, i1] (i0=0, i1=0); write B[i1, i1] (i0=0, i1=6) |
| n^1 | 0.559 | level | (5/16)·n^2 + (-11/8)·n + 7 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i1=0, i4=7) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i4=7) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i1=7) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i1=7) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (11/4)·n + (-3/4) | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0, i1=7) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0, i1=7) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (11/4)·n + (-3/4) | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i1=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i1=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i1] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (25/8)·n + (-7/16) | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i1=7, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i1=7, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n - 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i1=1) |
| n^1 | 0.354 | level | (1/8)·n^2 + (25/8)·n + 7 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (17/8)·n - 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i1=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n - 1 | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0, i1=1) |
| n^1 | 0.217 | level | 3 | (1/8)·n - 1 | 0.0417·n^-2 | read B[i1, i4] (i0=0, i4=0, i6=0) |
| n^0.5 | 9.9 | level | 2·n | 7 | 2.33·n^-3 | read A[i5, i4] (i0=0, i1=0, i4=0); read A[i5, i4] (i0=0, i1=0) |
| n^0.5 | 8.66 | level | 3·n + 1 | 5 | 1.67·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 8.49 | level | 2·n | 6 | 2·n^-3 | read A[i2, i1] (i0=0, i2=0) |
| n^0.5 | 8.49 | level | 2·n + 1 | 6 | 2·n^-3 | write B[i1, i4] (i0=0, i4=0) |
| n^0.5 | 7.07 | level | 2·n + 2 | 5 | 1.67·n^-3 | write C[i3, i1] (i0=0) |
| n^0.5 | 7.07 | level | 2·n + 2 | 5 | 1.67·n^-3 | write C[i3, i1] (i0=0, i3=0) |
| n^0.5 | 6 | level | n | 6 | 2·n^-3 | read A[i2, i1] (i0=0) |
| n^0.5 | 2 | level | 4·n + 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i4=7, i5=0) |
| n^0.5 | 1.73 | level | 3·n + 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0, i4=7) (+1) |
| n^0.5 | 1.73 | level | 3·n + 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.73 | level | 3·n + 2 | 1 | 0.333·n^-3 | write C[i3, i1] (i0=0, i3=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0, i2=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0, i2=0) |
| n^0.5 | 1.41 | level | 2·n + 3 | 1 | 0.333·n^-3 | write C[i3, i1] (i0=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0, i2=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0, i2=0) |
| n^0.5 | 1.41 | level | 2·n + 1 | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i4=0) |
| n^0.5 | 1.41 | level | 2·n + 1 | 1 | 0.333·n^-3 | write B[i1, i4] (i0=0, i4=0) |
| n^0.5 | 1 | level | n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0) |
| n^0.5 | 1 | level | n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0) |
| n^0.5 | 1 | level | n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0) |
| n^0.5 | 1 | level | n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0) |
| n^0.5 | 1 | level | n + 1 | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0, i1=0); read A[i3, i1] (i0=0, i1=0) |

Column orthogonalization re-reads the factor panel: `read A[i5,i4]` ramps from ~5n to (1/8)n^2 + (9/8)n lines (d = 4.0, headroom +1.0 — corrected from the old +0.5, whose fits ran over corrupted terms). The large n^3.5 coefficient (2.74, `read C[i5,i1]` at 2n + 1 lines) means the pre-asymptotic behavior is dominated by column-pair reuse until the panel term takes over.

## gramschmidt — single-shot  [`exact`]

Accesses $A(n) = 3·n^3 + (1/2)·n^2 + (1/2)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0161·n^4  +  2.74·n^3.5  +  2.61·n^3  +  22.4·n^2.5  +  12·n^2  +  50.7·n^1.5  +  52.1·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0122 | ramp | 5·n + 4  →  (1/8)·n^2 + (9/8)·n | (3/64)·n^3 + (-123/64)·n^2 + (159/8)·n - 18 | 0.0156 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^4 | 0.00203 | ramp | 5·n + 4  →  (1/8)·n^2 + (9/8)·n | (1/128)·n^3 + (-41/128)·n^2 + (53/16)·n - 3 | 0.0026 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^4 | 0.00193 | ramp | 6·n + 4  →  (1/8)·n^2 + (9/8)·n - 1 | (1/128)·n^3 + (-57/128)·n^2 + (103/16)·n - 6 | 0.0026 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3.5 | 1.41 | level | 2·n + 1 | n^3 - 2·n^2 + n | 0.333 | read C[i5, i1] (i0=0, i5=0); read C[i5, i1] (i0=0) (+1) |
| n^3.5 | 0.619 | level | 2·n | (7/16)·n^3 + (-105/16)·n^2 + (217/8)·n - 21 | 0.146 | read A[i6, i4] (i0=0, i4=6); read A[i6, i4] (i0=0) |
| n^3.5 | 0.541 | level | 2·n + 1 | (49/128)·n^3 + (-777/128)·n^2 + (427/16)·n - 21 | 0.128 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3.5 | 0.0884 | level | 2·n | (1/16)·n^3 + (1/16)·n^2 + (-49/8)·n + 21 | 0.0208 | read A[i6, i4] (i0=0, i4=6, i6=0); read A[i6, i4] (i0=0, i4=6) (+2) |
| n^3.5 | 0.0773 | level | 2·n + 1 | (7/128)·n^3 + (-63/128)·n^2 + (7/16)·n | 0.0182 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 1 | level | 4 | (1/2)·n^3 - n^2 + (1/2)·n | 0.167 | read B[i1, i4] (i0=0) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (-49/8)·n^2 + 21·n | 0.146 | write A[i6, i4] (i0=0, i4=6); write A[i6, i4] (i0=0) |
| n^3 | 0.244 | ramp | 3·n + 2  →  (1/8)·n^2 + (9/8)·n | n^2 - 11·n + 10 | 0.333/n | read A[i2, i1] (i0=0, i2=0); read A[i2, i1] (i0=0) |
| n^3 | 0.182 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (3/4)·n^2 + (-51/4)·n + 12 | 0.25/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 0.152 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (5/8)·n^2 + (-85/8)·n + 10 | 0.208/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 0.108 | level | 3 | (1/16)·n^3 + (-3/8)·n^2 | 0.0208 | write A[i6, i4] (i0=0, i4=6); write A[i6, i4] (i0=0) |
| n^3 | 0.0303 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.0417/n | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^3 | 0.0303 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.0417/n | read A[i5, i4] (i0=0, i4=0, i5=0); read A[i5, i4] (i0=0, i4=0) |
| n^3 | 0.0303 | ramp | 4·n + 3  →  (1/8)·n^2 + (9/8)·n | (1/8)·n^2 + (-17/8)·n + 2 | 0.0417/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^3 | 0.03 | ramp | 5·n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (1/8)·n^2 + (-25/8)·n + 3 | 0.0417/n | read A[i5, i4] (i0=0, i4=7, i5=0); read A[i5, i4] (i0=0, i4=7) |
| n^3 | 0.0295 | ramp | 4·n + 4  →  (1/8)·n^2 + (9/8)·n - 3 | (1/8)·n^2 + (-13/4)·n + 6 | 0.0417/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0123 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (3/64)·n^2 + (-7/4)·n + 16 | 0.0156/n | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0) |
| n^3 | 0.00204 | ramp | 5·n + 3  →  (1/8)·n^2 + (9/8)·n - 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.0026/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00194 | ramp | 6·n + 3  →  (1/8)·n^2 + (9/8)·n - 2 | (1/128)·n^2 + (-7/16)·n + 6 | 0.0026/n | read A[i5, i4] (i0=0) |
| n^2.5 | 7.42 | level | 2·n | (21/4)·n^2 + (-105/4)·n + 21 | 1.75/n | read A[i6, i4] (i0=0) |
| n^2.5 | 5.3 | level | 2·n + 1 | (15/4)·n^2 + (-75/4)·n + 15 | 1.25/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^2.5 | 1.41 | level | 2·n - 1 | n^2 - 2·n + 1 | 0.333/n | read C[i5, i1] (i0=0, i4=0, i5=0); read C[i5, i1] (i0=0, i4=0) |
| n^2.5 | 1.34 | ramp | 2·n + 4  →  3·n + 1 | (7/8)·n^2 + (-35/4)·n + 14 | 0.292/n | write C[i3, i1] (i0=0) |
| n^2.5 | 1.24 | level | 2·n | (7/8)·n^2 + (-7/8)·n | 0.292/n | read A[i5, i4] (i0=0, i4=0, i5=0); read A[i5, i4] (i0=0, i4=0) |
| n^2.5 | 1.21 | ramp | n + 2  →  2·n - 1 | n^2 - 2·n | 0.333/n | read A[i3, i1] (i0=0) |
| n^2.5 | 1.06 | level | 2·n + 1 | (3/4)·n^2 + (-27/4)·n + 6 | 0.25/n | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 1.06 | level | 2·n | (3/4)·n^2 + (21/4)·n - 21 | 0.25/n | read A[i6, i4] (i0=0, i6=0); read A[i6, i4] (i0=0) |
| n^2.5 | 0.884 | level | 2·n + 1 | (5/8)·n^2 + (-5/8)·n | 0.208/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^2.5 | 0.619 | level | 2·n + 1 | (7/16)·n^2 + (-7/8)·n | 0.146/n | write B[i1, i4] (i0=0) |
| n^2.5 | 0.541 | level | 2·n | (49/128)·n^2 + (-79/16)·n + 15 | 0.128/n | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0) |
| n^2.5 | 0.177 | level | 2·n + 1 | (1/8)·n^2 + (-1/8)·n | 0.0417/n | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 0.0773 | level | 2·n | (7/128)·n^2 + (-7/16)·n | 0.0182/n | read A[i5, i4] (i0=0) |
| n^2 | 9.09 | level | 3 | (21/4)·n^2 - 21·n | 1.75/n | write A[i6, i4] (i0=0) |
| n^2 | 1.3 | level | 3 | (3/4)·n^2 | 0.25/n | write A[i6, i4] (i0=0) |
| n^2 | 0.866 | level | 3 | (1/2)·n^2 + (-1/2)·n | 0.167/n | read C[i5, i1] (i0=0, i5=0); read C[i6, i1] (i0=0) (+1) |
| n^2 | 0.245 | ramp | 3·n + 1  →  (1/8)·n^2 + (9/8)·n - 1 | n - 10 | 0.333·n^-2 | read A[i2, i1] (i0=0) |
| n^2 | 0.183 | ramp | 3·n + 1  →  (1/8)·n^2 + (9/8)·n - 1 | (3/4)·n - 11 | 0.25·n^-2 | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0, i4=7) (+1) |
| n^2 | 0.152 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (5/8)·n - 10 | 0.208·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0304 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0304 | ramp | 4·n + 2  →  (1/8)·n^2 + (9/8)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0301 | ramp | 5·n + 2  →  (1/8)·n^2 + (9/8)·n - 2 | (1/8)·n - 3 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0301 | ramp | 5·n + 2  →  (1/8)·n^2 + (9/8)·n - 2 | (1/8)·n - 3 | 0.0417·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^2 | 0.0293 | ramp | 4·n + 2  →  (1/8)·n^2 + (1/8)·n - 2 | (1/8)·n - 3 | 0.0417·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 8.66 | level | 3·n + 2 | 5·n - 5 | 1.67·n^-2 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^1.5 | 8.49 | level | 2·n + 2 | 6·n - 12 | 2·n^-2 | write C[i3, i1] (i0=0) |
| n^1.5 | 8.48 | ramp | n + 2  →  2·n - 1 | 7·n - 14 | 2.33·n^-2 | read A[i2, i1] (i0=0) |
| n^1.5 | 5.3 | level | 2·n | (15/4)·n - 15 | 1.25·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 1.86 | ramp | 3·n + 3  →  4·n | n - 2 | 0.333·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^1.5 | 1.73 | level | 3·n + 2 | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 1.73 | level | 3·n + 2 | n - 1 | 0.333·n^-2 | read A[i2, i1] (i0=0, i2=0); read A[i2, i1] (i0=0) |
| n^1.5 | 1.73 | level | 3·n + 2 | n - 1 | 0.333·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0); read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 1.52 | level | 3·n + 2 | (7/8)·n - 7 | 0.292·n^-2 | write C[i3, i1] (i0=0, i3=0) |
| n^1.5 | 1.41 | level | 2·n - 1 | n - 1 | 0.333·n^-2 | read C[i5, i1] (i0=0, i4=0) |
| n^1.5 | 1.41 | level | 2·n + 2 | n | 0.333·n^-2 | write C[i3, i1] (i0=0, i3=0); write C[i3, i1] (i0=0) |
| n^1.5 | 1.41 | level | 2·n | n | 0.333·n^-2 | read A[i3, i1] (i0=0) |
| n^1.5 | 1.24 | level | 2·n | (7/8)·n | 0.292·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 1.24 | level | 2·n + 3 | (7/8)·n - 7 | 0.292·n^-2 | write C[i3, i1] (i0=0) |
| n^1.5 | 1.24 | level | 2·n + 1 | (7/8)·n | 0.292·n^-2 | write B[i1, i4] (i0=0, i4=0) |
| n^1.5 | 1.21 | ramp | n + 2  →  2·n - 1 | n - 2 | 0.333·n^-2 | read A[i2, i1] (i0=0) |
| n^1.5 | 1 | level | n + 1 | n | 0.333·n^-2 | read A[i3, i1] (i0=0, i3=0) |
| n^1.5 | 0.884 | level | 2·n | (5/8)·n | 0.208·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.177 | level | 2·n | (1/8)·n | 0.0417·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^0.5 | 9.9 | level | 2·n | 7 | 2.33·n^-3 | read A[i2, i1] (i0=0, i2=0) |
| n^0.5 | 8.66 | level | 3·n + 1 | 5 | 1.67·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 8.49 | level | 2·n + 2 | 6 | 2·n^-3 | write C[i3, i1] (i0=0) |
| n^0.5 | 8.49 | level | 2·n + 2 | 6 | 2·n^-3 | write C[i3, i1] (i0=0, i3=0) |
| n^0.5 | 7 | level | n | 7 | 2.33·n^-3 | read A[i2, i1] (i0=0) |
| n^0.5 | 2 | level | 4·n + 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i4=7, i5=0) |
| n^0.5 | 1.73 | level | 3·n + 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i4=6); read A[i5, i4] (i0=0, i4=7) (+1) |
| n^0.5 | 1.73 | level | 3·n + 1 | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0) |
| n^0.5 | 1.73 | level | 3·n + 1 | 1 | 0.333·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0, i2=0) |
| n^0.5 | 1 | level | n | 1 | 0.333·n^-3 | read A[i2, i1] (i0=0) |

Column orthogonalization re-reads the factor panel: `read A[i5,i4]` ramps from ~5n to (1/8)n^2 + (9/8)n lines (d = 4.0, headroom +1.0 — corrected from the old +0.5, whose fits ran over corrupted terms). The large n^3.5 coefficient (2.74, `read C[i5,i1]` at 2n + 1 lines) means the pre-asymptotic behavior is dominated by column-pair reuse until the panel term takes over.

## heat-3d — single-shot  [`exact`]

Accesses $A(n) = 16·n^4 - 64·n^3 + 96·n^2 - 64·n + 16$ (exact on n ≡ 0 mod 56); DMD order $n^{5.5}$, headroom **+1.5**; conservation Σmass/warm = 1 at n=1792, 1 at n=1848.

**DMD spectrum:**  0.75·n^5.5  +  1.06·n^5  +  17.7·n^4.5  +  49.9·n^4  +  72.7·n^3.5  +  36.8·n^3  +  71.5·n^2.5  +  1.41·n^2  +  0.5·n^1.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^5.5 | 0.125 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-7/2) | (1/4)·n^4 + (-51/8)·n^3 + (167/4)·n^2 + (-801/8)·n + (153/2) | 0.0156 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^5.5 | 0.125 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 2 | (1/4)·n^4 + (-49/8)·n^3 + (317/8)·n^2 + (-189/2)·n + 72 | 0.0156 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^5.5 | 0.125 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 3 | (1/4)·n^4 + (-37/8)·n^3 + (215/8)·n^2 + (-243/4)·n + 45 | 0.0156 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^5.5 | 0.125 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-9/2) | (1/4)·n^4 + (-51/8)·n^3 + (167/4)·n^2 + (-801/8)·n + (153/2) | 0.0156 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^5.5 | 0.125 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 1 | (1/4)·n^4 + (-49/8)·n^3 + (317/8)·n^2 + (-189/2)·n + 72 | 0.0156 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^5.5 | 0.125 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 4 | (1/4)·n^4 + (-37/8)·n^3 + (215/8)·n^2 + (-243/4)·n + 45 | 0.0156 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^5 | 0.177 | level | (1/2)·n^2 + 3·n + (-5/2) | (1/4)·n^4 - 6·n^3 + (133/4)·n^2 + (-123/2)·n + 34 | 0.0156 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^5 | 0.177 | level | (1/2)·n^2 + (1/2)·n - 1 | (1/4)·n^4 + (-23/4)·n^3 + (63/2)·n^2 - 58·n + 32 | 0.0156 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^5 | 0.177 | level | (1/2)·n^2 + (5/2)·n - 2 | (1/4)·n^4 + (-17/4)·n^3 + 21·n^2 - 37·n + 20 | 0.0156 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^5 | 0.177 | level | (1/2)·n^2 + 3·n + (-9/2) | (1/4)·n^4 - 6·n^3 + (133/4)·n^2 + (-123/2)·n + 34 | 0.0156 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^5 | 0.177 | level | (1/2)·n^2 + (1/2)·n - 3 | (1/4)·n^4 + (-23/4)·n^3 + (63/2)·n^2 - 58·n + 32 | 0.0156 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^5 | 0.177 | level | (1/2)·n^2 + (5/2)·n - 4 | (1/4)·n^4 + (-17/4)·n^3 + 21·n^2 - 37·n + 20 | 0.0156 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-7/2) | 2·n^3 - 17·n^2 + 47·n - 39 | 0.125/n | write B[i2, i3, i4] (i0=0, i3=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+1) |
| n^4.5 | 1 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 2 | 2·n^3 - 17·n^2 + 47·n - 39 | 0.125/n | write B[i2, i3, i4] (i0=0, i3=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+1) |
| n^4.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 3 | 2·n^3 - 17·n^2 + 47·n - 39 | 0.125/n | write B[i2, i3, i4] (i0=0, i3=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+1) |
| n^4.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-9/2) | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | read A[i2 + 1, i3, i4] (i0=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 1 | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | read A[i2 + 1, i3, i4] (i0=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 4 | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | read A[i2 + 1, i3, i4] (i0=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-7/2) | 2·n^3 - 17·n^2 + 47·n - 39 | 0.125/n | write B[i2, i3, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0) (+1) |
| n^4.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-9/2) | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-7/2) | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 2 | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-9/2) | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + n^2 + (-7/2)·n | 2·n^3 - 17·n^2 + 45·n - 36 | 0.125/n | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 1 | level | (1/4)·n^3 + (5/2)·n^2 + (-19/4)·n + 2 | 2·n^3 - 8·n^2 + 10·n - 4 | 0.125/n | read A[i2, i3, i4 + 1] (i0=0); read B[i5, i6, i7 + 1] (i0=0, i5=0, i6=0) (+3) |
| n^4.5 | 0.438 | level | (1/4)·n^3 + (5/2)·n^2 + (-19/4)·n + 2 | (7/8)·n^3 + (-135/8)·n^2 + (193/4)·n - 36 | 0.0547/n | read A[i2, i3 - 1, i4] (i0=0); read A[i2 - 1, i3, i4] (i0=0) (+8) |
| n^4.5 | 0.438 | level | (1/4)·n^3 + (9/4)·n^2 + (11/4)·n + (-21/4) | (7/8)·n^3 + (-59/4)·n^2 + (329/8)·n + (-121/4) | 0.0547/n | read A[i2, i3 - 1, i4] (i0=0); read A[i2 - 1, i3, i4] (i0=0) (+8) |
| n^4.5 | 0.438 | level | (1/4)·n^3 + 2·n^2 + (9/4)·n + (-9/2) | (7/8)·n^3 + (-93/8)·n^2 + 31·n + (-45/2) | 0.0547/n | read A[i2, i3 - 1, i4] (i0=0); read A[i2 - 1, i3, i4] (i0=0) (+8) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-5/2) | (1/2)·n^3 + (-45/4)·n^2 + (199/4)·n - 51 | 0.0312/n | write B[i2, i3, i4] (i0=0, i3=0); write B[i2, i3, i4] (i0=0) (+1) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 3 | (1/2)·n^3 + (-43/4)·n^2 + 47·n - 48 | 0.0312/n | write B[i2, i3, i4] (i0=0, i3=0); write B[i2, i3, i4] (i0=0) (+1) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 2 | (1/2)·n^3 + (-31/4)·n^2 + (61/2)·n - 30 | 0.0312/n | write B[i2, i3, i4] (i0=0, i3=0); write B[i2, i3, i4] (i0=0) (+1) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-11/2) | (1/2)·n^3 + (-45/4)·n^2 + (199/4)·n - 51 | 0.0312/n | read A[i2 + 1, i3, i4] (i0=0, i3=0); read A[i2 + 1, i3, i4] (i0=0) (+1) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + n^2 + (-7/2)·n | (1/2)·n^3 + (-43/4)·n^2 + 47·n - 48 | 0.0312/n | read A[i2 + 1, i3, i4] (i0=0, i3=0); read A[i2 + 1, i3, i4] (i0=0) (+1) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 5 | (1/2)·n^3 + (-31/4)·n^2 + (61/2)·n - 30 | 0.0312/n | read A[i2 + 1, i3, i4] (i0=0, i3=0); read A[i2 + 1, i3, i4] (i0=0) (+1) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-21/8) | (1/2)·n^3 + (-45/4)·n^2 + (199/4)·n - 51 | 0.0312/n | read A[i2, i3 + 1, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0) (+2) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + n^2 + (-19/8)·n | (1/2)·n^3 + (-43/4)·n^2 + 47·n - 48 | 0.0312/n | read A[i2, i3 + 1, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0) (+2) |
| n^4.5 | 0.25 | level | (1/4)·n^3 + (3/2)·n^2 + (-3/8)·n + (-9/4) | (1/2)·n^3 + (-31/4)·n^2 + (61/2)·n - 30 | 0.0312/n | read A[i2, i3 + 1, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0) (+2) |
| n^4.5 | 0.217 | level | (3/4)·n | (1/4)·n^4 - 5·n^3 + (69/4)·n^2 + (-41/2)·n + 8 | 0.0156 | read A[i2, i3 - 1, i4] (i0=0); read B[i5, i6 - 1, i7] (i0=0) |
| n^4.5 | 0.217 | level | (3/4)·n - 1 | (1/4)·n^4 - 5·n^3 + (69/4)·n^2 + (-41/2)·n + 8 | 0.0156 | read A[i2, i3, i4 + 1] (i0=0); read B[i5, i6, i7 + 1] (i0=0) |
| n^4.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 6  →  (1/4)·n^3 + n^2 + (-29/8)·n | (1/4)·n^3 + (-41/8)·n^2 + (153/8)·n - 18 | 0.0156/n | write B[i2, i3, i4] (i0=0, i2=0); write A[i5, i6, i7] (i0=0) |
| n^4.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 3  →  (1/4)·n^3 + n^2 + (-29/8)·n - 3 | (1/4)·n^3 + (-41/8)·n^2 + (153/8)·n - 18 | 0.0156/n | read A[i2 + 1, i3, i4] (i0=0, i2=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n + 5  →  (1/4)·n^3 + n^2 + (-29/8)·n - 1 | (1/4)·n^3 + (-43/8)·n^2 + (47/2)·n - 24 | 0.0156/n | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^4.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n + 3  →  (1/4)·n^3 + n^2 + (-29/8)·n - 3 | (1/4)·n^3 + (-43/8)·n^2 + (47/2)·n - 24 | 0.0156/n | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^4.5 | 0.0625 | level | (1/4)·n^3 + (5/2)·n^2 + (-19/4)·n + 2 | (1/8)·n^3 + (-17/8)·n^2 + (7/4)·n + 4 | 0.00781/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0, i6=0) (+2) |
| n^4.5 | 0.0625 | level | (1/4)·n^3 + (9/4)·n^2 + (11/4)·n + (-21/4) | (1/8)·n^3 + (-5/4)·n^2 + (-1/8)·n + (17/4) | 0.00781/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0, i6=0) (+2) |
| n^4.5 | 0.0625 | level | (1/4)·n^3 + 2·n^2 + (9/4)·n + (-9/2) | (1/8)·n^3 + (-11/8)·n^2 + n + (5/2) | 0.00781/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0, i6=0) (+2) |
| n^4 | 18.4 | level | 6 | (15/2)·n^4 + (-49/2)·n^3 + (57/2)·n^2 + (-27/2)·n + 2 | 0.469 | read A[i2 + 1, i3, i4] (i0=0); read A[i2 - 1, i3, i4] (i0=0) (+8) |
| n^4 | 3.91 | level | 5 | (7/4)·n^4 + (-29/4)·n^3 + (45/4)·n^2 + (-31/4)·n + 2 | 0.109 | read A[i2, i3, i4 + 1] (i0=0); read B[i5, i6, i7 + 1] (i0=0) |
| n^4 | 3.31 | level | 7 | (5/4)·n^4 + (-47/4)·n^3 + (111/4)·n^2 + (-101/4)·n + 8 | 0.0781 | read A[i2 + 1, i3, i4] (i0=0); read A[i2 - 1, i3, i4] (i0=0) (+8) |
| n^4 | 2.12 | level | 2 | (3/2)·n^4 + (-9/2)·n^3 + (9/2)·n^2 + (-3/2)·n | 0.0938 | read A[i2, i3, i4] (i0=0); read B[i5, i6, i7] (i0=0) |
| n^4 | 1.5 | level | 1 | (3/2)·n^4 + (-9/2)·n^3 + (9/2)·n^2 + (-3/2)·n | 0.0938 | read A[i2, i3, i4 - 1] (i0=0); read B[i5, i6, i7 - 1] (i0=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + 3·n + (-5/2) | 2·n^3 - 12·n^2 + 22·n - 12 | 0.125/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i7=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 1 | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i7=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + (5/2)·n - 2 | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i7=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + 3·n + (-9/2) | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i7=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 3 | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i7=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + (5/2)·n - 4 | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i7=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + 3·n + (-5/2) | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + 3·n + (-9/2) | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + 3·n + (-5/2) | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 1 | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + 3·n + (-9/2) | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^4 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 3 | 2·n^3 - 14·n^2 + 28·n - 16 | 0.125/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^4 | 0.866 | level | 3 | (1/2)·n^4 + (-11/2)·n^3 + (27/2)·n^2 + (-25/2)·n + 4 | 0.0312 | read A[i2, i3, i4] (i0=0); read B[i5, i6, i7] (i0=0) |
| n^4 | 0.661 | level | 7 | (1/4)·n^4 + (-11/4)·n^3 + (27/4)·n^2 + (-25/4)·n + 2 | 0.0156 | read A[i2, i3, i4 - 1] (i0=0); read B[i5, i6, i7 - 1] (i0=0) |
| n^4 | 0.612 | level | 6 | (1/4)·n^4 + (-3/4)·n^3 + (3/4)·n^2 + (-1/4)·n | 0.0156 | read A[i2, i3, i4 - 1] (i0=0); read B[i5, i6, i7 - 1] (i0=0) |
| n^4 | 0.177 | level | (1/2)·n^2 + (13/4)·n + (-7/4) | (1/4)·n^3 - 3·n^2 + (29/4)·n + (-9/2) | 0.0156/n | read A[i2, i3, i4 + 1] (i0=0); read B[i5, i6, i7 + 1] (i0=0, i6=0) |
| n^4 | 0.177 | level | (1/2)·n^2 + (3/4)·n - 1 | (1/4)·n^3 + (-19/4)·n^2 + (25/2)·n - 8 | 0.0156/n | read A[i2, i3, i4 + 1] (i0=0); read B[i5, i6, i7 + 1] (i0=0, i6=0) |
| n^4 | 0.177 | level | (1/2)·n^2 + (11/4)·n + (-3/2) | (1/4)·n^3 + (-13/4)·n^2 + 8·n - 5 | 0.0156/n | read A[i2, i3, i4 + 1] (i0=0); read B[i5, i6, i7 + 1] (i0=0, i6=0) |
| n^4 | 0.177 | level | (1/2)·n^2 + (13/4)·n + (-15/4) | (1/4)·n^3 - 5·n^2 + (53/4)·n + (-17/2) | 0.0156/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^4 | 0.177 | level | (1/2)·n^2 + (3/4)·n - 3 | (1/4)·n^3 + (-19/4)·n^2 + (25/2)·n - 8 | 0.0156/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^4 | 0.175 | ramp | (1/2)·n^2 + (1/4)·n + 3  →  (1/2)·n^2 + (1/2)·n - 3 | (1/4)·n^3 + (-19/4)·n^2 + (25/2)·n - 8 | 0.0156/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^4 | 0.175 | ramp | (1/2)·n^2 + (1/4)·n + 2  →  (1/2)·n^2 + (1/2)·n - 4 | (1/4)·n^3 + (-19/4)·n^2 + (25/2)·n - 8 | 0.0156/n | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i6=0) |
| n^4 | 0.175 | ramp | (1/2)·n^2 + (1/4)·n  →  (1/2)·n^2 + (1/2)·n - 6 | (1/4)·n^3 + (-19/4)·n^2 + (25/2)·n - 8 | 0.0156/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^4 | 0.175 | ramp | (1/2)·n^2 + (1/4)·n  →  (1/2)·n^2 + (1/2)·n - 6 | (1/4)·n^3 + (-19/4)·n^2 + (25/2)·n - 8 | 0.0156/n | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i6=0) |
| n^3.5 | 4 | level | (1/4)·n^3 + (5/2)·n^2 + (-19/4)·n + 2 | 8·n^2 - 24·n + 16 | 0.5·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read A[i2, i3 - 1, i4] (i0=0) (+10) |
| n^3.5 | 4 | level | (1/4)·n^3 + (5/2)·n^2 + (-19/4)·n + 2 | 8·n^2 - 24·n + 16 | 0.5·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read A[i2, i3 - 1, i4] (i0=0) (+10) |
| n^3.5 | 4 | level | (1/4)·n^3 + (9/4)·n^2 + (11/4)·n + (-21/4) | 8·n^2 - 24·n + 16 | 0.5·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read A[i2, i3 - 1, i4] (i0=0) (+10) |
| n^3.5 | 3 | level | (1/4)·n^3 + (9/4)·n^2 + (11/4)·n + (-21/4) | 6·n^2 - 18·n + 12 | 0.375·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read A[i2, i3 - 1, i4] (i0=0) (+6) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-5/2) | 4·n^2 - 20·n + 21 | 0.25·n^-2 | write B[i2, i3, i4] (i0=0, i3=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 3 | 4·n^2 - 22·n + 24 | 0.25·n^-2 | write B[i2, i3, i4] (i0=0, i3=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 2 | 4·n^2 - 20·n + 21 | 0.25·n^-2 | write B[i2, i3, i4] (i0=0, i3=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-11/2) | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i3=0, i4=0); read A[i2 + 1, i3, i4] (i0=0, i4=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + n^2 + (-7/2)·n | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i3=0, i4=0); read A[i2 + 1, i3, i4] (i0=0, i4=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 5 | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i3=0, i4=0); read A[i2 + 1, i3, i4] (i0=0, i4=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-5/2) | 4·n^2 - 20·n + 21 | 0.25·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0, i3=0) (+3) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-11/2) | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i3=0); read A[i2 + 1, i3, i4] (i0=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-5/2) | 4·n^2 - 20·n + 21 | 0.25·n^-2 | write B[i2, i3, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0, i3=0) (+2) |
| n^3.5 | 2 | level | (1/4)·n^3 + n^2 + (-7/2)·n + 3 | 4·n^2 - 20·n + 21 | 0.25·n^-2 | write B[i2, i3, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0, i3=0) (+2) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-11/2) | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i3=0); read A[i2 + 1, i3, i4] (i0=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + n^2 + (-7/2)·n - 1 | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i3=0); read A[i2 + 1, i3, i4] (i0=0) (+1) |
| n^3.5 | 2 | level | (1/4)·n^3 + n^2 + (-19/8)·n | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+2) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-21/8) | 4·n^2 - 22·n + 24 | 0.25·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0) (+2) |
| n^3.5 | 2 | level | (1/4)·n^3 + (9/4)·n^2 + (11/4)·n + (-21/4) | 4·n^2 - 14·n + 12 | 0.25·n^-2 | read A[i2, i3 - 1, i4] (i0=0, i2=0, i3=0); read A[i2 - 1, i3, i4] (i0=0, i2=0) (+5) |
| n^3.5 | 2 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-21/8) | 4·n^2 - 20·n + 21 | 0.25·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0); write B[i2, i3, i4] (i0=0, i3=0) (+3) |
| n^3.5 | 1.73 | level | (3/4)·n | 2·n^3 - 8·n^2 + 10·n - 4 | 0.125/n | read A[i2, i3 - 1, i4] (i0=0); read B[i5, i6 - 1, i7] (i0=0, i7=0) |
| n^3.5 | 1.73 | level | (3/4)·n - 1 | 2·n^3 - 8·n^2 + 10·n - 4 | 0.125/n | read A[i2, i3, i4] (i0=0); read B[i5, i6, i7] (i0=0, i7=0) |
| n^3.5 | 1.73 | level | (3/4)·n | 2·n^3 - 8·n^2 + 10·n - 4 | 0.125/n | read A[i2, i3 - 1, i4] (i0=0); read B[i5, i6 - 1, i7] (i0=0) |
| n^3.5 | 1.73 | level | (3/4)·n + (21/4) | 2·n^3 - 8·n^2 + 10·n - 4 | 0.125/n | read A[i2, i3 - 1, i4] (i0=0); read B[i5, i6 - 1, i7] (i0=0) |
| n^3.5 | 1.73 | level | (3/4)·n - 1 | 2·n^3 - 8·n^2 + 10·n - 4 | 0.125/n | read A[i2, i3, i4 + 1] (i0=0, i1=0); read B[i5, i6, i7 + 1] (i0=0, i1=0) (+5) |
| n^3.5 | 1.73 | level | (3/4)·n + (13/4) | 2·n^3 - 8·n^2 + 10·n - 4 | 0.125/n | read A[i2, i3, i4 + 1] (i0=0, i1=0); read B[i5, i6, i7 + 1] (i0=0, i1=0) (+5) |
| n^3.5 | 1 | level | (1/4)·n^3 + (9/4)·n^2 + (11/4)·n + (-21/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read A[i2, i3 - 1, i4] (i0=0) (+10) |
| n^3.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-21/8) | 2·n^2 - 11·n + 12 | 0.125·n^-2 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-21/8) | 2·n^2 - 11·n + 12 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i4=0); write B[i2, i3, i4] (i0=0, i4=0) (+2) |
| n^3.5 | 1 | level | (1/4)·n^3 + n^2 + (-19/8)·n | 2·n^2 - 9·n + 9 | 0.125·n^-2 | write B[i2, i3, i4] (i0=0, i3=0); write B[i2, i3, i4] (i0=0) (+1) |
| n^3.5 | 1 | level | (1/4)·n^3 + n^2 + (-19/8)·n - 1 | 2·n^2 - 11·n + 12 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^3.5 | 0.999 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n + 2  →  (1/4)·n^3 + n^2 + (-29/8)·n + 2 | 2·n^2 - 9·n + 9 | 0.125·n^-2 | write B[i2, i3, i4] (i0=0, i2=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.999 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n - 1  →  (1/4)·n^3 + n^2 + (-29/8)·n - 1 | 2·n^2 - 9·n + 9 | 0.125·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.999 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n - 2  →  (1/4)·n^3 + n^2 + (-29/8)·n - 2 | 2·n^2 - 9·n + 9 | 0.125·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i2=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.999 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 4  →  (1/4)·n^3 + n^2 + (-31/8)·n + 4 | 2·n^2 - 9·n + 9 | 0.125·n^-2 | write B[i2, i3, i4] (i0=0, i2=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.999 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 1  →  (1/4)·n^3 + n^2 + (-31/8)·n + 1 | 2·n^2 - 9·n + 9 | 0.125·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.998 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-19/8)·n + 1  →  (1/4)·n^3 + n^2 + (-29/8)·n + 1 | 2·n^2 - 11·n + 12 | 0.125·n^-2 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.998 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n + 3  →  (1/4)·n^3 + n^2 + (-31/8)·n + 3 | 2·n^2 - 11·n + 12 | 0.125·n^-2 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.998 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n  →  (1/4)·n^3 + n^2 + (-31/8)·n | 2·n^2 - 11·n + 12 | 0.125·n^-2 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.125 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/4)·n + (-7/4) | (1/4)·n^2 + (-37/8)·n + (51/8) | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.125 | level | (1/4)·n^3 + n^2 + (-9/4)·n + 1 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.125 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/4)·n + (-15/4) | (1/4)·n^2 + (-21/8)·n + (27/8) | 0.0156·n^-2 | read A[i2, i3, i4 + 1] (i0=0, i2=0, i3=0); read B[i5, i6, i7 + 1] (i0=0) |
| n^3.5 | 0.125 | level | (1/4)·n^3 + n^2 + (-9/4)·n - 1 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2, i3, i4 + 1] (i0=0, i2=0, i3=0); read B[i5, i6, i7 + 1] (i0=0) |
| n^3.5 | 0.125 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/4)·n + (-7/2) | (1/4)·n^2 + (-23/8)·n + (15/4) | 0.0156·n^-2 | read A[i2, i3, i4 + 1] (i0=0, i2=0, i3=0); read B[i5, i6, i7 + 1] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-5/2)·n + 2  →  (1/4)·n^3 + n^2 + (-19/8)·n - 1 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0, i3=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-5/2)·n + 1  →  (1/4)·n^3 + n^2 + (-19/8)·n - 2 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-21/8)·n + 3  →  (1/4)·n^3 + n^2 + (-19/8)·n - 3 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i3=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-21/8)·n + 3  →  (1/4)·n^3 + n^2 + (-19/8)·n - 3 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-29/8)·n + 5  →  (1/4)·n^3 + n^2 + (-7/2)·n + 2 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0, i2=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-29/8)·n + 4  →  (1/4)·n^3 + n^2 + (-7/2)·n + 1 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0, i3=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-29/8)·n + 1  →  (1/4)·n^3 + n^2 + (-7/2)·n - 2 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i2=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + n^2 + (-29/8)·n + 1  →  (1/4)·n^3 + n^2 + (-7/2)·n - 2 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i3=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 5  →  (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n - 1 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 3  →  (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n - 3 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n + 9  →  (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0, i2=0, i3=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n + 8  →  (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n - 1 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n + 4  →  (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n - 5 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3.5 | 0.124 | ramp | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n + 4  →  (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n - 5 | (1/4)·n^2 + (-35/8)·n + 6 | 0.0156·n^-2 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i3=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 1 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + 3·n + (-11/2) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 4 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (11/4)·n + (-13/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/4)·n | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (9/4)·n + (-5/2) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (11/4)·n + (-21/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/4)·n - 2 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (9/4)·n + (-9/2) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (11/4)·n + (-1/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (11/4)·n + (-13/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (11/4)·n + (-5/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/4)·n + 1 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 2 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i6=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + 3·n + (-7/2) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3 + 1, i4] (i0=0); read B[i5, i6 + 1, i7] (i0=0, i6=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (11/4)·n + (-17/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/4)·n - 2 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (1/2)·n - 4 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i6=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (3/4)·n - 3 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (3/4)·n - 1 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3, i4] (i0=0); read B[i5, i6, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (13/4)·n + (-7/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3, i4] (i0=0); read B[i5, i6, i7] (i0=0, i6=0, i7=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (13/4)·n + (-27/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (3/4)·n - 3 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (13/4)·n + (-11/4) | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2 - 1, i3, i4] (i0=0); read B[i5 - 1, i6, i7] (i0=0) |
| n^3 | 1.41 | level | (1/2)·n^2 + (3/4)·n - 1 | 2·n^2 - 6·n + 4 | 0.125·n^-2 | read A[i2, i3, i4 + 1] (i0=0, i1=0); read B[i5, i6, i7 + 1] (i0=0, i1=0, i6=0) (+3) |
| n^3 | 1.41 | level | (1/2)·n^2 + (13/4)·n + (-7/4) | 2·n^2 - 8·n + 8 | 0.125·n^-2 | read A[i2, i3, i4 + 1] (i0=0); read B[i5, i6, i7 + 1] (i0=0, i5=1, i6=0) (+1) |
| n^2.5 | 2 | level | (1/4)·n^3 + (3/2)·n^2 + (-13/8)·n + (15/8) | 4·n - 6 | 0.25·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i3=0); write B[i2, i3, i4] (i0=0) (+1) |
| n^2.5 | 2 | level | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 3 | 4·n - 6 | 0.25·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i3=0); write B[i2, i3, i4] (i0=0) (+1) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-11/8)·n + (13/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-13/8)·n + (7/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-21/8)·n + 1 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (5/4)·n^2 + (-9/8)·n + (5/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n + 2 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (5/4)·n^2 + (-11/8)·n + (3/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-19/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-29/8)·n + 4 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-5/8)·n + (-7/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 - 2·n + (9/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i3=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n + 6 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i3=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (5/4)·n^2 + (-7/4)·n + (5/2) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i2=0, i3=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-13/8)·n + (-25/8) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n - 2 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (5/4)·n^2 + (-11/8)·n + (-13/4) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-13/2) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i3=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-7/2)·n - 1 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i3=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-1/2)·n - 6 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i3=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-51/8) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-29/8)·n | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-5/8)·n + (-23/4) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 - 2·n + (-11/4) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i3=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n + 1 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i3=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (5/4)·n^2 + (-7/4)·n + (-5/2) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i3=0, i4=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-13/8)·n + (31/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 - 2·n + (17/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-19/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i3=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-13/8)·n + (15/8) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 - 2·n + (1/4) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-43/8) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i3=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-15/2) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 - 2·n + (9/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n + 5 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-27/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i3=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-29/8)·n + 3 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i3=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-13/8)·n + (-1/8) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 - 2·n + (-15/4) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-13/4)·n - 1 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-51/8) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i3=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-29/8)·n - 1 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i3=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/2)·n + (-13/2) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-7/2)·n - 2 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/2)·n^2 + (-13/8)·n + (-25/8) | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i3=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (3/4)·n^2 + (-23/8)·n - 3 | 2·n - 3 | 0.125·n^-3 | read A[i2 + 1, i3, i4] (i0=0, i2=0, i3=0); read B[i5 + 1, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-29/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-19/8)·n - 1 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-5/2)·n + 1 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i3=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-19/8)·n - 1 | 2·n - 3 | 0.125·n^-3 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i4=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-29/8) | 2·n - 3 | 0.125·n^-3 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i4=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-21/8)·n + 1 | 2·n - 3 | 0.125·n^-3 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i3=0, i4=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-27/8) | 2·n - 3 | 0.125·n^-3 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i3=0, i4=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-11/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-29/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i3=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-5/8)·n + (-19/8) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-21/8)·n + 1 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-5/2)·n - 1 | 2·n - 3 | 0.125·n^-3 | read A[i2, i3 + 1, i4] (i0=0, i2=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-3/8)·n + (-29/8) | 2·n - 3 | 0.125·n^-3 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i3=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-19/8)·n - 2 | 2·n - 3 | 0.125·n^-3 | read A[i2, i3 + 1, i4] (i0=0, i2=0, i3=0); read B[i5, i6 + 1, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-9/4)·n + 1 | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0, i4=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-9/4)·n - 1 | 2·n - 3 | 0.125·n^-3 | read A[i2, i3, i4] (i0=0, i2=0, i3=0, i4=0); read B[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/4)·n + (-15/4) | 2·n - 3 | 0.125·n^-3 | read A[i2, i3, i4] (i0=0, i2=0, i3=0, i4=0); read B[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/4)·n + (-19/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-9/4)·n | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/4)·n + (-7/4) | 2·n - 3 | 0.125·n^-3 | write B[i2, i3, i4] (i0=0); write A[i5, i6, i7] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/4)·n + (-15/4) | 2·n - 4 | 0.125·n^-3 | read A[i2, i3, i4 + 1] (i0=0, i2=0, i3=0); read B[i5, i6, i7 + 1] (i0=0) |
| n^2.5 | 1 | level | (1/4)·n^3 + n^2 + (-9/4)·n - 2 | 2·n - 3 | 0.125·n^-3 | read B[i5, i6, i7 + 1] (i0=0, i1=0); read A[i2, i3, i4 + 1] (i0=0, i2=0, i3=0) (+1) |
| n^2.5 | 0.5 | level | (1/4)·n^3 + (9/4)·n^2 + (11/4)·n + (-21/4) | n - 2 | 0.0625·n^-3 | read A[i2 - 1, i3, i4] (i0=0, i2=0, i3=0) |
| n^2 | 1.41 | level | (1/2)·n^2 + (13/4)·n + (-7/4) | 2·n - 4 | 0.125·n^-3 | read A[i2, i3, i4 + 1] (i0=0, i1=0); read B[i5, i6, i7 + 1] (i0=0, i1=0, i6=0) |
| n^1.5 | 0.5 | level | (1/4)·n^3 + (7/4)·n^2 + (-1/4)·n + (-15/4) | 1 | 0.0625·n^-4 | read B[i5, i6, i7 + 1] (i0=0, i1=0, i5=0, i6=0) |

3-D two-array stencil, the suite's only headroom +1.5 kernel: accesses grow as n^4 (time × volume) while the cross-sweep reuses of `write B[i2,i3,i4]` and the neighbor reads sit at the two-array volume footprint (1/4)n^3 + O(n^2) lines — reuse-distance growth ρ = 3, so d = a + ρ/2 = 5.5. Eight symmetric families of coefficient 0.125 sum to 0.75·n^5.5. The time-tiling cliff is at 64·((1/4)n^3) bytes; below it every sweep refetches both volumes.

## jacobi-1d — infinite-repeat  [`exact`]

Accesses $A(n) = 8·n^2 - 8·n$ (exact on n ≡ 0 mod 8); DMD order $n^{2.5}$, headroom **+0.5**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.25·n^2.5  +  9.4·n^2  +  6.12·n^1.5  +  4.25·n^1  +  2·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^2.5 | 0.125 | level | (1/4)·n + 1 | (1/4)·n^2 - 4·n | 0.0312 | write B[i2] (i0=0, i1=0); write A[i3] (i0=0, i1=0) (+2) |
| n^2.5 | 0.125 | level | (1/4)·n + 1 | (1/4)·n^2 + (-17/4)·n + 4 | 0.0312 | read A[i2 + 1] (i0=0, i1=1); read A[i2 + 1] (i0=0) (+1) |
| n^2 | 2.12 | level | 2 | (3/2)·n^2 - 2·n | 0.188 | write B[i2] (i0=0, i1=0); write A[i3] (i0=0, i1=0) (+2) |
| n^2 | 2.12 | level | 2 | (3/2)·n^2 + (-3/2)·n | 0.188 | read A[i2 - 1] (i0=0); read B[i3 - 1] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 + (-15/4)·n + 2 | 0.219 | read A[i2 + 1] (i0=0); read B[i3 + 1] (i0=0) |
| n^2 | 1.5 | level | 1 | (3/2)·n^2 | 0.188 | read A[i2] (i0=0, i1=0); read B[i3] (i0=0, i1=0) (+2) |
| n^2 | 0.866 | level | 3 | (1/2)·n^2 - 4·n | 0.0625 | read A[i2 - 1] (i0=0, i1=0); read A[i2] (i0=0, i1=0) (+6) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 | 0.0312 | write B[i2] (i0=0, i1=0); write A[i3] (i0=0, i1=0) (+2) |
| n^2 | 0.354 | level | 2 | (1/4)·n^2 + (-1/2)·n | 0.0312 | read A[i2 - 1] (i0=0, i1=0); read B[i3 - 1] (i0=0, i1=0) (+2) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 - 2·n + 2 | 0.0312 | read A[i2] (i0=0, i1=0); read B[i3] (i0=0, i1=0) (+2) |
| n^1.5 | 2 | level | (1/4)·n + 1 | 4·n | 0.5/n | read A[i2 - 1] (i0=0, i1=0, i2=0); write B[i2] (i0=0, i1=0, i2=0) (+7) |
| n^1.5 | 1 | level | (1/4)·n + 1 | 2·n | 0.25/n | write B[i2] (i0=0, i1=0); write A[i3] (i0=0, i1=0) (+2) |
| n^1.5 | 1 | level | (1/4)·n + (7/4) | 2·n | 0.25/n | write B[i2] (i0=0, i1=0); write A[i3] (i0=0, i1=0) (+2) |
| n^1.5 | 1 | level | (1/4)·n - 1 | 2·n - 2 | 0.25/n | read A[i2 + 1] (i0=0, i1=1); read A[i2 + 1] (i0=0) (+1) |
| n^1.5 | 1 | level | (1/4)·n + 2 | 2·n - 2 | 0.25/n | read A[i2 + 1] (i0=0, i1=1); read A[i2 + 1] (i0=0) (+1) |
| n^1.5 | 0.125 | level | (1/4)·n + 1 | (1/4)·n - 4 | 0.0312/n | read A[i2 + 1] (i0=0, i1=0); read B[i3 + 1] (i0=0, i1=0) |
| n^1 | 2 | level | 1 | 2·n | 0.25/n | read A[i2 - 1] (i0=0, i1=0, i2=0); read A[i2 + 1] (i0=0, i1=0, i2=0) (+7) |
| n^1 | 1.25 | level | 1 | (5/4)·n | 0.156/n | read A[i2 + 1] (i0=0, i1=0); read B[i3 + 1] (i0=0, i1=0) |
| n^1 | 0.5 | level | 1 | (1/2)·n - 4 | 0.0625/n | read A[i2 + 1] (i0=0, i1=0); read B[i3 + 1] (i0=0, i1=0) |
| n^1 | 0.5 | level | 1 | (1/2)·n + (-5/2) | 0.0625/n | read A[i2 + 1] (i0=0, i1=0); read B[i3 + 1] (i0=0, i1=0) |
| n^0.5 | 1 | level | (1/4)·n - 1 | 2 | 0.25·n^-2 | read A[i2 + 1] (i0=0, i1=0); read B[i3 + 1] (i0=0, i1=0) |
| n^0.5 | 1 | level | (1/4)·n + 2 | 2 | 0.25·n^-2 | read A[i2 + 1] (i0=0, i1=0); read B[i3 + 1] (i0=0, i1=0) |

Two-array 1-D stencil: the cross-sweep reuses (`write B[i2]` consumed by the next sweep, right-neighbor reads) sit at the array footprint (1/4)n + 1 lines — order n^2.5, headroom +0.5, coefficient 0.25 (two arrays × two sweep directions). The bulk (distance ≤ 2) is neighbor line reuse. The correct transformation is time-tiling/fusion of the two sweeps; repetition adds nothing (the arrays are re-touched every time step already).

## jacobi-1d — single-shot  [`exact`]

Accesses $A(n) = 8·n^2 - 8·n$ (exact on n ≡ 0 mod 8); DMD order $n^{2.5}$, headroom **+0.5**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.25·n^2.5  +  9.4·n^2  +  6·n^1.5  +  2·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^2.5 | 0.125 | level | (1/4)·n + 1 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0312 | write B[i2] (i0=0); write A[i3] (i0=0) |
| n^2.5 | 0.125 | level | (1/4)·n + 1 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0312 | read A[i2 + 1] (i0=0); read B[i3 + 1] (i0=0) |
| n^2 | 2.12 | level | 2 | (3/2)·n^2 - 2·n | 0.188 | write B[i2] (i0=0); write A[i3] (i0=0) |
| n^2 | 2.12 | level | 2 | (3/2)·n^2 | 0.188 | read A[i2 - 1] (i0=0); read B[i3 - 1] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 - 2·n | 0.219 | read A[i2 + 1] (i0=0); read B[i3 + 1] (i0=0) |
| n^2 | 1.5 | level | 1 | (3/2)·n^2 | 0.188 | read A[i2] (i0=0); read B[i3] (i0=0) |
| n^2 | 0.866 | level | 3 | (1/2)·n^2 - 4·n | 0.0625 | read A[i2 - 1] (i0=0); read A[i2] (i0=0) (+2) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 | 0.0312 | write B[i2] (i0=0); write A[i3] (i0=0) |
| n^2 | 0.354 | level | 2 | (1/4)·n^2 - 2·n | 0.0312 | read A[i2 - 1] (i0=0); read B[i3 - 1] (i0=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 - 2·n | 0.0312 | read A[i2] (i0=0); read B[i3] (i0=0) |
| n^1.5 | 2 | level | (1/4)·n + 1 | 4·n - 2 | 0.5/n | read A[i2 - 1] (i0=0, i2=0); write B[i2] (i0=0, i2=0) (+2) |
| n^1.5 | 1 | level | (1/4)·n + 1 | 2·n - 1 | 0.25/n | write B[i2] (i0=0); write A[i3] (i0=0) |
| n^1.5 | 1 | level | (1/4)·n + (7/4) | 2·n - 1 | 0.25/n | write B[i2] (i0=0); write A[i3] (i0=0) |
| n^1.5 | 1 | level | (1/4)·n - 1 | 2·n - 1 | 0.25/n | read A[i2 + 1] (i0=0); read B[i3 + 1] (i0=0) |
| n^1.5 | 1 | level | (1/4)·n + 2 | 2·n - 2 | 0.25/n | read A[i2 + 1] (i0=0); read B[i3 + 1] (i0=0) |
| n^1 | 2 | level | 1 | 2·n | 0.25/n | read A[i2 - 1] (i0=0, i2=0); read A[i2] (i0=0, i2=0) (+2) |

Two-array 1-D stencil: the cross-sweep reuses (`write B[i2]` consumed by the next sweep, right-neighbor reads) sit at the array footprint (1/4)n + 1 lines — order n^2.5, headroom +0.5, coefficient 0.25 (two arrays × two sweep directions). The bulk (distance ≤ 2) is neighbor line reuse. The correct transformation is time-tiling/fusion of the two sweeps; repetition adds nothing (the arrays are re-touched every time step already).

## jacobi-2d — infinite-repeat  [`exact`]

Accesses $A(n) = 12·n^3 - 24·n^2 + 12·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.5·n^4  +  0.707·n^3.5  +  30.4·n^3  +  8.84·n^2.5  +  33.4·n^2  +  1.41·n^1.5  +  1.5·n^1  +  7.41·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.125 | level | (1/4)·n^2 + n - 2 | (1/4)·n^3 + (-21/4)·n^2 + 20·n | 0.0208 | write B[i2, i3] (i0=0, i1=0, i2=1); write B[i2, i3] (i0=0, i1=0) (+4) |
| n^4 | 0.125 | level | (1/4)·n^2 + (7/4)·n | (1/4)·n^3 + (-7/2)·n^2 + (45/4)·n | 0.0208 | write B[i2, i3] (i0=0, i1=0, i2=1); write B[i2, i3] (i0=0, i1=0) (+4) |
| n^4 | 0.125 | level | (1/4)·n^2 + n - 3 | (1/4)·n^3 + (-21/4)·n^2 + 20·n | 0.0208 | read A[i2 + 1, i3] (i0=0, i1=0, i2=1); read A[i2 + 1, i3] (i0=0, i1=0) (+6) |
| n^4 | 0.125 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/4)·n^3 + (-7/2)·n^2 + (45/4)·n | 0.0208 | read A[i2 + 1, i3] (i0=0, i1=0, i2=1); read A[i2 + 1, i3] (i0=0, i1=0) (+6) |
| n^3.5 | 0.177 | level | (1/2)·n + (5/2) | (1/4)·n^3 + (-19/4)·n^2 + (17/2)·n | 0.0208 | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+2) |
| n^3.5 | 0.177 | level | (1/2)·n | (1/4)·n^3 + (-9/2)·n^2 + 8·n | 0.0208 | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+2) |
| n^3.5 | 0.177 | level | (1/2)·n + 1 | (1/4)·n^3 + (-19/4)·n^2 + (25/2)·n - 8 | 0.0208 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3.5 | 0.177 | level | (1/2)·n + (7/2) | (1/4)·n^3 - 3·n^2 + (29/4)·n + (-9/2) | 0.0208 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 6 | level | 4 | 3·n^3 + (-17/2)·n^2 + (15/2)·n - 2 | 0.25 | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+4) |
| n^3 | 4 | level | 4 | 2·n^3 - 6·n^2 + 4·n | 0.167 | read A[i2, i3] (i0=0); read B[i4, i5] (i0=0) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (-7/4)·n^2 - 2·n + 2 | 0.146 | read A[i2, i3 - 1] (i0=0, i1=0); read B[i4, i5 - 1] (i0=0, i1=0) (+2) |
| n^3 | 1.68 | level | 5 | (3/4)·n^3 + (-11/4)·n^2 + 2·n | 0.0625 | read A[i2, i3 - 1] (i0=0, i1=0); read A[i2 + 1, i3] (i0=0, i1=0) (+10) |
| n^3 | 1.5 | level | 4 | (3/4)·n^3 + (-7/4)·n^2 + (-3/4)·n + (7/4) | 0.0625 | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+1) |
| n^3 | 1.5 | level | 4 | (3/4)·n^3 - n^2 + (-3/4)·n + 1 | 0.0625 | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+1) |
| n^3 | 1.5 | level | 4 | (3/4)·n^3 + (-5/2)·n^2 + (11/4)·n - 1 | 0.0625 | read A[i2 - 1, i3] (i0=0) |
| n^3 | 1.5 | level | 1 | (3/2)·n^3 - 3·n^2 + (3/2)·n | 0.125 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 1 | level | (1/4)·n^2 + n - 2 | 2·n^2 - 8·n | 0.167/n | read A[i2, i3] (i0=0, i1=0, i2=0, i3=0); write B[i2, i3] (i0=0, i1=0, i2=2, i3=0) (+9) |
| n^3 | 1 | level | (1/4)·n^2 + n - 3 | 2·n^2 - 8·n | 0.167/n | read A[i2 + 1, i3] (i0=0, i1=0, i2=2, i3=0); read A[i2 + 1, i3] (i0=0, i1=0, i3=0) (+10) |
| n^3 | 1 | level | (1/4)·n^2 + (7/4)·n | 2·n^2 - 10·n | 0.167/n | write B[i2, i3] (i0=0, i1=0, i2=2, i3=0); write B[i2, i3] (i0=0, i1=0, i3=0) (+4) |
| n^3 | 1 | level | (1/4)·n^2 + (7/4)·n - 1 | 2·n^2 - 6·n | 0.167/n | read A[i2, i3] (i0=0, i1=0, i2=0, i3=0); read A[i2 + 1, i3] (i0=0, i1=0, i2=2, i3=0) (+15) |
| n^3 | 1 | level | (1/4)·n^2 + n - 3 | 2·n^2 - 8·n | 0.167/n | write B[i2, i3] (i0=0, i1=0, i2=0); write B[i2, i3] (i0=0, i1=0) (+4) |
| n^3 | 1 | level | (1/4)·n^2 + n - 4 | 2·n^2 - 10·n | 0.167/n | read A[i2 + 1, i3] (i0=0, i1=0, i2=1); read A[i2 + 1, i3] (i0=0, i1=0) (+6) |
| n^3 | 1 | level | (1/4)·n^2 + (9/4)·n - 2 | 2·n^2 - 4·n + 2 | 0.167/n | read A[i2, i3 + 1] (i0=0, i1=1); read B[i4, i5 + 1] (i0=0, i1=1, i4=0) (+6) |
| n^3 | 0.559 | level | 5 | (1/4)·n^3 + (-1/2)·n^2 + (1/4)·n | 0.0208 | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+2) |
| n^3 | 0.559 | level | 5 | (1/4)·n^3 + (-1/4)·n^2 | 0.0208 | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+2) |
| n^3 | 0.354 | level | 2 | (1/4)·n^3 + (-3/4)·n^2 + (3/4)·n + (-1/4) | 0.0208 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 0.354 | level | 2 | (1/4)·n^3 + (-9/4)·n^2 + 2·n - 1 | 0.0208 | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0, i4=0) (+4) |
| n^3 | 0.25 | level | (1/4)·n^2 + (7/4)·n | (1/2)·n^2 + (-13/2)·n | 0.0417/n | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 2 | (1/4)·n^2 - 4·n | 0.0208/n | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 3 | (1/4)·n^2 - 4·n | 0.0208/n | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^3 | 0.125 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/4)·n^2 + (-9/4)·n | 0.0208/n | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 1 | (1/4)·n^2 - 4·n | 0.0208/n | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^3 | 0.125 | level | (1/4)·n^2 + (9/4)·n - 2 | (1/4)·n^2 - 4·n | 0.0208/n | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0, i4=0) (+3) |
| n^3 | 0.125 | level | (1/4)·n^2 + 2·n + (7/4) | (1/4)·n^2 + (-9/4)·n | 0.0208/n | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0, i4=0) (+3) |
| n^3 | 0.125 | level | (1/4)·n^2 + (7/4)·n + (3/2) | (1/4)·n^2 + (-5/2)·n | 0.0208/n | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0, i4=0) (+3) |
| n^3 | 0.125 | level | (1/4)·n^2 + (9/4)·n - 2 | (1/4)·n^2 - 4·n | 0.0208/n | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+4) |
| n^3 | 0.125 | level | (1/4)·n^2 + 2·n + (7/4) | (1/4)·n^2 + (-9/4)·n | 0.0208/n | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+4) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 2 | (1/4)·n^2 + (-17/4)·n + 4 | 0.0208/n | read A[i2, i3 + 1] (i0=0, i1=1, i2=0); read A[i2, i3 + 1] (i0=0, i2=0) (+1) |
| n^3 | 0.125 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/4)·n^2 + (-5/2)·n + (9/4) | 0.0208/n | read A[i2, i3 + 1] (i0=0, i1=1, i2=0); read A[i2, i3 + 1] (i0=0, i2=0) (+1) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n + 2  →  (1/4)·n^2 + n - 4 | (1/4)·n^2 - 4·n | 0.0208/n | write B[i2, i3] (i0=0, i1=0, i2=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n + 1  →  (1/4)·n^2 + n - 5 | (1/4)·n^2 - 4·n | 0.0208/n | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n  →  (1/4)·n^2 + n - 6 | (1/4)·n^2 - 4·n | 0.0208/n | read A[i2 + 1, i3] (i0=0, i1=0, i2=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n  →  (1/4)·n^2 + n - 6 | (1/4)·n^2 - 4·n | 0.0208/n | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2.5 | 1.41 | level | (1/2)·n | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0, i5=0) (+2) |
| n^2.5 | 1.41 | level | (1/2)·n + 1 | 2·n^2 - 4·n | 0.167/n | read A[i2, i3] (i0=0, i1=0); read B[i4, i5] (i0=0, i1=0, i5=0) (+2) |
| n^2.5 | 1.41 | level | (1/2)·n + (5/2) | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+2) |
| n^2.5 | 1.41 | level | (1/2)·n + 2 | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+2) |
| n^2.5 | 1.41 | level | (1/2)·n + (9/2) | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0) (+2) |
| n^2.5 | 1.41 | level | (1/2)·n - 1 | 2·n^2 - 6·n + 4 | 0.167/n | read A[i2, i3 + 1] (i0=0, i1=1); read B[i4, i5 + 1] (i0=0, i1=1) (+2) |
| n^2.5 | 0.177 | level | (1/2)·n + 1 | (1/4)·n^2 + (-9/2)·n + 8 | 0.0208/n | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0) |
| n^2.5 | 0.177 | level | (1/2)·n + (7/2) | (1/4)·n^2 + (-11/4)·n + (9/2) | 0.0208/n | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0) |
| n^2 | 3 | level | 4 | (3/2)·n^2 + (-7/2)·n + 2 | 0.125/n | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) |
| n^2 | 2 | level | (1/4)·n^2 + (7/4)·n | 4·n | 0.333·n^-2 | write B[i2, i3] (i0=0, i1=0, i2=1, i3=0); write B[i2, i3] (i0=0, i1=0, i3=0) (+4) |
| n^2 | 2 | level | (1/4)·n^2 + (9/4)·n - 2 | 4·n | 0.333·n^-2 | read A[i2 - 1, i3] (i0=0, i1=0); read A[i2 + 1, i3] (i0=0, i1=0) (+9) |
| n^2 | 2 | level | (1/4)·n^2 + 2·n + (7/4) | 4·n | 0.333·n^-2 | read A[i2 - 1, i3] (i0=0, i1=0); read A[i2 + 1, i3] (i0=0, i1=0) (+9) |
| n^2 | 1.25 | level | 1 | (5/4)·n^2 + (-5/4)·n - 5 | 0.104/n | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0, i4=0) (+1) |
| n^2 | 1 | level | (1/4)·n^2 + n - 2 | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0, i2=1, i3=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + (3/4)·n | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0, i2=0, i3=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + (3/2)·n + (1/4) | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0, i2=0, i3=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + n - 4 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0, i3=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 2 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0, i3=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + n - 3 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0, i2=1, i3=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 1 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0, i2=1, i3=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + (3/4)·n - 2 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0, i2=0, i3=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + (3/2)·n + (-7/4) | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0, i2=0, i3=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + n - 3 | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + n - 4 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + (3/4)·n - 4 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + n - 5 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0, i2=0); read B[i4 + 1, i5] (i0=0, i1=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + n - 1 | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0, i3=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + n | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n + 1 | 2·n | 0.167·n^-2 | write B[i2, i3] (i0=0, i1=0); write A[i4, i5] (i0=0, i1=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + (9/4)·n - 3 | 2·n | 0.167·n^-2 | read A[i2 - 1, i3] (i0=0, i1=0); read B[i4 - 1, i5] (i0=0, i1=0, i4=0) (+3) |
| n^2 | 1 | level | (1/4)·n^2 + (9/4)·n - 3 | 2·n | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i1=0); read B[i4 + 1, i5] (i0=0, i1=0) (+4) |
| n^2 | 1 | level | (1/4)·n^2 + (9/4)·n - 2 | 2·n - 2 | 0.167·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0, i4=0) (+2) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/4)·n - 3 | n | 0.0833·n^-2 | write B[i2, i3] (i0=0, i1=0); write B[i2, i3] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/4)·n - 3 | n | 0.0833·n^-2 | write A[i4, i5] (i0=0, i1=0); write A[i4, i5] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + n - 3 | n - 1 | 0.0833·n^-2 | read A[i2, i3 + 1] (i0=0, i1=1, i2=0); read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + n - 3 | n - 1 | 0.0833·n^-2 | read B[i4, i5 + 1] (i0=0, i1=1, i4=0); read B[i4, i5 + 1] (i0=0, i4=0) |
| n^2 | 0.354 | level | 2 | (1/4)·n^2 + (-1/2)·n + (1/4) | 0.0208/n | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (7/4)·n - 3 | 0.0208/n | read A[i2, i3] (i0=0, i1=0, i2=0, i3=0); read A[i2, i3 - 1] (i0=0, i1=0, i3=0) (+8) |
| n^2 | 0.125 | level | (1/4)·n^2 + n - 2 | (1/4)·n - 4 | 0.0208·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0); read B[i4, i5 + 1] (i0=0, i1=0) |
| n^2 | 0.125 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/4)·n + (-9/4) | 0.0208·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0); read B[i4, i5 + 1] (i0=0, i1=0) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read B[i4, i5 + 1] (i0=0, i1=0) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 + (-5/4)·n + (9/8) | 0.0104/n | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^1.5 | 1.41 | level | (1/2)·n - 1 | 2·n - 4 | 0.167·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (3/4)·n - 4 | 1 | 0.0833·n^-3 | read A[i2 + 1, i3] (i0=0, i1=1) |
| n^1 | 0.5 | level | (1/4)·n^2 + n - 3 | 1 | 0.0833·n^-3 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + n - 3 | 1 | 0.0833·n^-3 | read B[i4, i5 + 1] (i0=0, i1=0, i4=0) |
| n^0 | 5 | level | 1 | 5 | 0.417·n^-3 | read B[i4, i5 + 1] (i0=0, i1=0, i4=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.0833·n^-3 | read B[i4, i5 + 1] (i0=0, i1=0, i4=0) |
| n^0 | 1 | level | 1 | 1 | 0.0833·n^-3 | read B[i4, i5 + 1] (i0=0, i1=0, i4=0) |

Two-array 2-D stencil: cross-sweep plane reuses at (1/4)n^2 + O(n) lines (four symmetric write/read families, combined 0.5·n^4, headroom +1.0), row-window reuses at (3/8)n lines at order n^3.5. The plane boundary 64·((1/4)n^2) bytes = both arrays is the time-tiling cliff.

## jacobi-2d — single-shot  [`exact`]

Accesses $A(n) = 12·n^3 - 24·n^2 + 12·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.5·n^4  +  0.707·n^3.5  +  28.9·n^3  +  8.49·n^2.5  +  27·n^2

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.125 | level | (1/4)·n^2 + n - 2 | (1/4)·n^3 + (-43/8)·n^2 + (181/8)·n - 10 | 0.0208 | write B[i2, i3] (i0=0, i2=1); write B[i2, i3] (i0=0) (+1) |
| n^4 | 0.125 | level | (1/4)·n^2 + (7/4)·n | (1/4)·n^3 + (-29/8)·n^2 + 13·n + (-45/8) | 0.0208 | write B[i2, i3] (i0=0, i2=1); write B[i2, i3] (i0=0) (+1) |
| n^4 | 0.125 | level | (1/4)·n^2 + n - 3 | (1/4)·n^3 + (-43/8)·n^2 + (181/8)·n - 10 | 0.0208 | read A[i2 + 1, i3] (i0=0, i2=1); read A[i2 + 1, i3] (i0=0) (+1) |
| n^4 | 0.125 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/4)·n^3 + (-29/8)·n^2 + 13·n + (-45/8) | 0.0208 | read A[i2 + 1, i3] (i0=0, i2=1); read A[i2 + 1, i3] (i0=0) (+1) |
| n^3.5 | 0.177 | level | (1/2)·n + (5/2) | (1/4)·n^3 + (-19/4)·n^2 + (17/2)·n | 0.0208 | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^3.5 | 0.177 | level | (1/2)·n | (1/4)·n^3 + (-9/2)·n^2 + 8·n | 0.0208 | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^3.5 | 0.177 | level | (1/2)·n + 1 | (1/4)·n^3 + (-9/2)·n^2 + 8·n | 0.0208 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3.5 | 0.177 | level | (1/2)·n + (7/2) | (1/4)·n^3 + (-11/4)·n^2 + (9/2)·n | 0.0208 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 6 | level | 4 | 3·n^3 - 7·n^2 + 4·n | 0.25 | read A[i2 + 1, i3] (i0=0); write B[i2, i3] (i0=0) (+2) |
| n^3 | 4 | level | 4 | 2·n^3 - 6·n^2 + 4·n | 0.167 | read A[i2, i3] (i0=0); read B[i4, i5] (i0=0) |
| n^3 | 3 | level | 4 | (3/2)·n^3 + (-7/2)·n^2 + 2·n | 0.125 | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (-7/4)·n^2 | 0.146 | read A[i2, i3 - 1] (i0=0); read B[i4, i5 - 1] (i0=0) |
| n^3 | 1.68 | level | 5 | (3/4)·n^3 + (-11/4)·n^2 + 2·n | 0.0625 | read A[i2, i3 - 1] (i0=0); read A[i2 + 1, i3] (i0=0) (+4) |
| n^3 | 1.5 | level | 1 | (3/2)·n^3 + (-3/2)·n^2 | 0.125 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 1 | level | (1/4)·n^2 + n - 2 | 2·n^2 - 9·n + 4 | 0.167/n | read A[i2, i3] (i0=0, i2=0, i3=0); write B[i2, i3] (i0=0, i2=2, i3=0) (+3) |
| n^3 | 1 | level | (1/4)·n^2 + n - 3 | 2·n^2 - 9·n + 4 | 0.167/n | read A[i2 + 1, i3] (i0=0, i2=2, i3=0); read A[i2 + 1, i3] (i0=0, i3=0) (+3) |
| n^3 | 1 | level | (1/4)·n^2 + (7/4)·n | 2·n^2 - 11·n + 5 | 0.167/n | write B[i2, i3] (i0=0, i2=2, i3=0); write B[i2, i3] (i0=0, i3=0) (+1) |
| n^3 | 1 | level | (1/4)·n^2 + (7/4)·n - 1 | 2·n^2 - 7·n + 3 | 0.167/n | read A[i2, i3] (i0=0, i2=0, i3=0); read A[i2 + 1, i3] (i0=0, i2=2, i3=0) (+5) |
| n^3 | 1 | level | (1/4)·n^2 + n - 3 | 2·n^2 - 9·n + 4 | 0.167/n | write B[i2, i3] (i0=0, i2=0); write B[i2, i3] (i0=0) (+1) |
| n^3 | 1 | level | (1/4)·n^2 + n - 4 | 2·n^2 - 11·n + 5 | 0.167/n | read A[i2 + 1, i3] (i0=0, i2=1); read A[i2 + 1, i3] (i0=0) (+1) |
| n^3 | 1 | level | (1/4)·n^2 + (9/4)·n - 2 | 2·n^2 - 4·n + 2 | 0.167/n | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0, i4=0) (+2) |
| n^3 | 0.559 | level | 5 | (1/4)·n^3 + (-1/2)·n^2 + (1/4)·n | 0.0208 | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^3 | 0.559 | level | 5 | (1/4)·n^3 + (-1/4)·n^2 | 0.0208 | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^3 | 0.354 | level | 2 | (1/4)·n^3 + (-1/2)·n^2 + (1/4)·n | 0.0208 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 0.354 | level | 2 | (1/4)·n^3 + (-9/4)·n^2 + 2·n | 0.0208 | read A[i2, i3 + 1] (i0=0); read B[i4, i5 + 1] (i0=0, i4=0) (+1) |
| n^3 | 0.25 | level | (1/4)·n^2 + (7/4)·n | (1/2)·n^2 + (-27/4)·n + (13/4) | 0.0417/n | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 2 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 3 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/4)·n^2 + (-19/8)·n + (9/8) | 0.0208/n | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 1 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + (9/4)·n - 2 | (1/4)·n^2 + (-17/4)·n + 4 | 0.0208/n | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0, i4=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + 2·n + (7/4) | (1/4)·n^2 + (-5/2)·n + (9/4) | 0.0208/n | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0, i4=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + (7/4)·n + (3/2) | (1/4)·n^2 + (-11/4)·n + (5/2) | 0.0208/n | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0, i4=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + (9/4)·n - 2 | (1/4)·n^2 + (-17/4)·n + 4 | 0.0208/n | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + 2·n + (7/4) | (1/4)·n^2 + (-5/2)·n + (9/4) | 0.0208/n | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + n - 2 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | read A[i2, i3 + 1] (i0=0, i2=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 0.125 | level | (1/4)·n^2 + (7/4)·n - 1 | (1/4)·n^2 + (-19/8)·n + (9/8) | 0.0208/n | read A[i2, i3 + 1] (i0=0, i2=0); read B[i4, i5 + 1] (i0=0) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n + 2  →  (1/4)·n^2 + n - 4 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | write B[i2, i3] (i0=0, i2=0); write A[i4, i5] (i0=0) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n + 1  →  (1/4)·n^2 + n - 5 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n  →  (1/4)·n^2 + n - 6 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | read A[i2 + 1, i3] (i0=0, i2=0); read B[i4 + 1, i5] (i0=0) |
| n^3 | 0.119 | ramp | (1/4)·n^2 + (3/4)·n  →  (1/4)·n^2 + n - 6 | (1/4)·n^2 + (-33/8)·n + 2 | 0.0208/n | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^2.5 | 1.41 | level | (1/2)·n | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0, i5=0) |
| n^2.5 | 1.41 | level | (1/2)·n + 1 | 2·n^2 - 4·n | 0.167/n | read A[i2, i3] (i0=0); read B[i4, i5] (i0=0, i5=0) |
| n^2.5 | 1.41 | level | (1/2)·n + (5/2) | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^2.5 | 1.41 | level | (1/2)·n + 2 | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^2.5 | 1.41 | level | (1/2)·n + (9/2) | 2·n^2 - 4·n | 0.167/n | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0) |
| n^2.5 | 1.41 | level | (1/2)·n - 1 | 2·n^2 - 4·n | 0.167/n | read A[i2, i3 + 1] (i0=0, i1=0); read B[i4, i5 + 1] (i0=0, i1=0) (+2) |
| n^2 | 2 | level | (1/4)·n^2 + (7/4)·n | 4·n - 2 | 0.333·n^-2 | write B[i2, i3] (i0=0, i2=1, i3=0); write B[i2, i3] (i0=0, i3=0) (+1) |
| n^2 | 2 | level | (1/4)·n^2 + (9/4)·n - 2 | 4·n - 4 | 0.333·n^-2 | read A[i2 - 1, i3] (i0=0); read A[i2 + 1, i3] (i0=0) (+2) |
| n^2 | 2 | level | (1/4)·n^2 + 2·n + (7/4) | 4·n - 4 | 0.333·n^-2 | read A[i2 - 1, i3] (i0=0); read A[i2 + 1, i3] (i0=0) (+2) |
| n^2 | 1 | level | (1/4)·n^2 + n - 2 | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0, i2=1, i3=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (3/4)·n | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0, i2=0, i3=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (3/2)·n + (1/4) | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0, i2=0, i3=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + n - 4 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i3=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 2 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i3=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + n - 3 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i2=1, i3=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n - 1 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i2=1, i3=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (3/4)·n - 2 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i2=0, i3=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (3/2)·n + (-7/4) | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i2=0, i3=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + n - 3 | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + n - 4 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (3/4)·n - 4 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + n - 5 | 2·n - 1 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0, i2=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + n - 1 | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0, i3=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + n | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (7/4)·n + 1 | 2·n - 1 | 0.167·n^-2 | write B[i2, i3] (i0=0); write A[i4, i5] (i0=0) |
| n^2 | 1 | level | (1/4)·n^2 + (9/4)·n - 3 | 2·n - 2 | 0.167·n^-2 | read A[i2 - 1, i3] (i0=0); read B[i4 - 1, i5] (i0=0, i4=0) |
| n^2 | 1 | level | (1/4)·n^2 + (9/4)·n - 3 | 2·n - 2 | 0.167·n^-2 | read A[i2 + 1, i3] (i0=0); read B[i4 + 1, i5] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/4)·n - 3 | n - 1 | 0.0833·n^-2 | write B[i2, i3] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + (3/4)·n - 3 | n | 0.0833·n^-2 | write A[i4, i5] (i0=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + n - 3 | n - 1 | 0.0833·n^-2 | read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.5 | level | (1/4)·n^2 + n - 3 | n | 0.0833·n^-2 | read B[i4, i5 + 1] (i0=0, i1=0, i4=0); read B[i4, i5 + 1] (i0=0, i4=0) |

Two-array 2-D stencil: cross-sweep plane reuses at (1/4)n^2 + O(n) lines (four symmetric write/read families, combined 0.5·n^4, headroom +1.0), row-window reuses at (3/8)n lines at order n^3.5. The plane boundary 64·((1/4)n^2) bytes = both arrays is the time-tiling cliff.

## lu — infinite-repeat  [`exact`]

Accesses $A(n) = (4/3)·n^3 + (-1/2)·n^2 + (-5/6)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.00643·n^4  +  0.22·n^3.5  +  2.33·n^3  +  6.59·n^2.5  +  39.1·n^2  +  24.9·n^1.5  +  254·n^1  +  546·n^0.5  +  64.3·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.00309 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (7/384)·n^3 + (-123/64)·n^2 + (1619/24)·n - 790 | 0.0137 | read A[i3, i2] (i0=0) |
| n^4 | 0.00193 | ramp | (11/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 91 | (7/512)·n^3 + (-195/128)·n^2 + (901/16)·n - 690 | 0.0103 | read A[i5, i4] (i0=0) |
| n^4 | 0.000411 | ramp | (25/4)·n - 121  →  (1/16)·n^2 + (3/4)·n - 4 | (1/384)·n^3 + (-21/64)·n^2 + (329/24)·n - 190 | 0.00195 | read A[i3, i2] (i0=0) |
| n^4 | 0.000316 | ramp | 3·n - 16  →  (1/16)·n^2 + (3/4)·n - 86 | (7/3072)·n^3 + (-35/128)·n^2 + (259/24)·n - 140 | 0.00171 | read A[i5, i4] (i0=0) |
| n^4 | 0.00031 | ramp | (17/8)·n - 2  →  (1/16)·n^2 + (3/4)·n - 121 | (7/3072)·n^3 + (-35/128)·n^2 + (259/24)·n - 140 | 0.00171 | read A[i5, i4] (i0=0) |
| n^4 | 0.000268 | ramp | (9/4)·n - 4  →  (1/16)·n^2 + (3/4)·n - 91 | (1/512)·n^3 + (-15/64)·n^2 + (37/4)·n - 120 | 0.00146 | read A[i5, i4] (i0=0) |
| n^4 | 6.23e-05 | ramp | 2·n - 2  →  (1/16)·n^2 + (3/4)·n - 86 | (1/3072)·n^3 + (-55/48)·n + 25 | 0.000244 | read A[i5, i4] (i0=0) |
| n^4 | 4.1e-05 | ramp | (25/8)·n - 17  →  (1/16)·n^2 + (3/4)·n - 121 | (1/3072)·n^3 + (-3/64)·n^2 + (107/48)·n - 35 | 0.000244 | read A[i5, i4] (i0=0) |
| n^3.5 | 0.0988 | ramp | 11  →  (9/8)·n - 2 | (7/48)·n^3 + (-63/16)·n^2 + (217/6)·n - 112 | 0.109 | read A[i3, i2] (i0=0) |
| n^3.5 | 0.0847 | ramp | 6  →  (9/8)·n - 8 | (49/384)·n^3 + (-231/64)·n^2 + (203/6)·n - 105 | 0.0957 | read A[i5, i4] (i0=0) |
| n^3.5 | 0.012 | ramp | 12  →  (9/8)·n - 15 | (7/384)·n^3 + (-35/64)·n^2 + (119/24)·n - 14 | 0.0137 | read A[i5, i4] (i0=0) |
| n^3.5 | 0.011 | ramp | 22  →  (9/8)·n - 9 | (7/384)·n^3 + (-147/128)·n^2 + (1141/48)·n - 161 | 0.0137 | read A[i1, i3] (i0=0) |
| n^3.5 | 0.0108 | ramp | 14  →  (9/8)·n - 16 | (7/384)·n^3 + (-35/32)·n^2 + (259/12)·n - 140 | 0.0137 | read A[i1, i5] (i0=0) |
| n^3.5 | 0.00162 | ramp | 21  →  (9/8)·n - 15 | (1/384)·n^3 + (-17/128)·n^2 + (103/48)·n - 11 | 0.00195 | read A[i1, i3] (i0=0) |
| n^3.5 | 0.00148 | ramp | 15  →  (9/8)·n - 23 | (1/384)·n^3 + (-3/16)·n^2 + (13/3)·n - 32 | 0.00195 | read A[i1, i5] (i0=0) |
| n^3 | 0.253 | level | 3 | (7/48)·n^3 + (-21/32)·n^2 + (-581/24)·n + 161 | 0.109 | write A[i1, i4] (i0=0) |
| n^3 | 0.253 | level | 3 | (7/48)·n^3 + (-23/32)·n^2 + (-713/24)·n + 223 | 0.109 | read A[i1, i5] (i0=0, i4=0); read A[i1, i5] (i0=0) |
| n^3 | 0.221 | level | 3 | (49/384)·n^3 + (-469/128)·n^2 + (1687/48)·n - 112 | 0.0957 | read A[i1, i3] (i0=0) |
| n^3 | 0.189 | level | 3 | (7/64)·n^3 + (-399/128)·n^2 + (483/16)·n - 98 | 0.082 | write A[i1, i2] (i0=0) |
| n^3 | 0.167 | level | 1 | (1/6)·n^3 + (41/6)·n - 2 | 0.125 | read A[i1, i2] (i0=0, i1=1); read A[i1, i3] (i0=0) (+6) |
| n^3 | 0.146 | level | 1 | (7/48)·n^3 + (-7/16)·n^2 + (-35/6)·n + 35 | 0.109 | read A[i1, i2] (i0=0, i3=0); read A[i1, i2] (i0=0) |
| n^3 | 0.0737 | ramp | (3/2)·n - 3  →  (1/16)·n^2 + (3/4)·n - 31 | (49/128)·n^2 + (-315/16)·n + 253 | 0.287/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0732 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (49/128)·n^2 + (-371/16)·n + 351 | 0.287/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0731 | ramp | (13/4)·n - 23  →  (1/16)·n^2 + (3/4)·n - 11 | (49/128)·n^2 + (-357/16)·n + 322 | 0.287/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0711 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (3/8)·n^2 + (-47/2)·n + 370 | 0.281/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0598 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (5/16)·n^2 + (-75/4)·n + 280 | 0.234/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0595 | ramp | (11/8)·n + 2  →  (1/16)·n^2 + (3/4)·n - 57 | (21/64)·n^2 + (-87/4)·n + 360 | 0.246/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0538 | ramp | (7/2)·n - 31  →  (1/16)·n^2 + (3/4)·n - 4 | (35/128)·n^2 + (-225/16)·n + 180 | 0.205/n | read A[i2, i2] (i0=0) |
| n^3 | 0.0516 | ramp | (19/8)·n - 8  →  (1/16)·n^2 + (3/4)·n - 31 | (35/128)·n^2 + (-255/16)·n + 230 | 0.205/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0379 | ramp | (1/2)·n + 4  →  (1/16)·n^2 + (3/4)·n - 91 | (9/32)·n^2 - 19·n + 320 | 0.211/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0361 | level | 3 | (1/48)·n^3 + (5/32)·n^2 + (-83/24)·n + 14 | 0.0156 | write A[i1, i4] (i0=0, i5=0); write A[i1, i4] (i0=0) |
| n^3 | 0.0316 | level | 3 | (7/384)·n^3 + (-15/128)·n^2 + (-5/48)·n - 1 | 0.0137 | read A[i1, i3] (i0=0) |
| n^3 | 0.0316 | level | 3 | (7/384)·n^3 + (-63/128)·n^2 + (175/48)·n - 7 | 0.0137 | write A[i1, i2] (i0=0) |
| n^3 | 0.0316 | level | 3 | (7/384)·n^3 + (-3/64)·n^2 + (-13/24)·n - 2 | 0.0137 | write A[i1, i2] (i0=0) |
| n^3 | 0.0271 | level | 3 | (1/64)·n^3 + (-33/128)·n^2 + (17/16)·n | 0.0117 | write A[i1, i2] (i0=0) |
| n^3 | 0.0208 | level | 1 | (1/48)·n^3 + (-1/8)·n^2 + (-5/24)·n - 1 | 0.0156 | read A[i1, i2] (i0=0) |
| n^3 | 0.0193 | level | (1/8)·n^2 | (7/128)·n^2 + (-53/16)·n + 50 | 0.041/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.0127 | ramp | (3/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 91 | (3/32)·n^2 + (-49/8)·n + 100 | 0.0703/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0119 | ramp | 3·n - 18  →  (1/16)·n^2 + (3/4)·n - 28 | (1/16)·n^2 + (-29/8)·n + 52 | 0.0469/n | read A[i5, i4] (i0=0, i4=0) |
| n^3 | 0.0113 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-19/4)·n + 90 | 0.0469/n | read A[i3, i2] (i0=0, i3=7) |
| n^3 | 0.0113 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-19/4)·n + 90 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0113 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-19/4)·n + 90 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0112 | ramp | (29/8)·n - 37  →  (1/8)·n^2 + (-15/4)·n + 68 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i1, i4] (i0=0, i5=0) |
| n^3 | 0.011 | ramp | (27/8)·n - 28  →  (1/16)·n^2 + (3/4)·n - 4 | (7/128)·n^2 + (-39/16)·n + 27 | 0.041/n | read A[i2, i2] (i0=0) |
| n^3 | 0.0105 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i2, i2] (i0=0) |
| n^3 | 0.0105 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0104 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 25 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0103 | ramp | (9/4)·n - 9  →  (1/16)·n^2 + (3/4)·n - 49 | (7/128)·n^2 + (-49/16)·n + 42 | 0.041/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0103 | ramp | (19/8)·n - 12  →  (1/16)·n^2 + (3/4)·n - 32 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0103 | ramp | (9/4)·n - 6  →  (1/16)·n^2 + (3/4)·n - 46 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i5, i4] (i0=0, i4=6) |
| n^3 | 0.0103 | ramp | (17/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 49 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i5, i4] (i0=0, i4=7) |
| n^3 | 0.01 | ramp | (21/4)·n - 80  →  (1/16)·n^2 + (3/4)·n - 3 | (7/128)·n^2 + (-65/16)·n + 75 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00999 | ramp | (17/4)·n - 47  →  (1/16)·n^2 + (3/4)·n - 11 | (7/128)·n^2 + (-63/16)·n + 70 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00997 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (7/128)·n^2 + (-65/16)·n + 75 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0098 | ramp | 3·n - 17  →  (1/16)·n^2 + (3/4)·n - 53 | (7/128)·n^2 + (-63/16)·n + 70 | 0.041/n | read A[i5, i4] (i0=0, i4=8) |
| n^3 | 0.00969 | ramp | (17/8)·n - 3  →  (1/16)·n^2 + (3/4)·n - 81 | (7/128)·n^2 + (-63/16)·n + 70 | 0.041/n | read A[i5, i4] (i0=0, i4=15) |
| n^3 | 0.00902 | level | 3 | (1/192)·n^3 + (23/64)·n^2 + (113/12)·n - 97 | 0.00391 | read A[i1, i5] (i0=0, i1=1); read A[i1, i2] (i0=0, i1=2, i2=0) (+9) |
| n^3 | 0.00901 | ramp | (35/8)·n - 54  →  (1/16)·n^2 + (3/4)·n - 5 | (3/64)·n^2 + (-11/4)·n + 40 | 0.0352/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00835 | ramp | (9/4)·n - 5  →  (1/16)·n^2 + (3/4)·n - 57 | (3/64)·n^2 + (-27/8)·n + 60 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00754 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (3/4)·n - 11 | (5/128)·n^2 + (-35/16)·n + 30 | 0.0293/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00705 | ramp | (27/8)·n - 25  →  (1/16)·n^2 + (3/4)·n - 31 | (5/128)·n^2 + (-45/16)·n + 50 | 0.0293/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00671 | ramp | (5/4)·n + 3  →  (1/16)·n^2 + (3/4)·n - 57 | (3/64)·n^2 + (-21/8)·n + 36 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00641 | ramp | 2·n - 2  →  (1/16)·n^2 + (3/4)·n - 86 | (3/64)·n^2 + (-27/8)·n + 60 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00615 | ramp | (9/8)·n + 5  →  (1/16)·n^2 + (3/4)·n - 121 | (3/64)·n^2 + (-27/8)·n + 60 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00561 | ramp | (11/8)·n + 2  →  (1/16)·n^2 + (3/4)·n - 57 | (5/128)·n^2 + (-35/16)·n + 30 | 0.0293/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00276 | level | (1/8)·n^2 + (7/8)·n | (1/128)·n^2 + (-37/64)·n + (1353/128) | 0.00586/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.00276 | level | (1/8)·n^2 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.00211 | ramp | 2·n - 2  →  (1/16)·n^2 + (3/4)·n - 86 | (1/64)·n^2 + (-5/4)·n + 25 | 0.0117/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^3 | 0.00203 | ramp | (9/8)·n + 5  →  (1/16)·n^2 + (3/4)·n - 121 | (1/64)·n^2 + (-5/4)·n + 25 | 0.0117/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^3 | 0.00176 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/128)·n^2 + (1/16)·n - 9 | 0.00586/n | read A[i5, i4] (i0=0, i4=8) |
| n^3 | 0.00152 | ramp | (35/8)·n - 55  →  (1/8)·n^2 + (-37/8)·n + 89 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i1, i4] (i0=0, i5=0) |
| n^3 | 0.00151 | ramp | (17/4)·n - 50  →  (1/16)·n^2 + (3/4)·n - 10 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00151 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (3/4)·n - 11 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00151 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (3/4)·n - 11 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00143 | ramp | (41/8)·n - 75  →  (1/16)·n^2 + (3/4)·n - 9 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00143 | ramp | (41/8)·n - 77  →  (1/16)·n^2 + (3/4)·n - 11 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00142 | ramp | (33/8)·n - 43  →  (1/16)·n^2 + (3/4)·n - 25 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0014 | ramp | (13/4)·n - 22  →  (1/16)·n^2 + (3/4)·n - 46 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0, i4=6) |
| n^3 | 0.0014 | ramp | (13/4)·n - 26  →  (1/16)·n^2 + (3/4)·n - 50 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0014 | ramp | (25/8)·n - 19  →  (1/16)·n^2 + (3/4)·n - 49 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0, i4=7) |
| n^3 | 0.00136 | ramp | (49/8)·n - 115  →  (1/16)·n^2 + (3/4)·n - 10 | (1/128)·n^2 + (-11/16)·n + 15 | 0.00586/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00131 | ramp | (25/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 81 | (1/128)·n^2 + (-11/16)·n + 15 | 0.00586/n | read A[i5, i4] (i0=0, i4=15) |
| n^3 | 0.00114 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00114 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0011 | ramp | (5/4)·n + 3  →  (1/16)·n^2 + (3/4)·n - 77 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0011 | ramp | (9/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 81 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00107 | ramp | (17/8)·n - 3  →  (1/16)·n^2 + (3/4)·n - 81 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00104 | ramp | 3·n - 16  →  (1/16)·n^2 + (3/4)·n - 86 | (1/128)·n^2 + (-11/16)·n + 15 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^2.5 | 1.53 | ramp | 5  →  (9/8)·n - 2 | (15/8)·n^2 + (-45/2)·n + 70 | 1.41/n | read A[i5, i4] (i0=0) |
| n^2.5 | 1.22 | ramp | 5  →  (9/8)·n - 2 | (35/16)·n^2 + (-235/8)·n + 105 | 1.64/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.315 | ramp | 5  →  (9/8)·n - 1 | (3/8)·n^2 + (-15/8)·n + 1 | 0.281/n | read A[i5, i4] (i0=0, i4=0) |
| n^2.5 | 0.298 | ramp | 6  →  (9/8)·n - 8 | (3/8)·n^2 + (-57/8)·n + 33 | 0.281/n | read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 0.277 | ramp | 14  →  (9/8)·n - 10 | (3/8)·n^2 - 15·n + 144 | 0.281/n | read A[i1, i5] (i0=0) |
| n^2.5 | 0.267 | ramp | 5  →  (9/8)·n - 9 | (1/2)·n^2 + (-21/2)·n + 55 | 0.375/n | read A[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.255 | ramp | 11  →  (9/8)·n - 7 | (5/16)·n^2 + (-15/4)·n + 10 | 0.234/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.246 | ramp | 4  →  (9/8)·n - 2 | (7/16)·n^2 + (-23/8)·n + 5 | 0.328/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.235 | ramp | 12  →  (9/8)·n - 15 | (7/16)·n^2 + (-85/8)·n + 58 | 0.328/n | read A[i1, i5] (i0=0) |
| n^2.5 | 0.223 | ramp | 5  →  (9/8)·n - 16 | (7/16)·n^2 - 14·n + 112 | 0.328/n | read A[i1, i5] (i0=0, i5=0) |
| n^2.5 | 0.211 | ramp | 3  →  (9/8)·n - 2 | (3/8)·n^2 + (-9/8)·n | 0.281/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.211 | ramp | 4  →  (9/8)·n - 2 | (3/8)·n^2 + (-17/8)·n + 3 | 0.281/n | read A[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.207 | ramp | 4  →  (9/8)·n - 8 | (49/128)·n^2 + (-91/16)·n + 21 | 0.287/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.205 | ramp | 5  →  (9/8)·n - 9 | (49/128)·n^2 + (-119/16)·n + 36 | 0.287/n | read A[i5, i4] (i0=0, i5=0) |
| n^2.5 | 0.198 | ramp | 14  →  (9/8)·n - 9 | (3/8)·n^2 + (-105/8)·n + 114 | 0.281/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.177 | ramp | 5  →  (9/8)·n - 8 | (21/64)·n^2 + (-21/4)·n + 21 | 0.246/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.051 | ramp | 11  →  (9/8)·n - 7 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 0.0508 | ramp | 11  →  (9/8)·n - 7 | (1/16)·n^2 + (-7/8)·n + 3 | 0.0469/n | read A[i5, i4] (i0=0, i4=0) |
| n^2.5 | 0.0466 | ramp | 21  →  (9/8)·n - 8 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0469/n | read A[i1, i5] (i0=0, i4=0) |
| n^2.5 | 0.0465 | ramp | 20  →  (9/8)·n - 9 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0469/n | read A[i1, i5] (i0=0, i4=1) |
| n^2.5 | 0.0351 | ramp | 11  →  (9/8)·n - 7 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | read A[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.0351 | ramp | 11  →  (9/8)·n - 7 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.0332 | ramp | 13  →  (9/8)·n - 14 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0469/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0331 | ramp | 11  →  (9/8)·n - 16 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0469/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0316 | ramp | 13  →  (9/8)·n - 23 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0469/n | read A[i1, i5] (i0=0) |
| n^2.5 | 0.031 | ramp | 6  →  (9/8)·n - 23 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0469/n | read A[i1, i5] (i0=0, i5=0) |
| n^2.5 | 0.0294 | ramp | 12  →  (9/8)·n - 15 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | read A[i5, i4] (i0=0, i5=0) |
| n^2.5 | 0.0294 | ramp | 12  →  (9/8)·n - 15 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.0294 | ramp | 12  →  (9/8)·n - 15 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.0277 | ramp | 13  →  (9/8)·n - 23 | (7/128)·n^2 + (-35/16)·n + 21 | 0.041/n | read A[i5, i4] (i0=0) |
| n^2 | 2.27 | level | 3 | (21/16)·n^2 + (-161/8)·n + 77 | 0.984/n | read A[i1, i3] (i0=0) |
| n^2 | 1.31 | level | 1 | (21/16)·n^2 + (-97/8)·n + 33 | 0.984/n | read A[i1, i3] (i0=0) |
| n^2 | 1.23 | ramp | (1/16)·n^2 + (5/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 3 | (21/4)·n - 132 | 3.94·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 1.08 | level | 3 | (5/8)·n^2 - 10·n + 40 | 0.469/n | write A[i1, i2] (i0=0) |
| n^2 | 1.03 | ramp | (1/16)·n^2 + (5/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 13 | (35/8)·n - 100 | 3.28·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 1.01 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (21/4)·n - 139 | 3.94·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.997 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (21/4)·n - 133 | 3.94·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.976 | ramp | (5/8)·n  →  (1/16)·n^2 + (3/4)·n - 30 | (21/4)·n - 117 | 3.94·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.972 | level | 2 | (11/16)·n^2 + (-19/8)·n | 0.516/n | write A[i1, i2] (i0=0) |
| n^2 | 0.964 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (5/8)·n | 5·n - 130 | 3.75·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.884 | level | 2 | (5/8)·n^2 + (-15/4)·n | 0.469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.85 | ramp | (5/2)·n - 13  →  (1/16)·n^2 + (5/8)·n - 1 | (35/8)·n - 95 | 3.28·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.847 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (35/8)·n - 110 | 3.28·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.833 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (35/8)·n - 105 | 3.28·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.807 | ramp | (1/2)·n + 3  →  (1/16)·n^2 + (3/4)·n - 57 | (9/2)·n - 134 | 3.38·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-63/8)·n + 35 | 0.328/n | read A[i1, i3] (i0=0) |
| n^2 | 0.729 | ramp | (21/8)·n - 14  →  (1/16)·n^2 + (3/4)·n - 3 | (15/4)·n - 84 | 2.81·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.695 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (15/4)·n - 95 | 2.81·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.663 | level | 3 | (49/128)·n^2 + (-105/16)·n + 28 | 0.287/n | write A[i1, i4] (i0=0, i4=0) |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-57/8)·n + 33 | 0.281/n | write A[i1, i2] (i0=0) |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-7/8)·n | 0.328/n | write A[i1, i2] (i0=0) |
| n^2 | 0.578 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (5/8)·n | 3·n - 78 | 2.25·n^-2 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^2 | 0.541 | level | 3 | (5/16)·n^2 + (-45/8)·n + 25 | 0.234/n | write A[i1, i2] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 + (-15/8)·n + 2 | 0.328/n | read A[i1, i3] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 + (-7/4)·n - 14 | 0.328/n | read A[i1, i4] (i0=0, i1=1, i5=0); read A[i1, i4] (i0=0, i5=0) |
| n^2 | 0.38 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (5/8)·n + 1 | 2·n - 68 | 1.5·n^-2 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^2 | 0.365 | ramp | (7/4)·n - 3  →  (1/16)·n^2 + (3/4)·n - 5 | (15/8)·n - 30 | 1.41·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.354 | level | 2 | (1/4)·n^2 + (-3/4)·n | 0.188/n | write A[i1, i2] (i0=0) |
| n^2 | 0.342 | ramp | (9/4)·n - 9  →  (1/16)·n^2 + (5/8)·n - 1 | (7/4)·n - 31 | 1.31·n^-2 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^2 | 0.309 | level | (1/8)·n^2 | (7/8)·n - 16 | 0.656·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.309 | level | (1/8)·n^2 | (7/8)·n - 23 | 0.656·n^-2 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^2 | 0.269 | ramp | (3/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 57 | (3/2)·n - 43 | 1.12·n^-2 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.265 | level | (1/8)·n^2 | (3/4)·n - 24 | 0.562·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.233 | ramp | (3/2)·n  →  (1/16)·n^2 + (3/4)·n - 31 | (5/4)·n - 30 | 0.938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.231 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/4)·n - 35 | 0.938·n^-2 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.231 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/4)·n - 35 | 0.938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.223 | ramp | (19/8)·n - 12  →  (1/8)·n^2 + (-15/8)·n + 16 | n - 24 | 0.75·n^-2 | read A[i1, i4] (i0=0, i5=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-11/8)·n + 3 | 0.0938/n | write A[i1, i2] (i0=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-5/4)·n + 2 | 0.0938/n | write A[i1, i2] (i0=0) |
| n^2 | 0.205 | ramp | (1/16)·n^2 + (5/8)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 21 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.205 | ramp | (1/16)·n^2 + (5/8)·n - 19  →  (1/16)·n^2 + (3/4)·n - 23 | (7/8)·n - 21 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.205 | ramp | (1/16)·n^2 + (5/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 25 | (7/8)·n - 21 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.205 | ramp | (1/16)·n^2 + (5/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 9 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.204 | ramp | (1/16)·n^2 + (5/8)·n - 24  →  (1/16)·n^2 + (3/4)·n - 28 | (7/8)·n - 22 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.193 | ramp | (21/8)·n - 17  →  (1/8)·n^2 + (-11/4)·n + 37 | (7/8)·n - 22 | 0.656·n^-2 | read A[i1, i4] (i0=0, i5=0) |
| n^2 | 0.171 | ramp | (1/16)·n^2 + (5/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 4 | (3/4)·n - 24 | 0.562·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (5/8)·n | (7/8)·n - 22 | 0.656·n^-2 | read A[i2, i2] (i0=0, i2=8) |
| n^2 | 0.167 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 29 | 0.656·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.167 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 29 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.167 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 29 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.167 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 21 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.167 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 21 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.167 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 21 | 0.656·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.167 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 21 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.165 | ramp | (13/4)·n - 23  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 28 | 0.656·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.165 | ramp | (13/4)·n - 23  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 28 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.165 | ramp | (13/4)·n - 23  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 28 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.163 | ramp | (1/2)·n  →  (1/16)·n^2 + (3/4)·n - 30 | (7/8)·n - 16 | 0.656·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^2 | 0.162 | ramp | (11/8)·n - 2  →  (1/16)·n^2 + (3/4)·n - 31 | (7/8)·n - 22 | 0.656·n^-2 | read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.16 | ramp | (1/16)·n^2 + (5/8)·n - 3  →  (1/8)·n^2 - 3·n + 50 | (5/8)·n - 20 | 0.469·n^-2 | read A[i1, i4] (i0=0, i5=0) |
| n^2 | 0.147 | ramp | (5/2)·n - 12  →  (1/16)·n^2 + (3/4)·n - 3 | (3/4)·n - 14 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.147 | ramp | (1/16)·n^2 + (5/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 14 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.145 | ramp | (27/8)·n - 27  →  (1/16)·n^2 + (3/4)·n - 3 | (3/4)·n - 19 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.145 | ramp | (27/8)·n - 28  →  (1/16)·n^2 + (3/4)·n - 4 | (3/4)·n - 19 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.145 | ramp | (27/8)·n - 29  →  (1/16)·n^2 + (5/8)·n - 1 | (3/4)·n - 19 | 0.562·n^-2 | read A[i2, i2] (i0=0, i2=7) |
| n^2 | 0.144 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (3/4)·n - 19 | 0.562·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.142 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (3/4)·n - 25 | 0.562·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.141 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (3/4)·n - 19 | 0.562·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.141 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (3/4)·n - 18 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.139 | ramp | (11/8)·n - 2  →  (1/16)·n^2 + (5/8)·n - 27 | (3/4)·n - 18 | 0.562·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.138 | ramp | (5/4)·n  →  (1/16)·n^2 + (3/4)·n - 48 | (3/4)·n - 18 | 0.562·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.138 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (3/4)·n - 19 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.137 | ramp | (9/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 49 | (3/4)·n - 19 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.136 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (3/4)·n - 24 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.133 | ramp | (9/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 81 | (3/4)·n - 24 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=15) |
| n^2 | 0.129 | ramp | (1/16)·n^2 + (5/8)·n + 13  →  (1/8)·n^2 + (-17/8)·n + 29 | (1/2)·n - 16 | 0.375·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.123 | ramp | (19/8)·n - 10  →  (1/16)·n^2 + (3/4)·n - 3 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.123 | ramp | (9/4)·n - 8  →  (1/16)·n^2 + (3/4)·n - 4 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.123 | ramp | (5/2)·n - 12  →  (1/16)·n^2 + (3/4)·n - 3 | (5/8)·n - 11 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.121 | ramp | (13/8)·n - 2  →  (1/16)·n^2 + (3/4)·n - 8 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.121 | ramp | (3/2)·n - 1  →  (1/16)·n^2 + (3/4)·n - 11 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.12 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (5/8)·n - 15 | 0.469·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.12 | ramp | (25/8)·n - 22  →  (1/16)·n^2 + (3/4)·n - 10 | (5/8)·n - 15 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.12 | ramp | (25/8)·n - 23  →  (1/16)·n^2 + (5/8)·n - 7 | (5/8)·n - 15 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.119 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (5/8)·n - 20 | 0.469·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.118 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (5/8)·n - 15 | 0.469·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.117 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.116 | ramp | (3/2)·n - 3  →  (1/16)·n^2 + (5/8)·n - 27 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.116 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.116 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.115 | ramp | (19/8)·n - 8  →  (1/16)·n^2 + (3/4)·n - 31 | (5/8)·n - 20 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0469/n | read A[i1, i3] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0469/n | write A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0966 | ramp | (13/8)·n - 2  →  (1/16)·n^2 + (3/4)·n - 11 | (1/2)·n - 8 | 0.375·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0947 | level | 3 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | write A[i1, i4] (i0=0, i4=0) |
| n^2 | 0.0947 | level | 3 | (7/128)·n^2 + (-35/16)·n + 21 | 0.041/n | write A[i1, i4] (i0=0, i4=0) |
| n^2 | 0.094 | ramp | 23  →  (1/16)·n^2 + (-3/8)·n + 1 | (3/4)·n - 12 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-1/2)·n | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (1/8)·n | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (3/8)·n | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0716 | ramp | (21/8)·n - 13  →  (1/16)·n^2 + (3/4)·n - 13 | (3/8)·n - 9 | 0.281·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0714 | ramp | (5/2)·n - 12  →  (1/16)·n^2 + (3/4)·n - 16 | (3/8)·n - 9 | 0.281·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0705 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (3/8)·n - 9 | 0.281·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.048 | ramp | (25/8)·n - 23  →  (1/16)·n^2 + (5/8)·n - 7 | (1/4)·n - 6 | 0.188·n^-2 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^2 | 0.047 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (1/4)·n - 6 | 0.188·n^-2 | read A[i3, i2] (i0=0, i3=0); read A[i2, i2] (i0=0) |
| n^2 | 0.0466 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/4)·n - 7 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0); read A[i5, i4] (i0=0, i4=0, i5=7) |
| n^2 | 0.0457 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (1/4)·n - 7 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6, i5=7) |
| n^2 | 0.0456 | ramp | (9/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 49 | (1/4)·n - 7 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=7, i5=0); read A[i5, i4] (i0=0, i4=7, i5=7) |
| n^2 | 0.045 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/4)·n - 9 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=8, i5=0); read A[i5, i4] (i0=0, i4=8, i5=7) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n + (-25/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n + (-33/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n + (-33/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n + (-33/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (1/8)·n | (1/8)·n + (-39/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n + (-33/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n + (-25/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n + (-33/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (3/4)·n | (1/8)·n + (-17/4) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 + (5/8)·n | (1/8)·n + (-35/8) | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.044 | ramp | (9/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 81 | (1/4)·n - 9 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=15, i5=0); read A[i5, i4] (i0=0, i4=15, i5=7) |
| n^2 | 0.033 | ramp | (1/16)·n^2 + (5/8)·n  →  (1/8)·n^2 + (-11/4)·n + 44 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^2 | 0.0324 | ramp | (1/16)·n^2 + (5/8)·n + 23  →  (1/8)·n^2 + (-15/8)·n + 23 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0324 | ramp | (1/16)·n^2 + (5/8)·n + 21  →  (1/8)·n^2 - 2·n + 26 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0322 | ramp | (1/16)·n^2 + (5/8)·n + 11  →  (1/8)·n^2 + (-21/8)·n + 41 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0321 | ramp | (1/16)·n^2 + (5/8)·n + 7  →  (1/8)·n^2 + (-23/8)·n + 47 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i4=1, i5=0) |
| n^2 | 0.0319 | ramp | (1/16)·n^2 + (5/8)·n - 5  →  (1/8)·n^2 + (-29/8)·n + 65 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i4=7, i5=0) |
| n^2 | 0.0293 | ramp | (1/16)·n^2 + (5/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 24 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 7  →  (1/16)·n^2 + (3/4)·n - 12 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 24  →  (1/16)·n^2 + (3/4)·n - 29 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.027 | ramp | (27/8)·n - 29  →  (1/8)·n^2 + (-29/8)·n + 51 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i5=0) |
| n^2 | 0.0245 | ramp | (19/8)·n - 10  →  (1/16)·n^2 + (3/4)·n - 7 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0244 | ramp | (9/4)·n - 8  →  (1/16)·n^2 + (3/4)·n - 8 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 8 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (13/4)·n - 26  →  (1/16)·n^2 + (5/8)·n - 6 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0, i2=7) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 22  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 22  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.024 | ramp | (25/8)·n - 23  →  (1/16)·n^2 + (5/8)·n - 7 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0, i2=7) |
| n^2 | 0.024 | ramp | 3·n - 19  →  (1/16)·n^2 + (3/4)·n - 11 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.024 | ramp | 3·n - 20  →  (1/16)·n^2 + (3/4)·n - 12 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0239 | ramp | (3/2)·n - 1  →  (1/16)·n^2 + (3/4)·n - 19 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0239 | ramp | (23/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 14 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0239 | ramp | (11/8)·n  →  (1/16)·n^2 + (3/4)·n - 21 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.0237 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (5/2)·n - 11  →  (1/16)·n^2 + (3/4)·n - 19 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (5/8)·n - 6 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0, i2=8) |
| n^2 | 0.0237 | ramp | (19/8)·n - 9  →  (1/16)·n^2 + (3/4)·n - 21 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (19/8)·n - 10  →  (1/16)·n^2 + (3/4)·n - 22 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0236 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 23 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 8  →  (1/16)·n^2 + (5/8)·n - 24 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 8  →  (1/16)·n^2 + (5/8)·n - 24 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0234 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0234 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0234 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0234 | ramp | (7/2)·n - 34  →  (1/16)·n^2 + (5/8)·n - 19 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i1=2, i5=0) |
| n^2 | 0.0234 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.0234 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0234 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0233 | ramp | (27/8)·n - 31  →  (1/16)·n^2 + (5/8)·n - 21 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^2 | 0.0233 | ramp | (27/8)·n - 32  →  (1/16)·n^2 + (5/8)·n - 22 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i1=1, i5=0) |
| n^2 | 0.0233 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.0233 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0233 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0232 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0232 | ramp | 3·n - 18  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.023 | ramp | (11/8)·n - 2  →  (1/16)·n^2 + (5/8)·n - 42 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.023 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.023 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.023 | ramp | (5/4)·n  →  (1/16)·n^2 + (3/4)·n - 48 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^2 | 0.0229 | ramp | (5/4)·n - 1  →  (1/16)·n^2 + (5/8)·n - 45 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0229 | ramp | (9/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0228 | ramp | (9/4)·n - 6  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.0228 | ramp | (9/4)·n - 6  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.0228 | ramp | (9/4)·n - 8  →  (1/16)·n^2 + (3/4)·n - 48 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0228 | ramp | (9/4)·n - 9  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.0227 | ramp | (9/4)·n - 10  →  (1/16)·n^2 + (5/8)·n - 45 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0227 | ramp | (17/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0227 | ramp | (17/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0224 | ramp | 3·n - 17  →  (1/16)·n^2 + (3/4)·n - 53 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.0156 | ramp | 22  →  (1/16)·n^2 + (-3/8)·n - 5 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^1.5 | 4.03 | ramp | 4  →  (9/8)·n - 10 | 6·n - 81 | 4.5·n^-2 | read A[i1, i5] (i0=0, i5=0) |
| n^1.5 | 2.62 | ramp | 3  →  (9/8)·n - 2 | (15/4)·n - 15 | 2.81·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 2.62 | ramp | 4  →  (9/8)·n - 2 | (15/4)·n - 20 | 2.81·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^1.5 | 2.3 | ramp | 13  →  (9/8)·n - 8 | (27/8)·n - 54 | 2.53·n^-2 | read A[i1, i5] (i0=0, i4=0); read A[i1, i5] (i0=0, i4=1) (+1) |
| n^1.5 | 2.18 | ramp | 4  →  (9/8)·n - 2 | (25/8)·n - 15 | 2.34·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.685 | ramp | 5  →  (9/8)·n - 8 | n - 10 | 0.75·n^-2 | read A[i1, i5] (i0=0, i4=0, i5=0) |
| n^1.5 | 0.601 | ramp | 5  →  (9/8)·n - 8 | (7/8)·n - 8 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=7, i5=0) |
| n^1.5 | 0.592 | ramp | 13  →  (9/8)·n - 14 | (7/8)·n - 14 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=7); read A[i5, i4] (i0=0, i4=8) (+1) |
| n^1.5 | 0.579 | ramp | n + 4  →  (9/8)·n - 1 | (5/8)·n - 20 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.54 | ramp | (1/4)·n + 3  →  n - 1 | (3/4)·n - 18 | 0.562·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.532 | ramp | 4  →  (9/8)·n - 1 | (3/4)·n - 1 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0) |
| n^1.5 | 0.516 | ramp | 5  →  (9/8)·n - 8 | (3/4)·n - 6 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0) |
| n^1.5 | 0.516 | ramp | 5  →  (9/8)·n - 8 | (3/4)·n - 6 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 0.516 | ramp | 5  →  (9/8)·n - 8 | (3/4)·n - 6 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 0.514 | ramp | 4  →  (9/8)·n - 9 | (3/4)·n - 6 | 0.562·n^-2 | read A[i1, i5] (i0=0, i4=1, i5=0) |
| n^1.5 | 0.512 | ramp | 3  →  (9/8)·n - 10 | (3/4)·n - 6 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.503 | ramp | 11  →  (9/8)·n - 16 | (3/4)·n - 12 | 0.562·n^-2 | read A[i1, i5] (i0=0) |
| n^1.5 | 0.475 | ramp | (3/8)·n + 3  →  n - 1 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.445 | ramp | 4  →  (9/8)·n - 1 | (5/8)·n | 0.469·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.438 | ramp | 12  →  (9/8)·n - 6 | (5/8)·n - 5 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.436 | ramp | 11  →  (9/8)·n - 7 | (5/8)·n - 5 | 0.469·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^1.5 | 0.436 | ramp | 11  →  (9/8)·n - 7 | (5/8)·n - 5 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.436 | ramp | 11  →  (9/8)·n - 7 | (5/8)·n - 5 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.171 | ramp | 11  →  (9/8)·n - 7 | (1/4)·n - 3 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.156 | ramp | 2·n - 2  →  (17/8)·n - 8 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=7) |
| n^1.5 | 0.122 | ramp | (9/8)·n + 4  →  (5/4)·n - 1 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=9, i5=7) |
| n^1.5 | 0.119 | ramp | n + 3  →  (9/8)·n - 1 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=8, i5=6) |
| n^1.5 | 0.119 | ramp | n + 3  →  (9/8)·n - 1 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=8, i5=7) |
| n^1.5 | 0.116 | ramp | n + 4  →  (9/8)·n - 1 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^1.5 | 0.0876 | ramp | 12  →  (9/8)·n - 6 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.0872 | ramp | 11  →  (9/8)·n - 7 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0) |
| n^1.5 | 0.0872 | ramp | 11  →  (9/8)·n - 7 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0) |
| n^1.5 | 0.0872 | ramp | 11  →  (9/8)·n - 7 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 0.0872 | ramp | 11  →  (9/8)·n - 7 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 0.0872 | ramp | 11  →  (9/8)·n - 7 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.0845 | ramp | 13  →  (9/8)·n - 14 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1.5 | 0.0842 | ramp | 12  →  (9/8)·n - 15 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=1) |
| n^1.5 | 0.0842 | ramp | 12  →  (9/8)·n - 15 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=1, i5=0) |
| n^1.5 | 0.0839 | ramp | 11  →  (9/8)·n - 16 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=1) |
| n^1.5 | 0.0839 | ramp | 11  →  (9/8)·n - 16 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=1, i5=0) |
| n^1.5 | 0.0839 | ramp | 11  →  (9/8)·n - 16 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1.5 | 0.0647 | ramp | (1/4)·n + 3  →  (3/8)·n - 1 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=2, i5=0) |
| n^1 | 24.2 | level | 3 | 14·n - 118 | 10.5·n^-2 | read A[i1, i5] (i0=0) |
| n^1 | 19.7 | level | 3 | (91/8)·n - 91 | 8.53·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 10.6 | level | 3 | (49/8)·n - 49 | 4.59·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 10.6 | level | 3 | (49/8)·n - 56 | 4.59·n^-2 | read A[i1, i5] (i0=0) |
| n^1 | 9.9 | level | 2 | 7·n - 21 | 5.25·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 7 | level | 1 | 7·n - 36 | 5.25·n^-2 | read A[i1, i5] (i0=0) |
| n^1 | 6 | level | 1 | 6·n - 33 | 4.5·n^-2 | read A[i1, i3] (i0=0, i3=0) |
| n^1 | 4.55 | level | 3 | (21/8)·n - 21 | 1.97·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 4.55 | level | 3 | (21/8)·n - 21 | 1.97·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1 | 4.24 | level | 2 | 3·n + 30 | 2.25·n^-2 | read A[i1, i2] (i0=0, i1=1, i2=0); read A[i2, i2] (i0=0, i1=1, i2=0) (+7) |
| n^1 | 4 | level | 4 | 2·n - 25 | 1.5·n^-2 | read A[i1, i5] (i0=0, i1=1); read A[i1, i2] (i0=0, i1=2, i2=0) (+3) |
| n^1 | 3.25 | level | 3 | (15/8)·n - 15 | 1.41·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 2.65 | level | 2 | (15/8)·n | 1.41·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 2.62 | level | 1 | (21/8)·n - 6 | 1.97·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1 | 2.12 | level | (1/8)·n^2 | 6 | 4.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 2.12 | level | (1/8)·n^2 | 6 | 4.5·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 1.77 | level | (1/8)·n^2 | 5 | 3.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 1.75 | level | (1/16)·n^2 + (3/2)·n + (-9/16) | 7 | 5.25·n^-3 | read A[i2, i2] (i0=0, i1=1, i2=0); read A[i1, i2] (i0=0, i1=2, i2=0) (+3) |
| n^1 | 1.75 | level | (1/16)·n^2 + (5/8)·n - 1 | 7 | 5.25·n^-3 | read A[i2, i2] (i0=0, i1=1, i2=0); read A[i1, i2] (i0=0, i1=2, i2=0) (+3) |
| n^1 | 1.75 | level | 4 | (7/8)·n - 7 | 0.656·n^-2 | read A[i5, i4] (i0=0, i1=2, i5=0) |
| n^1 | 1.73 | level | 3 | n - 8 | 0.75·n^-2 | write A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n - 14 | 0.656·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n - 8 | 0.656·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n - 14 | 0.656·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n - 7 | 0.656·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n - 7 | 0.656·n^-2 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 20 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 18 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 16 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 14 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 12 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 7 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 6 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 5 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 4 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 3 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 2 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 8 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 27 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=8) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 24 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=7) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 22 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=6) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 10 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 1.3 | level | 3 | (3/4)·n - 6 | 0.562·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 5 | 3.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 1.08 | level | 3 | (5/8)·n - 5 | 0.469·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 1.06 | level | 2 | (3/4)·n | 0.562·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 1 | level | (1/16)·n^2 + (5/8)·n - 10 | 4 | 3·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.884 | level | 2 | (5/8)·n - 5 | 0.469·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n - 7 | 0.656·n^-2 | read A[i1, i4] (i0=0, i4=1, i5=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.656·n^-2 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.75 | level | (1/16)·n^2 + (5/8)·n - 12 | 3 | 2.25·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.75 | level | (1/16)·n^2 + (3/4)·n - 9 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.75 | level | 1 | (3/4)·n - 6 | 0.562·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (7/8)·n | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0, i1=15, i2=0); read A[i1, i2] (i0=0, i1=8, i2=0) |
| n^1 | 0.707 | level | (1/8)·n^2 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0, i1=15, i2=0); read A[i1, i2] (i0=0, i1=8, i2=0) |
| n^1 | 0.5 | level | (1/16)·n^2 + (5/8)·n - 14 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-15/8)·n + (155/4) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=7, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-21/8)·n + 35 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=7, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-9/8)·n + 26 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=1, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/2)·n + 33 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-19/8)·n + 31 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-9/4)·n + 29 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-17/8)·n + 27 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - 2·n + 25 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-15/8)·n + 23 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=1, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=23, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=23, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-13/8)·n + 19 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/2)·n + 17 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-11/8)·n + 15 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/4)·n + 13 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-9/8)·n + 11 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + (27/4) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 9 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/4)·n + 6 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/8)·n + 5 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/2)·n + 4 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/8)·n + 3 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n + (3/8) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 1 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 9 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 7 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (1/4) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/4)·n + 15 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=7, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-13/8)·n + 14 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-3/2)·n + 13 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-11/8)·n + 12 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-5/4)·n + 11 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-9/8)·n + 10 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + (63/4) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=7, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 8 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=1, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (1/8) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=1, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + (191/8) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/4)·n + 21 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 9 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=1, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-7/8)·n + 7 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (1/4) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (63/8) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=16, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=16, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=16, i2=8, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=16, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 21 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 19 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 17 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 15 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 13 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 8 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 7 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 7 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 8 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 20 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 18 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 14 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 12 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 28 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 25 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 23 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 27 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 24 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 22 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-3/2) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-9/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (9/8)·n + (-51/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (-21/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (-53/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/4)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + n + (-21/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-107/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (9/8)·n + (-99/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-99/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-139/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-43/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-9/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (11/8)·n + (-43/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-171/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + n + (-29/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-41/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 11 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 10 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 18 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 14 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 12 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 10 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 7 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 18 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 14 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 12 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 10 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-171/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-203/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-131/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-49/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 22 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 20 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 22 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 20 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-9/16) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=7, i4=1, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=7, i4=1, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-89/16) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=2, i4=6, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=2, i4=6, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-25/16) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=1, i2=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=1, i2=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 24 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 8 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-121/16) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=1, i4=7, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=1, i4=7, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (23/16) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=9, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=9, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 3 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 4 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 5 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-235/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-57/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (11/8)·n + (-219/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-9/16) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-203/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-49/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-131/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (11/8)·n + (-187/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (-73/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (103/16) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=14, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 6 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=14, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-105/16) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=1, i4=7, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 7 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=1, i4=7, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (119/16) | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=15, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (11/8)·n + 7 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=15, i2=8, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 7 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=15, i2=8, i3=0) |
| n^1 | 0.217 | level | 3 | (1/8)·n - 1 | 0.0938·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 0.217 | level | 3 | (1/8)·n - 1 | 0.0938·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 0.177 | level | 2 | (1/8)·n - 1 | 0.0938·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 0.177 | level | 2 | (1/8)·n - 1 | 0.0938·n^-2 | write A[i1, i4] (i0=0, i4=0) |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^0.5 | 8.84 | level | (25/8)·n - 22 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.66 | level | 3·n - 20 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.48 | level | (23/8)·n - 18 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.29 | level | (11/4)·n - 16 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.1 | level | (21/8)·n - 14 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.91 | level | (5/2)·n - 12 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.71 | level | (19/8)·n - 10 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.71 | level | (19/8)·n - 11 | 5 | 3.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 7.5 | level | (9/4)·n - 8 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.5 | level | (9/4)·n - 9 | 5 | 3.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 7.29 | level | (17/8)·n - 6 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.29 | level | (17/8)·n - 7 | 5 | 3.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 7.07 | level | 2·n - 5 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.85 | level | (15/8)·n - 4 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.61 | level | (7/4)·n - 3 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.37 | level | (13/8)·n - 2 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.32 | level | (5/2)·n - 13 | 4 | 3·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 6.12 | level | (3/2)·n - 1 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.86 | level | (11/8)·n | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.66 | level | 2·n - 6 | 4 | 3·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 5.59 | level | (5/4)·n + 1 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.3 | level | (25/8)·n - 22 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i1=25, i2=8, i3=0); read A[i3, i2] (i0=0, i1=25, i2=8, i3=7) (+1) |
| n^0.5 | 5.3 | level | (9/8)·n + 2 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.3 | level | (9/8)·n + 1 | 5 | 3.75·n^-3 | read A[i2, i2] (i0=0, i1=9, i2=0); read A[i2, i2] (i0=0, i1=9, i2=1) (+3) |
| n^0.5 | 5.2 | level | 3·n - 20 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 5.09 | level | (23/8)·n - 18 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 5 | level | n + 3 | 5 | 3.75·n^-3 | read A[i5, i4] (i0=0, i4=8) |
| n^0.5 | 5 | level | n + 2 | 5 | 3.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 4.97 | level | (11/4)·n - 16 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.86 | level | (21/8)·n - 14 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.86 | level | (21/8)·n - 15 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 4.74 | level | (5/2)·n - 12 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.62 | level | (19/8)·n - 10 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.5 | level | (9/4)·n - 8 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.37 | level | (17/8)·n - 6 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i1=17, i2=8, i3=0); read A[i3, i2] (i0=0, i1=17, i2=8, i3=7) (+1) |
| n^0.5 | 4.24 | level | (9/8)·n + 2 | 4 | 3·n^-3 | read A[i3, i2] (i0=0, i1=9, i2=8, i3=0); read A[i3, i2] (i0=0, i1=9, i2=8, i3=7) (+2) |
| n^0.5 | 4.11 | level | (15/8)·n - 5 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 4.06 | level | (33/8)·n - 45 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i1=33, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 4 | level | 4·n - 42 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.94 | level | (31/8)·n - 39 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.87 | level | (15/4)·n - 36 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.81 | level | (29/8)·n - 33 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.74 | level | (7/8)·n + 2 | 4 | 3·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 3.74 | level | (7/2)·n - 30 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.67 | level | (27/8)·n - 27 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.61 | level | (13/4)·n - 24 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.54 | level | (25/8)·n - 21 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i1=25, i2=16, i3=6); read A[i3, i2] (i0=0, i1=25, i2=16, i3=7) |
| n^0.5 | 3.52 | level | (11/8)·n - 1 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 3.46 | level | 3·n - 19 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.39 | level | (23/8)·n - 17 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.35 | level | (5/4)·n | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 3.35 | level | (5/4)·n | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0, i1=10, i2=0); read A[i2, i2] (i0=0, i1=10, i2=1) (+1) |
| n^0.5 | 3.32 | level | (11/4)·n - 15 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.32 | level | (11/4)·n - 17 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 3.24 | level | (21/8)·n - 13 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.18 | level | (9/8)·n + 1 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0, i1=9) |
| n^0.5 | 3.16 | level | (5/2)·n - 11 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.08 | level | (19/8)·n - 9 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3 | level | (9/4)·n - 7 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3 | level | n + 2 | 3 | 2.25·n^-3 | read A[i5, i4] (i0=0, i1=8, i4=0, i5=0); read A[i5, i4] (i0=0, i1=8, i4=0, i5=6) (+1) |
| n^0.5 | 2.92 | level | (17/8)·n - 5 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i1=17, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 2.92 | level | (17/8)·n - 7 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=17, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 2.83 | level | 2·n - 5 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.74 | level | (15/8)·n - 3 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.74 | level | (15/8)·n - 4 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.65 | level | (7/4)·n - 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.65 | level | (7/4)·n - 3 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.65 | level | (7/4)·n - 4 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 2.6 | level | (3/4)·n + 2 | 3 | 2.25·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 2.55 | level | (13/8)·n - 1 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.55 | level | (13/8)·n - 2 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.45 | level | (3/2)·n | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.45 | level | (3/2)·n - 1 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.45 | level | (3/2)·n - 2 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 2.35 | level | (11/8)·n + 1 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.35 | level | (11/8)·n | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.24 | level | (5/4)·n + 1 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.24 | level | (5/4)·n + 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=10, i4=6, i5=7); read A[i5, i4] (i0=0, i4=6) |
| n^0.5 | 2.12 | level | (9/8)·n + 3 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=9, i4=7, i5=7); read A[i5, i4] (i0=0, i4=7) |
| n^0.5 | 2 | level | n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 2 | level | n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.87 | level | (7/8)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.87 | level | (7/2)·n + (-33/2) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=2, i5=0) |
| n^0.5 | 1.84 | level | (27/8)·n + (-123/8) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.84 | level | (27/8)·n + (-115/8) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.77 | level | (25/8)·n - 22 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=25, i2=8) |
| n^0.5 | 1.73 | level | (3/4)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.73 | level | 3·n - 20 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.73 | level | 3·n - 21 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.7 | level | (23/8)·n - 18 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.7 | level | (23/8)·n - 19 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.7 | level | (23/8)·n - 19 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.66 | level | (11/4)·n - 16 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.66 | level | (11/4)·n - 17 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.62 | level | (21/8)·n - 14 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.62 | level | (21/8)·n - 15 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.58 | level | (5/8)·n + 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 1.58 | level | (5/2)·n - 12 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.58 | level | (5/2)·n - 13 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.58 | level | (5/2)·n - 15 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=2, i5=0) |
| n^0.5 | 1.58 | level | (5/2)·n + (-9/2) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=2, i5=0) |
| n^0.5 | 1.54 | level | (19/8)·n - 10 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.54 | level | (19/8)·n - 11 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.54 | level | (19/8)·n - 14 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.54 | level | (19/8)·n + (-35/8) | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.54 | level | (19/8)·n - 13 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.54 | level | (19/8)·n + (-27/8) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.5 | level | (9/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=18, i2=7) |
| n^0.5 | 1.5 | level | (9/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.5 | level | (9/4)·n - 8 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=18, i2=8) |
| n^0.5 | 1.5 | level | (9/4)·n - 11 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=8, i5=0) |
| n^0.5 | 1.46 | level | (17/8)·n - 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=17, i2=7) |
| n^0.5 | 1.46 | level | (17/8)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=17, i2=8) |
| n^0.5 | 1.46 | level | (17/8)·n - 10 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=7, i5=0) |
| n^0.5 | 1.41 | level | 2·n - 3 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8, i5=7) |
| n^0.5 | 1.41 | level | 2·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=16, i2=8) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=16, i2=7) |
| n^0.5 | 1.41 | level | (1/2)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=4, i2=0); read A[i2, i2] (i0=0, i1=4, i2=1) |
| n^0.5 | 1.41 | level | 2·n - 4 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=16, i4=0, i5=7) |
| n^0.5 | 1.41 | level | 2·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.41 | level | 2·n - 8 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^0.5 | 1.37 | level | (15/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=15, i2=7) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.37 | level | (15/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.37 | level | (15/8)·n - 7 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.32 | level | (7/4)·n - 6 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.27 | level | (13/8)·n - 5 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i5=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=12, i2=8) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.22 | level | (3/8)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=3, i2=0); read A[i2, i2] (i0=0, i1=3, i2=1) |
| n^0.5 | 1.22 | level | (3/2)·n - 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=2, i5=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=11, i2=7) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=11, i2=1) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=11, i2=8) |
| n^0.5 | 1.17 | level | (11/8)·n - 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.17 | level | (11/8)·n + (-3/8) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 3 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.12 | level | (5/4)·n + 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.12 | level | (5/4)·n - 1 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=9, i5=7) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=10, i2=7) |
| n^0.5 | 1.12 | level | (5/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=10, i2=8) |
| n^0.5 | 1.06 | level | (9/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=8, i5=6) |
| n^0.5 | 1.06 | level | (9/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=8, i5=7) |
| n^0.5 | 1.06 | level | (9/8)·n + (-1/4) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 7 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n + (-65/8) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 6 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 5 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 4 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 3 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 1 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1 | level | n + 3 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8, i5=0) |
| n^0.5 | 1 | level | n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1 | level | (1/4)·n + 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=2, i4=6, i5=0); read A[i5, i4] (i0=0, i1=2, i4=6, i5=1) |
| n^0.5 | 1 | level | n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 1 | level | n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 1 | level | n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1 | level | n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=5, i2=1) |
| n^0.5 | 0.791 | level | (5/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=2, i5=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=2, i2=0) |
| n^0 | 36 | level | 1 | 36 | 27·n^-3 | read A[i1, i5] (i0=0) |
| n^0 | 8.66 | level | 3 | 5 | 3.75·n^-3 | read A[i5, i4] (i0=0, i1=2, i5=0) |
| n^0 | 3.32 | level | 11 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 3.16 | level | 10 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 3 | level | 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.83 | level | 8 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.65 | level | 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.45 | level | 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=2, i2=1) |

*bin 132: finite-size slope 3.33 at n≈256–312; converges to n^3; bin 133: finite-size slope 3.26 at n≈256–312; converges to n^3; bin 136: finite-size slope 3.27 at n≈256–312; converges to n^3.*

Row-panel and trailing-submatrix re-reads (`read A[i3,i2]`, `read A[i5,i4]`) ramp to (1/16)n^2 + (3/4)n lines with cubic populations — d = 4.0, headroom +1.0. At the anchor sizes (n ≈ 256–312) the finite-size slope of these families still reads ≈ 4.4 because the negative n^2 corrections in their populations have not yet died out; the exact degrees settle it at 4.0.

## lu — single-shot  [`exact`]

Accesses $A(n) = (4/3)·n^3 + (-1/2)·n^2 + (-5/6)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.00643·n^4  +  0.221·n^3.5  +  2.29·n^3  +  6.37·n^2.5  +  37.5·n^2  +  20.5·n^1.5  +  205·n^1  +  521·n^0.5  +  47.6·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.00309 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (7/384)·n^3 + (-123/64)·n^2 + (1619/24)·n - 790 | 0.0137 | read A[i3, i2] (i0=0) |
| n^4 | 0.00193 | ramp | (11/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 91 | (7/512)·n^3 + (-195/128)·n^2 + (901/16)·n - 690 | 0.0103 | read A[i5, i4] (i0=0) |
| n^4 | 0.000411 | ramp | (25/4)·n - 121  →  (1/16)·n^2 + (3/4)·n - 4 | (1/384)·n^3 + (-21/64)·n^2 + (329/24)·n - 190 | 0.00195 | read A[i3, i2] (i0=0) |
| n^4 | 0.000316 | ramp | 3·n - 16  →  (1/16)·n^2 + (3/4)·n - 86 | (7/3072)·n^3 + (-35/128)·n^2 + (259/24)·n - 140 | 0.00171 | read A[i5, i4] (i0=0) |
| n^4 | 0.00031 | ramp | (17/8)·n - 2  →  (1/16)·n^2 + (3/4)·n - 121 | (7/3072)·n^3 + (-35/128)·n^2 + (259/24)·n - 140 | 0.00171 | read A[i5, i4] (i0=0) |
| n^4 | 0.000268 | ramp | (9/4)·n - 4  →  (1/16)·n^2 + (3/4)·n - 91 | (1/512)·n^3 + (-15/64)·n^2 + (37/4)·n - 120 | 0.00146 | read A[i5, i4] (i0=0) |
| n^4 | 6.23e-05 | ramp | 2·n - 2  →  (1/16)·n^2 + (3/4)·n - 86 | (1/3072)·n^3 + (-55/48)·n + 25 | 0.000244 | read A[i5, i4] (i0=0) |
| n^4 | 4.1e-05 | ramp | (25/8)·n - 17  →  (1/16)·n^2 + (3/4)·n - 121 | (1/3072)·n^3 + (-3/64)·n^2 + (107/48)·n - 35 | 0.000244 | read A[i5, i4] (i0=0) |
| n^3.5 | 0.0988 | ramp | 11  →  (9/8)·n - 2 | (7/48)·n^3 + (-63/16)·n^2 + (217/6)·n - 112 | 0.109 | read A[i3, i2] (i0=0) |
| n^3.5 | 0.0854 | ramp | 5  →  (9/8)·n - 8 | (49/384)·n^3 + (-413/128)·n^2 + (1309/48)·n - 77 | 0.0957 | read A[i5, i4] (i0=0) |
| n^3.5 | 0.0126 | ramp | 21  →  (9/8)·n - 9 | (1/48)·n^3 + (-41/32)·n^2 + (311/12)·n - 172 | 0.0156 | read A[i1, i3] (i0=0) |
| n^3.5 | 0.0121 | ramp | 12  →  (9/8)·n - 15 | (7/384)·n^3 + (-63/128)·n^2 + (175/48)·n - 7 | 0.0137 | read A[i5, i4] (i0=0) |
| n^3.5 | 0.0108 | ramp | 14  →  (9/8)·n - 16 | (7/384)·n^3 + (-35/32)·n^2 + (259/12)·n - 140 | 0.0137 | read A[i1, i5] (i0=0) |
| n^3.5 | 0.00148 | ramp | 15  →  (9/8)·n - 23 | (1/384)·n^3 + (-3/16)·n^2 + (13/3)·n - 32 | 0.00195 | read A[i1, i5] (i0=0) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (-273/32)·n^2 + (91/2)·n - 42 | 0.328 | read A[i1, i3] (i0=0); write A[i1, i2] (i0=0) (+1) |
| n^3 | 0.333 | level | 1 | (1/3)·n^3 - n^2 + (23/3)·n - 14 | 0.25 | read A[i1, i2] (i0=0); read A[i1, i4] (i0=0, i5=0) (+1) |
| n^3 | 0.289 | level | 3 | (1/6)·n^3 + (-169/6)·n + 146 | 0.125 | read A[i2, i2] (i0=0, i1=2, i2=1); read A[i5, i4] (i0=0, i1=2) (+3) |
| n^3 | 0.0835 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 11 | (7/16)·n^2 + (-51/2)·n + 368 | 0.328/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0737 | ramp | (3/2)·n - 3  →  (1/16)·n^2 + (3/4)·n - 31 | (49/128)·n^2 + (-315/16)·n + 253 | 0.287/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0732 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (49/128)·n^2 + (-371/16)·n + 351 | 0.287/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0711 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (3/8)·n^2 + (-47/2)·n + 370 | 0.281/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0598 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (5/16)·n^2 + (-75/4)·n + 280 | 0.234/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0595 | ramp | (11/8)·n + 2  →  (1/16)·n^2 + (3/4)·n - 57 | (21/64)·n^2 + (-87/4)·n + 360 | 0.246/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0538 | ramp | (7/2)·n - 31  →  (1/16)·n^2 + (3/4)·n - 4 | (35/128)·n^2 + (-225/16)·n + 180 | 0.205/n | read A[i2, i2] (i0=0) |
| n^3 | 0.0516 | ramp | (19/8)·n - 8  →  (1/16)·n^2 + (3/4)·n - 31 | (35/128)·n^2 + (-255/16)·n + 230 | 0.205/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0379 | ramp | (1/2)·n + 4  →  (1/16)·n^2 + (3/4)·n - 91 | (9/32)·n^2 - 19·n + 320 | 0.211/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0361 | level | 3 | (1/48)·n^3 + (-11/32)·n^2 + (37/24)·n - 1 | 0.0156 | write A[i1, i2] (i0=0) |
| n^3 | 0.0127 | ramp | (3/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 91 | (3/32)·n^2 + (-49/8)·n + 100 | 0.0703/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.012 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-15/4)·n + 56 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0114 | ramp | (33/8)·n - 43  →  (1/16)·n^2 + (3/4)·n - 11 | (1/16)·n^2 + (-9/2)·n + 80 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0113 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-19/4)·n + 90 | 0.0469/n | read A[i3, i2] (i0=0, i3=7) |
| n^3 | 0.0113 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-19/4)·n + 90 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0113 | ramp | (21/4)·n - 81  →  (1/16)·n^2 + (3/4)·n - 4 | (1/16)·n^2 + (-19/4)·n + 90 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^3 | 0.011 | ramp | (27/8)·n - 28  →  (1/16)·n^2 + (3/4)·n - 4 | (7/128)·n^2 + (-39/16)·n + 27 | 0.041/n | read A[i2, i2] (i0=0) |
| n^3 | 0.0105 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (3/4)·n - 4 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i2, i2] (i0=0) |
| n^3 | 0.0105 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0104 | ramp | 3·n - 18  →  (1/16)·n^2 + (3/4)·n - 28 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i5, i4] (i0=0, i4=0) |
| n^3 | 0.0103 | ramp | (9/4)·n - 9  →  (1/16)·n^2 + (3/4)·n - 49 | (7/128)·n^2 + (-49/16)·n + 42 | 0.041/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0103 | ramp | (19/8)·n - 12  →  (1/16)·n^2 + (3/4)·n - 32 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0103 | ramp | (9/4)·n - 6  →  (1/16)·n^2 + (3/4)·n - 46 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i5, i4] (i0=0, i4=6) |
| n^3 | 0.0103 | ramp | (17/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 49 | (7/128)·n^2 + (-51/16)·n + 46 | 0.041/n | read A[i5, i4] (i0=0, i4=7) |
| n^3 | 0.01 | ramp | (21/4)·n - 80  →  (1/16)·n^2 + (3/4)·n - 3 | (7/128)·n^2 + (-65/16)·n + 75 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00997 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (7/128)·n^2 + (-65/16)·n + 75 | 0.041/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0098 | ramp | 3·n - 17  →  (1/16)·n^2 + (3/4)·n - 53 | (7/128)·n^2 + (-63/16)·n + 70 | 0.041/n | read A[i5, i4] (i0=0, i4=8) |
| n^3 | 0.00969 | ramp | (17/8)·n - 3  →  (1/16)·n^2 + (3/4)·n - 81 | (7/128)·n^2 + (-63/16)·n + 70 | 0.041/n | read A[i5, i4] (i0=0, i4=15) |
| n^3 | 0.00901 | ramp | (35/8)·n - 54  →  (1/16)·n^2 + (3/4)·n - 5 | (3/64)·n^2 + (-11/4)·n + 40 | 0.0352/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00835 | ramp | (9/4)·n - 5  →  (1/16)·n^2 + (3/4)·n - 57 | (3/64)·n^2 + (-27/8)·n + 60 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00754 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (3/4)·n - 11 | (5/128)·n^2 + (-35/16)·n + 30 | 0.0293/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00705 | ramp | (27/8)·n - 25  →  (1/16)·n^2 + (3/4)·n - 31 | (5/128)·n^2 + (-45/16)·n + 50 | 0.0293/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00671 | ramp | (5/4)·n + 3  →  (1/16)·n^2 + (3/4)·n - 57 | (3/64)·n^2 + (-21/8)·n + 36 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00671 | ramp | (5/4)·n + 3  →  (1/16)·n^2 + (3/4)·n - 57 | (3/64)·n^2 + (-21/8)·n + 36 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00641 | ramp | 2·n - 2  →  (1/16)·n^2 + (3/4)·n - 86 | (3/64)·n^2 + (-27/8)·n + 60 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00615 | ramp | (9/8)·n + 5  →  (1/16)·n^2 + (3/4)·n - 121 | (3/64)·n^2 + (-27/8)·n + 60 | 0.0352/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00211 | ramp | 2·n - 2  →  (1/16)·n^2 + (3/4)·n - 86 | (1/64)·n^2 + (-5/4)·n + 25 | 0.0117/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^3 | 0.00203 | ramp | (9/8)·n + 5  →  (1/16)·n^2 + (3/4)·n - 121 | (1/64)·n^2 + (-5/4)·n + 25 | 0.0117/n | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^3 | 0.00176 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/128)·n^2 + (1/16)·n - 9 | 0.00586/n | read A[i5, i4] (i0=0, i4=8) |
| n^3 | 0.00151 | ramp | (17/4)·n - 50  →  (1/16)·n^2 + (3/4)·n - 10 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00151 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (3/4)·n - 11 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00151 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (3/4)·n - 11 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00143 | ramp | (41/8)·n - 75  →  (1/16)·n^2 + (3/4)·n - 9 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00143 | ramp | (41/8)·n - 77  →  (1/16)·n^2 + (3/4)·n - 11 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i2, i2] (i0=0) |
| n^3 | 0.00142 | ramp | 4·n - 40  →  (1/16)·n^2 + (3/4)·n - 28 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0, i4=0) |
| n^3 | 0.0014 | ramp | (13/4)·n - 22  →  (1/16)·n^2 + (3/4)·n - 46 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0, i4=6) |
| n^3 | 0.0014 | ramp | (13/4)·n - 26  →  (1/16)·n^2 + (3/4)·n - 50 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0014 | ramp | (25/8)·n - 19  →  (1/16)·n^2 + (3/4)·n - 49 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0, i4=7) |
| n^3 | 0.00136 | ramp | (49/8)·n - 115  →  (1/16)·n^2 + (3/4)·n - 10 | (1/128)·n^2 + (-11/16)·n + 15 | 0.00586/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00131 | ramp | (25/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 81 | (1/128)·n^2 + (-11/16)·n + 15 | 0.00586/n | read A[i5, i4] (i0=0, i4=15) |
| n^3 | 0.00114 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.0011 | ramp | (9/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 81 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00107 | ramp | (17/8)·n - 3  →  (1/16)·n^2 + (3/4)·n - 81 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00107 | ramp | 2·n - 2  →  (1/16)·n^2 + (3/4)·n - 86 | (1/128)·n^2 + (-9/16)·n + 10 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^3 | 0.00104 | ramp | 3·n - 16  →  (1/16)·n^2 + (3/4)·n - 86 | (1/128)·n^2 + (-11/16)·n + 15 | 0.00586/n | read A[i5, i4] (i0=0) |
| n^2.5 | 1.54 | ramp | 4  →  (9/8)·n - 2 | (15/8)·n^2 + (-75/4)·n + 50 | 1.41/n | read A[i5, i4] (i0=0) |
| n^2.5 | 1.22 | ramp | 5  →  (9/8)·n - 2 | (35/16)·n^2 + (-235/8)·n + 105 | 1.64/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.366 | ramp | 5  →  (9/8)·n - 1 | (7/16)·n^2 + (-21/8)·n + 3 | 0.328/n | read A[i5, i4] (i0=0, i4=0) |
| n^2.5 | 0.323 | ramp | 14  →  (9/8)·n - 9 | (7/16)·n^2 + (-35/2)·n + 168 | 0.328/n | read A[i1, i5] (i0=0) |
| n^2.5 | 0.3 | ramp | 5  →  (9/8)·n - 8 | (3/8)·n^2 + (-51/8)·n + 27 | 0.281/n | read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 0.267 | ramp | 5  →  (9/8)·n - 9 | (1/2)·n^2 + (-21/2)·n + 55 | 0.375/n | read A[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.257 | ramp | 11  →  (9/8)·n - 7 | (5/16)·n^2 + (-25/8)·n + 5 | 0.234/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.246 | ramp | 4  →  (9/8)·n - 2 | (7/16)·n^2 + (-23/8)·n + 5 | 0.328/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.235 | ramp | 12  →  (9/8)·n - 15 | (7/16)·n^2 + (-21/2)·n + 56 | 0.328/n | read A[i1, i5] (i0=0) |
| n^2.5 | 0.223 | ramp | 5  →  (9/8)·n - 16 | (7/16)·n^2 - 14·n + 112 | 0.328/n | read A[i1, i5] (i0=0, i5=0) |
| n^2.5 | 0.211 | ramp | 3  →  (9/8)·n - 2 | (3/8)·n^2 + (-9/8)·n | 0.281/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.211 | ramp | 4  →  (9/8)·n - 2 | (3/8)·n^2 + (-17/8)·n + 3 | 0.281/n | read A[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.207 | ramp | 4  →  (9/8)·n - 8 | (49/128)·n^2 + (-91/16)·n + 21 | 0.287/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.205 | ramp | 4  →  (9/8)·n - 9 | (49/128)·n^2 + (-105/16)·n + 28 | 0.287/n | read A[i5, i4] (i0=0, i5=0) |
| n^2.5 | 0.198 | ramp | 14  →  (9/8)·n - 9 | (3/8)·n^2 + (-105/8)·n + 114 | 0.281/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0513 | ramp | 11  →  (9/8)·n - 7 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0469/n | read A[i5, i4] (i0=0, i4=6) |
| n^2.5 | 0.0466 | ramp | 21  →  (9/8)·n - 8 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0469/n | read A[i1, i5] (i0=0, i4=0) |
| n^2.5 | 0.0351 | ramp | 11  →  (9/8)·n - 7 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | read A[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.0351 | ramp | 11  →  (9/8)·n - 7 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.0332 | ramp | 13  →  (9/8)·n - 14 | (1/16)·n^2 + (-7/4)·n + 12 | 0.0469/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0331 | ramp | 11  →  (9/8)·n - 16 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0469/n | read A[i1, i3] (i0=0) |
| n^2.5 | 0.0316 | ramp | 13  →  (9/8)·n - 23 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0469/n | read A[i1, i5] (i0=0) |
| n^2.5 | 0.031 | ramp | 6  →  (9/8)·n - 23 | (1/16)·n^2 + (-5/2)·n + 24 | 0.0469/n | read A[i1, i5] (i0=0, i5=0) |
| n^2.5 | 0.0294 | ramp | 12  →  (9/8)·n - 15 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | read A[i5, i4] (i0=0) |
| n^2.5 | 0.0294 | ramp | 12  →  (9/8)·n - 15 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | read A[i5, i4] (i0=0, i5=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 + (-49/2)·n + 84 | 1.31/n | read A[i1, i3] (i0=0) |
| n^2 | 2.27 | level | 3 | (21/16)·n^2 + (-35/2)·n + 56 | 0.984/n | write A[i1, i2] (i0=0) |
| n^2 | 1.86 | level | 2 | (21/16)·n^2 + (-49/8)·n | 0.984/n | write A[i1, i2] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 - 14·n + 35 | 1.31/n | read A[i1, i3] (i0=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-63/4)·n + 70 | 0.656/n | read A[i1, i3] (i0=0); write A[i1, i2] (i0=0) |
| n^2 | 1.23 | ramp | (1/16)·n^2 + (5/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 3 | (21/4)·n - 132 | 3.94·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 1.14 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 11 | 6·n - 152 | 4.5·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 1.03 | ramp | (1/16)·n^2 + (5/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 13 | (35/8)·n - 100 | 3.28·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 1.01 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (21/4)·n - 139 | 3.94·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.976 | ramp | (5/8)·n  →  (1/16)·n^2 + (3/4)·n - 30 | (21/4)·n - 117 | 3.94·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.964 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (5/8)·n | 5·n - 130 | 3.75·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.95 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 11 | 5·n - 120 | 3.75·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.85 | ramp | (5/2)·n - 13  →  (1/16)·n^2 + (5/8)·n - 1 | (35/8)·n - 95 | 3.28·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.847 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (35/8)·n - 110 | 3.28·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.807 | ramp | (1/2)·n + 3  →  (1/16)·n^2 + (3/4)·n - 57 | (9/2)·n - 134 | 3.38·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 - 7·n + 28 | 0.328/n | write A[i1, i2] (i0=0) |
| n^2 | 0.729 | ramp | (21/8)·n - 14  →  (1/16)·n^2 + (3/4)·n - 3 | (15/4)·n - 84 | 2.81·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.707 | level | 2 | (1/2)·n^2 + (3/2)·n + 4 | 0.375/n | read A[i5, i4] (i0=0, i1=1, i4=0, i5=0); read A[i5, i4] (i0=0, i1=1, i4=6, i5=0) (+6) |
| n^2 | 0.695 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (15/4)·n - 95 | 2.81·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.578 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (5/8)·n | 3·n - 78 | 2.25·n^-2 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^2 | 0.53 | level | 2 | (3/8)·n^2 + (-3/8)·n | 0.281/n | write A[i1, i2] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 + (-63/8)·n + 35 | 0.328/n | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 + (-7/8)·n - 21 | 0.328/n | read A[i1, i4] (i0=0, i5=0) |
| n^2 | 0.38 | ramp | (17/4)·n - 49  →  (1/16)·n^2 + (5/8)·n + 1 | 2·n - 68 | 1.5·n^-2 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^2 | 0.365 | ramp | (7/4)·n - 3  →  (1/16)·n^2 + (3/4)·n - 5 | (15/8)·n - 30 | 1.41·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.342 | ramp | (9/4)·n - 9  →  (1/16)·n^2 + (5/8)·n - 1 | (7/4)·n - 31 | 1.31·n^-2 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^2 | 0.269 | ramp | (3/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 57 | (3/2)·n - 43 | 1.12·n^-2 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.233 | ramp | (3/2)·n  →  (1/16)·n^2 + (3/4)·n - 31 | (5/4)·n - 30 | 0.938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.231 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/4)·n - 35 | 0.938·n^-2 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.231 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/4)·n - 35 | 0.938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.205 | ramp | (1/16)·n^2 + (5/8)·n - 19  →  (1/16)·n^2 + (3/4)·n - 23 | (7/8)·n - 21 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.205 | ramp | (1/16)·n^2 + (5/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 25 | (7/8)·n - 21 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.205 | ramp | (1/16)·n^2 + (5/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 9 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.204 | ramp | (1/16)·n^2 + (5/8)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 22 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.204 | ramp | (1/16)·n^2 + (5/8)·n - 24  →  (1/16)·n^2 + (3/4)·n - 28 | (7/8)·n - 22 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.19 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 11 | n - 24 | 0.75·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.19 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 11 | n - 24 | 0.75·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.19 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 11 | n - 24 | 0.75·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.188 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 11 | n - 32 | 0.75·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.188 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 11 | n - 32 | 0.75·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.188 | ramp | (25/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 11 | n - 32 | 0.75·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.171 | ramp | (1/16)·n^2 + (5/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 4 | (3/4)·n - 24 | 0.562·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 22 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.169 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (5/8)·n | (7/8)·n - 22 | 0.656·n^-2 | read A[i2, i2] (i0=0, i2=8) |
| n^2 | 0.167 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 29 | 0.656·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.167 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 29 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.167 | ramp | (17/4)·n - 48  →  (1/16)·n^2 + (3/4)·n - 3 | (7/8)·n - 29 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.167 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 11 | (7/8)·n - 21 | 0.656·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.162 | ramp | (11/8)·n - 2  →  (1/16)·n^2 + (3/4)·n - 31 | (7/8)·n - 22 | 0.656·n^-2 | read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.162 | ramp | (11/8)·n - 2  →  (1/16)·n^2 + (5/8)·n - 27 | (7/8)·n - 22 | 0.656·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.158 | ramp | (1/2)·n  →  (1/16)·n^2 + (3/4)·n - 52 | (7/8)·n - 22 | 0.656·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^2 | 0.147 | ramp | (5/2)·n - 12  →  (1/16)·n^2 + (3/4)·n - 3 | (3/4)·n - 14 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.147 | ramp | (1/16)·n^2 + (5/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 14 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.145 | ramp | (27/8)·n - 27  →  (1/16)·n^2 + (3/4)·n - 3 | (3/4)·n - 19 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.145 | ramp | (27/8)·n - 28  →  (1/16)·n^2 + (3/4)·n - 4 | (3/4)·n - 19 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.145 | ramp | (27/8)·n - 29  →  (1/16)·n^2 + (5/8)·n - 1 | (3/4)·n - 19 | 0.562·n^-2 | read A[i2, i2] (i0=0, i2=7) |
| n^2 | 0.144 | ramp | (11/8)·n  →  (1/16)·n^2 + (3/4)·n - 11 | (3/4)·n - 12 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.144 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (3/4)·n - 19 | 0.562·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.142 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (3/4)·n - 25 | 0.562·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.14 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (3/4)·n - 19 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.139 | ramp | (11/8)·n - 2  →  (1/16)·n^2 + (5/8)·n - 27 | (3/4)·n - 18 | 0.562·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.138 | ramp | (5/4)·n  →  (1/16)·n^2 + (3/4)·n - 48 | (3/4)·n - 18 | 0.562·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.138 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (3/4)·n - 19 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.137 | ramp | (9/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 49 | (3/4)·n - 19 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.136 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (3/4)·n - 24 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.133 | ramp | (9/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 81 | (3/4)·n - 24 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=15) |
| n^2 | 0.123 | ramp | (19/8)·n - 10  →  (1/16)·n^2 + (3/4)·n - 3 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.123 | ramp | (9/4)·n - 8  →  (1/16)·n^2 + (3/4)·n - 4 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.123 | ramp | (5/2)·n - 12  →  (1/16)·n^2 + (3/4)·n - 3 | (5/8)·n - 11 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.121 | ramp | (13/8)·n - 2  →  (1/16)·n^2 + (3/4)·n - 8 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.121 | ramp | (3/2)·n - 1  →  (1/16)·n^2 + (3/4)·n - 11 | (5/8)·n - 10 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.12 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (5/8)·n - 15 | 0.469·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.12 | ramp | (25/8)·n - 22  →  (1/16)·n^2 + (3/4)·n - 10 | (5/8)·n - 15 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.12 | ramp | (25/8)·n - 23  →  (1/16)·n^2 + (5/8)·n - 7 | (5/8)·n - 15 | 0.469·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.119 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (5/8)·n - 20 | 0.469·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.117 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.116 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.116 | ramp | (11/8)·n + 1  →  (1/16)·n^2 + (3/4)·n - 31 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.115 | ramp | (19/8)·n - 8  →  (1/16)·n^2 + (3/4)·n - 31 | (5/8)·n - 20 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.094 | ramp | 23  →  (1/16)·n^2 + (-3/8)·n + 1 | (3/4)·n - 12 | 0.562·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-1/2)·n | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0716 | ramp | (21/8)·n - 13  →  (1/16)·n^2 + (3/4)·n - 13 | (3/8)·n - 9 | 0.281·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0714 | ramp | (5/2)·n - 12  →  (1/16)·n^2 + (3/4)·n - 16 | (3/8)·n - 9 | 0.281·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0705 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (3/8)·n - 9 | 0.281·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.048 | ramp | (25/8)·n - 23  →  (1/16)·n^2 + (5/8)·n - 7 | (1/4)·n - 6 | 0.188·n^-2 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^2 | 0.0457 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (1/4)·n - 7 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6, i5=7) |
| n^2 | 0.0456 | ramp | (9/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 49 | (1/4)·n - 7 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=7, i5=0); read A[i5, i4] (i0=0, i4=7, i5=7) |
| n^2 | 0.045 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/4)·n - 9 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=8, i5=0); read A[i5, i4] (i0=0, i4=8, i5=7) |
| n^2 | 0.044 | ramp | (9/8)·n + 4  →  (1/16)·n^2 + (3/4)·n - 81 | (1/4)·n - 9 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=15, i5=0); read A[i5, i4] (i0=0, i4=15, i5=7) |
| n^2 | 0.0293 | ramp | (1/16)·n^2 + (5/8)·n - 20  →  (1/16)·n^2 + (3/4)·n - 24 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 7  →  (1/16)·n^2 + (3/4)·n - 12 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0285 | ramp | (1/16)·n^2 + (5/8)·n - 24  →  (1/16)·n^2 + (3/4)·n - 29 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.0245 | ramp | (19/8)·n - 10  →  (1/16)·n^2 + (3/4)·n - 7 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0244 | ramp | (9/4)·n - 8  →  (1/16)·n^2 + (3/4)·n - 8 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (13/4)·n - 24  →  (1/16)·n^2 + (3/4)·n - 8 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (13/4)·n - 25  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (13/4)·n - 26  →  (1/16)·n^2 + (5/8)·n - 6 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0, i2=7) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 22  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0241 | ramp | (25/8)·n - 22  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.024 | ramp | (25/8)·n - 23  →  (1/16)·n^2 + (5/8)·n - 7 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0, i2=7) |
| n^2 | 0.024 | ramp | 3·n - 19  →  (1/16)·n^2 + (3/4)·n - 11 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.024 | ramp | 3·n - 20  →  (1/16)·n^2 + (3/4)·n - 12 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0239 | ramp | (23/8)·n - 18  →  (1/16)·n^2 + (3/4)·n - 14 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.0237 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 44  →  (1/16)·n^2 + (3/4)·n - 9 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 45  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (5/2)·n - 11  →  (1/16)·n^2 + (3/4)·n - 19 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (33/8)·n - 46  →  (1/16)·n^2 + (5/8)·n - 6 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0, i2=8) |
| n^2 | 0.0237 | ramp | (19/8)·n - 9  →  (1/16)·n^2 + (3/4)·n - 21 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0237 | ramp | (19/8)·n - 10  →  (1/16)·n^2 + (3/4)·n - 22 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0236 | ramp | (9/4)·n - 7  →  (1/16)·n^2 + (3/4)·n - 23 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0236 | ramp | (9/4)·n - 8  →  (1/16)·n^2 + (3/4)·n - 24 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 5  →  (1/16)·n^2 + (3/4)·n - 25 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0235 | ramp | (17/8)·n - 8  →  (1/16)·n^2 + (5/8)·n - 24 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0234 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0234 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0234 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0) |
| n^2 | 0.0234 | ramp | 2·n - 4  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0234 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i3, i2] (i0=0, i3=7) |
| n^2 | 0.0234 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0234 | ramp | (41/8)·n - 76  →  (1/16)·n^2 + (3/4)·n - 10 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0232 | ramp | (25/8)·n - 21  →  (1/16)·n^2 + (3/4)·n - 26 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0232 | ramp | 3·n - 18  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=7) |
| n^2 | 0.0232 | ramp | 3·n - 18  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.0232 | ramp | 3·n - 18  →  (1/16)·n^2 + (3/4)·n - 28 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^2 | 0.023 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.023 | ramp | (5/4)·n + 2  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.0229 | ramp | (5/4)·n - 1  →  (1/16)·n^2 + (5/8)·n - 45 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0229 | ramp | (9/8)·n + 3  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0228 | ramp | (9/4)·n - 6  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.0228 | ramp | (9/4)·n - 6  →  (1/16)·n^2 + (3/4)·n - 46 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=6) |
| n^2 | 0.0228 | ramp | (9/4)·n - 9  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=7) |
| n^2 | 0.0227 | ramp | (9/4)·n - 10  →  (1/16)·n^2 + (5/8)·n - 45 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0) |
| n^2 | 0.0227 | ramp | (17/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0227 | ramp | (17/8)·n - 4  →  (1/16)·n^2 + (3/4)·n - 49 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=7) |
| n^2 | 0.0227 | ramp | (17/8)·n - 6  →  (1/16)·n^2 + (3/4)·n - 51 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^2 | 0.0226 | ramp | 2·n - 3  →  (1/16)·n^2 + (3/4)·n - 53 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.0224 | ramp | 3·n - 17  →  (1/16)·n^2 + (3/4)·n - 53 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=8) |
| n^2 | 0.0222 | ramp | (5/4)·n  →  (1/16)·n^2 + (3/4)·n - 80 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^2 | 0.0156 | ramp | 22  →  (1/16)·n^2 + (-3/8)·n - 5 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i2, i2] (i0=0) |
| n^1.5 | 4.71 | ramp | 4  →  (9/8)·n - 9 | 7·n - 91 | 5.25·n^-2 | read A[i1, i5] (i0=0, i5=0) |
| n^1.5 | 2.62 | ramp | 3  →  (9/8)·n - 2 | (15/4)·n - 15 | 2.81·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 2.62 | ramp | 3  →  (9/8)·n - 2 | (15/4)·n - 15 | 2.81·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^1.5 | 2.3 | ramp | 13  →  (9/8)·n - 8 | (27/8)·n - 54 | 2.53·n^-2 | read A[i1, i5] (i0=0, i4=0); read A[i1, i5] (i0=0) |
| n^1.5 | 1.03 | ramp | 5  →  (9/8)·n - 8 | (3/2)·n - 12 | 1.12·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 0.872 | ramp | 11  →  (9/8)·n - 7 | (5/4)·n - 10 | 0.938·n^-2 | read A[i5, i4] (i0=0, i5=0); read A[i5, i4] (i0=0) |
| n^1.5 | 0.685 | ramp | 5  →  (9/8)·n - 8 | n - 10 | 0.75·n^-2 | read A[i1, i5] (i0=0, i4=0, i5=0) |
| n^1.5 | 0.62 | ramp | 4  →  (9/8)·n - 1 | (7/8)·n - 2 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.601 | ramp | 4  →  (9/8)·n - 8 | (7/8)·n - 7 | 0.656·n^-2 | read A[i5, i4] (i0=0, i4=7, i5=0) |
| n^1.5 | 0.587 | ramp | 11  →  (9/8)·n - 16 | (7/8)·n - 14 | 0.656·n^-2 | read A[i1, i5] (i0=0) |
| n^1.5 | 0.579 | ramp | n + 4  →  (9/8)·n - 1 | (5/8)·n - 20 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.54 | ramp | (1/4)·n + 3  →  n - 1 | (3/4)·n - 18 | 0.562·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.532 | ramp | 4  →  (9/8)·n - 1 | (3/4)·n - 1 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0) |
| n^1.5 | 0.512 | ramp | 3  →  (9/8)·n - 10 | (3/4)·n - 6 | 0.562·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.475 | ramp | (3/8)·n + 3  →  n - 1 | (5/8)·n - 15 | 0.469·n^-2 | read A[i5, i4] (i0=0) |
| n^1.5 | 0.174 | ramp | 11  →  (9/8)·n - 7 | (1/4)·n - 2 | 0.188·n^-2 | read A[i5, i4] (i0=0, i4=6, i5=0); read A[i5, i4] (i0=0, i4=6) |
| n^1.5 | 0.156 | ramp | 2·n - 2  →  (17/8)·n - 8 | (1/8)·n - 5 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=7) |
| n^1.5 | 0.122 | ramp | (9/8)·n + 4  →  (5/4)·n - 1 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=9, i5=7) |
| n^1.5 | 0.119 | ramp | n + 3  →  (9/8)·n - 1 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=8, i5=7) |
| n^1.5 | 0.116 | ramp | n + 4  →  (9/8)·n - 1 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=8, i5=6) |
| n^1.5 | 0.116 | ramp | n + 4  →  (9/8)·n - 1 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i5=0) |
| n^1.5 | 0.0872 | ramp | 11  →  (9/8)·n - 7 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0, i5=0) |
| n^1.5 | 0.0845 | ramp | 13  →  (9/8)·n - 14 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1.5 | 0.0839 | ramp | 11  →  (9/8)·n - 16 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i5] (i0=0, i4=0) |
| n^1.5 | 0.0839 | ramp | 11  →  (9/8)·n - 16 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i4=0) |
| n^1.5 | 0.0631 | ramp | (1/4)·n + 4  →  (3/8)·n - 1 | (1/8)·n - 4 | 0.0938·n^-2 | read A[i5, i4] (i0=0, i1=2, i5=0) |
| n^1 | 30.3 | level | 3 | (35/2)·n - 140 | 13.1·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 30.3 | level | 3 | (35/2)·n - 140 | 13.1·n^-2 | read A[i1, i5] (i0=0) |
| n^1 | 14.8 | level | 2 | (21/2)·n | 7.88·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 13.9 | level | 3 | 8·n - 64 | 6·n^-2 | write A[i1, i4] (i0=0) |
| n^1 | 12.1 | level | 3 | 7·n - 56 | 5.25·n^-2 | read A[i1, i5] (i0=0) |
| n^1 | 10.5 | level | 1 | (21/2)·n - 28 | 7.88·n^-2 | read A[i1, i5] (i0=0) |
| n^1 | 7 | level | 1 | 7·n - 35 | 5.25·n^-2 | read A[i1, i3] (i0=0, i3=0) |
| n^1 | 4 | level | 4 | 2·n - 25 | 1.5·n^-2 | read A[i1, i5] (i0=0, i1=9, i4=0, i5=0); read A[i1, i3] (i0=0) (+1) |
| n^1 | 1.52 | level | 3 | (7/8)·n - 7 | 0.656·n^-2 | read A[i5, i4] (i0=0, i1=1, i5=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 20 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 18 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 16 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 14 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 12 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 27 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=8) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 24 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=7) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 22 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=6) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 10 | 6 | 4.5·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 7 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 6 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 5 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 4 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 3 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 2 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.5 | level | (1/16)·n^2 + (3/4)·n - 8 | 6 | 4.5·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 5 | 3.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 1 | level | (1/16)·n^2 + (5/8)·n - 10 | 4 | 3·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.656·n^-2 | read A[i1, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.75 | level | (1/16)·n^2 + (5/8)·n - 12 | 3 | 2.25·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.75 | level | (1/16)·n^2 + (3/4)·n - 9 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.5 | level | (1/16)·n^2 + (5/8)·n - 14 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 21 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 19 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 17 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 15 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 13 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 18 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 14 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 12 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 10 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 18 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 14 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 12 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 10 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 20 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 18 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 16 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 14 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 12 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 28 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 25 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 22 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 24 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 22 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 23 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 20 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 20 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 48 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=15, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 45 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 42 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 39 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 36 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 33 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 30 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 27 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 24 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=7, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 22 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=6, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-3/2) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (9/8)·n + (-51/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (-21/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-9/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (-53/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (-69/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + n + (-21/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/4)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-107/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + n + (-25/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (9/8)·n + (-99/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-139/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-99/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (9/8)·n + (-99/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-139/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-115/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + n + (-29/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-171/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (-85/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + n + (-29/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-171/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-131/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-203/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-43/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-9/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (11/8)·n + (-43/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-171/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-203/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + n + (-29/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-41/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 8 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (7/8)·n + (-131/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-49/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 26 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (13/8)·n + (-235/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-57/4) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 11 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0, i5=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 7 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 10 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0, i5=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 7 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 8 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 7 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 9 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=7) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 7 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 6 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 8 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 8 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/2)·n + (-9/16) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (5/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (1/2)·n + (-73/16) | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 8.84 | level | (25/8)·n - 22 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.66 | level | 3·n - 20 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.48 | level | (23/8)·n - 18 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.29 | level | (11/4)·n - 16 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 8.1 | level | (21/8)·n - 14 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.91 | level | (5/2)·n - 12 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.71 | level | (19/8)·n - 10 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.71 | level | (19/8)·n - 11 | 5 | 3.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 7.5 | level | (9/4)·n - 8 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.5 | level | (9/4)·n - 9 | 5 | 3.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 7.29 | level | (17/8)·n - 6 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 7.29 | level | (17/8)·n - 7 | 5 | 3.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 7.07 | level | 2·n - 5 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.85 | level | (15/8)·n - 4 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.61 | level | (7/4)·n - 3 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.37 | level | (13/8)·n - 2 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 6.32 | level | (5/2)·n - 13 | 4 | 3·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 6.12 | level | (3/2)·n - 1 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.86 | level | (11/8)·n | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.66 | level | 2·n - 6 | 4 | 3·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 5.59 | level | (5/4)·n + 1 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.3 | level | (25/8)·n - 22 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i1=25, i2=8, i3=0); read A[i3, i2] (i0=0, i1=25, i2=8, i3=7) (+1) |
| n^0.5 | 5.3 | level | (9/8)·n + 2 | 5 | 3.75·n^-3 | read A[i3, i2] (i0=0) |
| n^0.5 | 5.2 | level | 3·n - 20 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 5.09 | level | (23/8)·n - 18 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 5 | level | n + 3 | 5 | 3.75·n^-3 | read A[i5, i4] (i0=0, i4=8) |
| n^0.5 | 5 | level | n + 2 | 5 | 3.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 4.97 | level | (11/4)·n - 16 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.86 | level | (21/8)·n - 14 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.86 | level | (21/8)·n - 15 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 4.74 | level | (5/2)·n - 12 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.62 | level | (19/8)·n - 10 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.5 | level | (9/4)·n - 8 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0, i2=8, i3=7) (+1) |
| n^0.5 | 4.37 | level | (17/8)·n - 6 | 3 | 2.25·n^-3 | read A[i3, i2] (i0=0, i1=17, i2=8, i3=0); read A[i3, i2] (i0=0, i1=17, i2=8, i3=7) (+1) |
| n^0.5 | 4.24 | level | (9/8)·n + 1 | 4 | 3·n^-3 | read A[i2, i2] (i0=0, i1=9, i2=0); read A[i2, i2] (i0=0, i1=9, i2=1) (+2) |
| n^0.5 | 4.11 | level | (15/8)·n - 5 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 4.06 | level | (33/8)·n - 45 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i1=33, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 4 | level | 4·n - 42 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.94 | level | (31/8)·n - 39 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.87 | level | (15/4)·n - 36 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.81 | level | (29/8)·n - 33 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.74 | level | (7/8)·n + 2 | 4 | 3·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 3.74 | level | (7/2)·n - 30 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.67 | level | (27/8)·n - 27 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.61 | level | (13/4)·n - 24 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.54 | level | (25/8)·n - 21 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i1=25, i2=16, i3=6); read A[i3, i2] (i0=0, i1=25, i2=16, i3=7) |
| n^0.5 | 3.52 | level | (11/8)·n - 1 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 3.46 | level | 3·n - 19 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.39 | level | (23/8)·n - 17 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.35 | level | (5/4)·n | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 3.35 | level | (5/4)·n | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0, i1=10, i2=0); read A[i2, i2] (i0=0, i1=10, i2=1) (+1) |
| n^0.5 | 3.32 | level | (11/4)·n - 15 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.32 | level | (11/4)·n - 17 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 3.24 | level | (21/8)·n - 13 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.18 | level | (9/8)·n + 1 | 3 | 2.25·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 3.16 | level | (5/2)·n - 11 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3.08 | level | (19/8)·n - 9 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3 | level | (9/4)·n - 7 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 3 | level | n + 2 | 3 | 2.25·n^-3 | read A[i5, i4] (i0=0, i1=8, i4=0, i5=0); read A[i5, i4] (i0=0, i1=8, i4=0, i5=6) (+1) |
| n^0.5 | 2.92 | level | (17/8)·n - 5 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=16, i3=7); read A[i3, i2] (i0=0) |
| n^0.5 | 2.92 | level | (17/8)·n - 7 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=17, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 2.83 | level | 2·n - 5 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.83 | level | 2·n - 4 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=16, i4=0, i5=7); read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 2.74 | level | (15/8)·n - 3 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.74 | level | (15/8)·n - 4 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.65 | level | (7/4)·n - 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.65 | level | (7/4)·n - 3 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.65 | level | (7/4)·n - 4 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 2.6 | level | (3/4)·n + 2 | 3 | 2.25·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 2.55 | level | (13/8)·n - 1 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.55 | level | (13/8)·n - 2 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.45 | level | (3/2)·n | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.45 | level | (3/2)·n - 1 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.45 | level | (3/2)·n - 2 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 2.35 | level | (11/8)·n + 1 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i5=7); read A[i5, i4] (i0=0) |
| n^0.5 | 2.35 | level | (11/8)·n | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.24 | level | (5/4)·n + 1 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.24 | level | (5/4)·n + 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=10, i4=6, i5=7); read A[i5, i4] (i0=0, i4=6) |
| n^0.5 | 2.12 | level | (9/8)·n + 2 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=0); read A[i3, i2] (i0=0) |
| n^0.5 | 2.12 | level | (9/8)·n + 3 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=9, i4=7, i5=7); read A[i5, i4] (i0=0, i4=7) |
| n^0.5 | 2.12 | level | (9/8)·n + 2 | 2 | 1.5·n^-3 | read A[i3, i2] (i0=0, i1=9, i2=8, i3=7); read A[i2, i2] (i0=0, i1=9, i2=8) |
| n^0.5 | 2 | level | n + 3 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=8, i4=8, i5=6); read A[i5, i4] (i0=0, i4=8, i5=0) |
| n^0.5 | 2 | level | n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 2 | level | n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.87 | level | (7/8)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.77 | level | (25/8)·n - 22 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=25, i2=8) |
| n^0.5 | 1.73 | level | (3/4)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.73 | level | 3·n - 20 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.73 | level | 3·n - 21 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.7 | level | (23/8)·n - 18 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.7 | level | (23/8)·n - 19 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.7 | level | (23/8)·n - 19 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.66 | level | (11/4)·n - 16 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.66 | level | (11/4)·n - 17 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.62 | level | (21/8)·n - 14 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.62 | level | (21/8)·n - 15 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.58 | level | (5/8)·n + 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 1.58 | level | (5/2)·n - 12 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.58 | level | (5/2)·n - 13 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.54 | level | (19/8)·n - 10 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.54 | level | (19/8)·n - 11 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.5 | level | (9/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=18, i2=7) |
| n^0.5 | 1.5 | level | (9/4)·n - 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.5 | level | (9/4)·n - 8 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=18, i2=8) |
| n^0.5 | 1.46 | level | (17/8)·n - 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=17, i2=7) |
| n^0.5 | 1.46 | level | (17/8)·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=17, i2=8) |
| n^0.5 | 1.41 | level | 2·n - 3 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=8, i5=7) |
| n^0.5 | 1.41 | level | 2·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=16, i2=8) |
| n^0.5 | 1.41 | level | (1/2)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=4, i2=0); read A[i2, i2] (i0=0, i1=4, i2=1) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.41 | level | 2·n - 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=16, i2=7) |
| n^0.5 | 1.41 | level | 2·n - 5 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.37 | level | (15/8)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=15, i2=7) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.37 | level | (15/8)·n - 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.37 | level | (15/8)·n - 4 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.32 | level | (7/4)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 4 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.32 | level | (7/4)·n - 3 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.27 | level | (13/8)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=8) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 3 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.27 | level | (13/8)·n - 2 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=7) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=12, i2=8) |
| n^0.5 | 1.22 | level | (3/8)·n + 1 | 2 | 1.5·n^-3 | read A[i2, i2] (i0=0, i1=3, i2=0); read A[i2, i2] (i0=0, i1=3, i2=1) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 2 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.22 | level | (3/2)·n - 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=11, i2=8) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=11, i2=7) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.17 | level | (11/8)·n - 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=11, i2=1) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.12 | level | (5/4)·n - 1 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=9, i5=7) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=10, i2=7) |
| n^0.5 | 1.12 | level | (5/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=10, i2=8) |
| n^0.5 | 1.12 | level | (5/4)·n + 1 | 1 | 0.75·n^-3 | read A[i3, i2] (i0=0, i2=8, i3=7) |
| n^0.5 | 1.06 | level | (9/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 1.06 | level | (9/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=8, i5=7) |
| n^0.5 | 1.06 | level | (9/8)·n + (-1/4) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 7 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n + (-65/8) | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 6 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 5 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 4 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 3 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n - 1 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i4=0) |
| n^0.5 | 1.06 | level | (9/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=9, i2=7) |
| n^0.5 | 1 | level | n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 1 | level | n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 1 | level | (1/4)·n + 2 | 2 | 1.5·n^-3 | read A[i5, i4] (i0=0, i1=2, i4=6, i5=0); read A[i5, i4] (i0=0, i1=2, i4=6, i5=1) |
| n^0.5 | 1 | level | n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1 | level | n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 1 | level | n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.935 | level | (7/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.935 | level | (7/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.791 | level | (5/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i2=0); read A[i2, i2] (i0=0, i2=1) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=5, i2=1) |
| n^0.5 | 0.791 | level | (5/8)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.707 | level | (1/2)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i5=0) |
| n^0.5 | 0.612 | level | (3/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0) |
| n^0.5 | 0.612 | level | (3/8)·n | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=2, i5=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 3 | 1 | 0.75·n^-3 | read A[i5, i4] (i0=0, i1=2, i5=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 1 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=2, i2=0) |
| n^0 | 28 | level | 1 | 28 | 21·n^-3 | read A[i1, i5] (i0=0, i5=0) |
| n^0 | 3.32 | level | 11 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 3.16 | level | 10 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 3 | level | 9 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.83 | level | 8 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.65 | level | 7 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.45 | level | 6 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i2, i2] (i0=0, i1=2, i2=1) |

*bin 100: finite-size slope 3.33 at n≈256–312; converges to n^3; bin 101: finite-size slope 3.26 at n≈256–312; converges to n^3; bin 104: finite-size slope 3.27 at n≈256–312; converges to n^3.*

Row-panel and trailing-submatrix re-reads (`read A[i3,i2]`, `read A[i5,i4]`) ramp to (1/16)n^2 + (3/4)n lines with cubic populations — d = 4.0, headroom +1.0. At the anchor sizes (n ≈ 256–312) the finite-size slope of these families still reads ≈ 4.4 because the negative n^2 corrections in their populations have not yet died out; the exact degrees settle it at 4.0.

## lu_decomp — infinite-repeat  [`exact`]

Accesses $A(n) = (4/3)·n^3 + (1/2)·n^2 + (1/6)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0101·n^4  +  0.0166·n^3.5  +  2.36·n^3  +  0.605·n^2.5  +  24.1·n^2  +  0.8·n^1.5  +  40.6·n^1  +  867·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.00766 | ramp | 72  →  (1/8)·n^2 | (1/32)·n^3 + (-121/128)·n^2 + (39/16)·n + 75 | 0.0234 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^4 | 0.00124 | ramp | 104  →  (1/8)·n^2 + (-3/4)·n | (1/192)·n^3 + (-25/128)·n^2 + (71/48)·n + 5 | 0.00391 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^4 | 0.00117 | ramp | 135  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/192)·n^3 + (-35/128)·n^2 + (143/48)·n + 14 | 0.00391 | read A[i3, i4] (i0=0) |
| n^3.5 | 0.0125 | ramp | 8  →  (1/4)·n + 2 | (1/32)·n^3 + (-121/128)·n^2 + (41/16)·n + 73 | 0.0234 | read A[i1, i4] (i0=0, i4=7); read A[i1, i4] (i0=0) |
| n^3.5 | 0.0021 | ramp | 10  →  (1/4)·n + 2 | (1/192)·n^3 + (-9/64)·n^2 + (-17/24)·n + 26 | 0.00391 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00204 | ramp | 10  →  (1/4)·n + 2 | (1/192)·n^3 + (-25/128)·n^2 + (71/48)·n + 5 | 0.00391 | read A[i1, i4] (i0=0) |
| n^3 | 0.442 | level | 3 | (49/192)·n^3 + (-413/128)·n^2 + (245/48)·n + 35 | 0.191 | write A[i3, i4] (i0=0, i4=6); read A[i3, i1] (i0=0) |
| n^3 | 0.442 | level | 3 | (49/192)·n^3 + (-469/128)·n^2 + (287/48)·n + 56 | 0.191 | write A[i3, i4] (i0=0) |
| n^3 | 0.388 | level | 3 | (43/192)·n^3 + (-193/64)·n^2 + (65/12)·n + 35 | 0.168 | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^3 | 0.255 | level | 1 | (49/192)·n^3 + (-365/128)·n^2 + (191/48)·n + 20 | 0.191 | read A[i3, i4] (i0=0) |
| n^3 | 0.0889 | ramp | 32  →  (1/8)·n^2 | (3/8)·n^2 + (-9/8)·n - 65 | 0.281/n | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^3 | 0.0872 | ramp | 22  →  (1/8)·n^2 + (-1/4)·n + 2 | (3/8)·n^2 + (-23/8)·n - 5 | 0.281/n | read A[i3, i1] (i0=0) |
| n^3 | 0.0833 | level | 4 | (1/24)·n^3 + (-29/32)·n^2 + (53/24)·n + 26 | 0.0312 | read A[i3, i1] (i0=0) |
| n^3 | 0.0738 | ramp | 57  →  (1/8)·n^2 + (-1/8)·n | (5/16)·n^2 + (-5/4)·n - 60 | 0.234/n | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^3 | 0.0631 | level | 3 | (7/192)·n^3 + (-119/128)·n^2 + (287/48)·n - 7 | 0.0273 | read A[i3, i1] (i0=0) |
| n^3 | 0.0631 | level | 3 | (7/192)·n^3 + (-119/128)·n^2 + (287/48)·n - 7 | 0.0273 | write A[i3, i4] (i0=0) |
| n^3 | 0.0631 | level | 3 | (7/192)·n^3 + (-35/128)·n^2 + (-7/48)·n | 0.0273 | write A[i3, i4] (i0=0) |
| n^3 | 0.0541 | level | 3 | (1/32)·n^3 + (-3/64)·n^2 + (-13/8)·n | 0.0234 | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^3 | 0.0541 | level | 3 | (1/32)·n^3 + (-27/128)·n^2 + (-5/16)·n | 0.0234 | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^3 | 0.0365 | level | 1 | (7/192)·n^3 + (-7/128)·n^2 + (-91/48)·n | 0.0273 | read A[i3, i4] (i0=0) |
| n^3 | 0.0183 | ramp | (1/5898240)·n^6 + (-7/24576)·n^5 + (7345/36864)·n^4 + (-28525/384)·n^3 + (22417669/1440)·n^2 + (-41731969/24)·n + 80868862  →  (1/8)·n^2 - 2 | (1/16)·n^2 + (-11/4)·n + 30 | 0.0469/n | read A[i3, i4] (i0=0, i1=0) |
| n^3 | 0.0144 | ramp | 54  →  (1/8)·n^2 + (-3/4)·n | (1/16)·n^2 + (-5/8)·n - 6 | 0.0469/n | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^3 | 0.0144 | ramp | 54  →  (1/8)·n^2 + (-3/4)·n | (1/16)·n^2 + (-3/4)·n - 4 | 0.0469/n | read A[i3, i4] (i0=0, i4=0) |
| n^3 | 0.0143 | ramp | 20  →  (1/8)·n^2 + (-7/8)·n + 2 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | read A[i3, i1] (i0=0) |
| n^3 | 0.0141 | ramp | 37  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/16)·n^2 + (-7/8)·n - 2 | 0.0469/n | read A[i3, i1] (i0=0) |
| n^3 | 0.0141 | ramp | 78  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/16)·n^2 + (-7/8)·n - 15 | 0.0469/n | read A[i3, i4] (i0=0, i4=7) |
| n^3 | 0.0141 | ramp | 77  →  (1/8)·n^2 + (-7/8)·n - 3 | (1/16)·n^2 + (-7/8)·n - 15 | 0.0469/n | read A[i3, i4] (i0=0) |
| n^3 | 0.0123 | ramp | 54  →  (1/8)·n^2 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | read A[i1, i2] (i0=0) |
| n^3 | 0.0121 | ramp | (-1/5898240)·n^6 + (7/24576)·n^5 + (-7345/36864)·n^4 + (28525/384)·n^3 + (-22417399/1440)·n^2 + (41731987/24)·n - 80868864  →  (1/8)·n^2 | (5/128)·n^2 + (-7/16)·n - 12 | 0.0293/n | read A[i3, i4] (i0=0, i1=0) |
| n^3 | 0.0102 | ramp | 70  →  (1/8)·n^2 - 2 | (3/64)·n^2 + (-7/4)·n + 16 | 0.0352/n | read A[i3, i4] (i0=0) |
| n^3 | 0.00902 | level | 3 | (1/192)·n^3 + (-1/128)·n^2 + (-13/48)·n | 0.00391 | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^3 | 0.00902 | level | 3 | (1/192)·n^3 + (95/128)·n^2 + (-289/48)·n + 5 | 0.00391 | read A[i3, i1] (i0=0, i1=0, i3=0); read A[i3, i1] (i0=0, i1=0, i4=7) (+2) |
| n^3 | 0.00246 | ramp | (1/3932160)·n^6 + (-7/16384)·n^5 + (7345/24576)·n^4 + (-28525/256)·n^3 + (2802203/120)·n^2 + (-41731977/16)·n + 121303296  →  (1/8)·n^2 | (1/128)·n^2 + (-1/16)·n - 3 | 0.00586/n | read A[i3, i4] (i0=0, i1=0) |
| n^3 | 0.00232 | ramp | (-1/3932160)·n^6 + (7/16384)·n^5 + (-7345/24576)·n^4 + (28525/256)·n^3 + (-5604361/240)·n^2 + (41731985/16)·n - 121303296  →  (1/8)·n^2 - n + 9 | (1/128)·n^2 + (-3/16)·n | 0.00586/n | read A[i3, i4] (i0=0, i1=0) |
| n^3 | 0.00232 | ramp | (-1/2949120)·n^6 + (7/12288)·n^5 + (-7345/18432)·n^4 + (28525/192)·n^3 + (-44834933/1440)·n^2 + (10432996/3)·n - 161737727  →  (1/8)·n^2 - n + 9 | (1/128)·n^2 + (-3/16)·n | 0.00586/n | read A[i3, i4] (i0=0, i1=0) |
| n^3 | 0.00167 | ramp | 101  →  (1/8)·n^2 + (-3/4)·n - 3 | (1/128)·n^2 + (-5/16)·n + 3 | 0.00586/n | read A[i3, i4] (i0=0) |
| n^3 | 0.00167 | ramp | 100  →  (1/8)·n^2 + (-7/8)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.00586/n | read A[i1, i2] (i0=0) |
| n^3 | 0.0016 | ramp | 164  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i3, i4] (i0=0, i3=0) |
| n^3 | 0.00157 | ramp | 130  →  (1/8)·n^2 + (-15/8)·n + 5 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i3, i4] (i0=0) |
| n^2.5 | 0.153 | ramp | 8  →  (1/4)·n + 2 | (3/8)·n^2 + (-9/8)·n - 78 | 0.281/n | read A[i1, i4] (i0=0, i4=7); read A[i1, i4] (i0=0) |
| n^2.5 | 0.152 | ramp | 3  →  (1/4)·n + 1 | (3/8)·n^2 + (-9/8)·n | 0.281/n | read A[i1, i1] (i0=0) |
| n^2.5 | 0.127 | ramp | 8  →  (1/4)·n + 2 | (5/16)·n^2 + (-5/4)·n - 60 | 0.234/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.026 | ramp | 8  →  (1/4)·n + 2 | (1/16)·n^2 + (1/4)·n - 20 | 0.0469/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.0259 | ramp | 8  →  (1/4)·n + 2 | (1/16)·n^2 + (1/8)·n - 18 | 0.0469/n | read A[i1, i4] (i0=0, i4=6) |
| n^2.5 | 0.0258 | ramp | 5  →  (1/4)·n + 1 | (1/16)·n^2 + (1/4)·n - 6 | 0.0469/n | read A[i1, i1] (i0=0) |
| n^2.5 | 0.0251 | ramp | 8  →  (1/4)·n + 2 | (1/16)·n^2 + (-5/8)·n - 6 | 0.0469/n | read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.0251 | ramp | 8  →  (1/4)·n + 2 | (1/16)·n^2 + (-5/8)·n - 6 | 0.0469/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.0248 | ramp | 5  →  (1/4)·n + 1 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0469/n | read A[i1, i1] (i0=0, i3=1); read A[i1, i1] (i0=0) |
| n^2.5 | 0.0149 | ramp | 5  →  (1/4)·n - 1 | (3/64)·n^2 + (-7/4)·n + 16 | 0.0352/n | read A[i1, i4] (i0=0, i3=0, i4=7); read A[i1, i4] (i0=0, i3=0) |
| n^2.5 | 0.00247 | ramp | 7  →  (1/4)·n - 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.00586/n | read A[i1, i4] (i0=0, i3=0) |
| n^2.5 | 0.00247 | ramp | 7  →  (1/4)·n - 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.00586/n | read A[i1, i4] (i0=0, i3=0) |
| n^2 | 2.17 | level | 3 | (5/4)·n^2 + (-45/8)·n - 35 | 0.938/n | write A[i3, i4] (i0=0) |
| n^2 | 1.88 | level | 1 | (15/8)·n^2 + (-15/4)·n - 20 | 1.41/n | read A[i3, i4] (i0=0) |
| n^2 | 1.62 | level | 3 | (15/16)·n^2 - 5·n - 20 | 0.703/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-9/2)·n - 20 | 0.656/n | read A[i3, i1] (i0=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-63/8)·n + 7 | 0.656/n | write A[i3, i4] (i0=0, i1=0) |
| n^2 | 1.33 | level | 2 | (15/16)·n^2 + (5/4)·n | 0.703/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2 | 0.972 | level | 2 | (11/16)·n^2 + (-3/8)·n | 0.516/n | read A[i3, i1] (i0=0) |
| n^2 | 0.938 | level | 1 | (15/16)·n^2 + (5/4)·n | 0.703/n | write A[i3, i4] (i0=0) |
| n^2 | 0.884 | level | 2 | (5/8)·n^2 + (5/4)·n | 0.469/n | read A[i3, i1] (i0=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 + (-63/8)·n + 35 | 0.656/n | read A[i1, i2] (i0=0); write A[i1, i2] (i0=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-7/8)·n - 21 | 0.328/n | read A[i3, i1] (i0=0, i4=7) |
| n^2 | 0.75 | level | 1 | (3/4)·n^2 + (-3/4)·n | 0.562/n | read A[i3, i4] (i0=0, i4=0); write A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-15/8)·n - 9 | 0.281/n | read A[i1, i4] (i0=0, i4=6) |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-9/8)·n - 15 | 0.281/n | read A[i3, i1] (i0=0, i4=6) |
| n^2 | 0.619 | level | 2 | (7/16)·n^2 + (-7/8)·n | 0.328/n | write A[i3, i1] (i0=0) |
| n^2 | 0.53 | level | 2 | (3/8)·n^2 + (-3/8)·n - 12 | 0.281/n | read A[i1, i4] (i0=0, i3=0, i4=0); read A[i1, i4] (i0=0, i4=0) |
| n^2 | 0.442 | level | 2 | (5/16)·n^2 + (15/8)·n | 0.234/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 + (-1/4)·n - 14 | 0.188/n | read A[i3, i1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 5 | 0.75·n^-2 | read A[i1, i2] (i0=0, i1=0); read A[i3, i1] (i0=0, i1=0, i3=0) (+3) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 2 | n - 2 | 0.75·n^-2 | read A[i1, i2] (i0=0, i1=1); read A[i3, i1] (i0=0, i1=1, i3=0) (+1) |
| n^2 | 0.354 | level | 2 | (1/4)·n^2 + (-5/2)·n + 4 | 0.188/n | read A[i3, i1] (i0=0) |
| n^2 | 0.331 | ramp | (1/8)·n^2 - n + 18  →  (1/8)·n^2 - 1 | n - 18 | 0.75·n^-2 | read A[i3, i4] (i0=0, i1=0, i4=7) |
| n^2 | 0.312 | level | 1 | (5/16)·n^2 + (-25/8)·n + 5 | 0.234/n | write A[i3, i4] (i0=0) |
| n^2 | 0.311 | ramp | 20  →  (1/8)·n^2 + (-1/4)·n + 2 | (7/4)·n - 16 | 1.31·n^-2 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^2 | 0.273 | level | 1 | (35/128)·n^2 + (25/16)·n | 0.205/n | read A[i3, i4] (i0=0) |
| n^2 | 0.25 | level | 4 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0938/n | read A[i3, i1] (i0=0, i1=0) |
| n^2 | 0.24 | ramp | (-1/3932160)·n^6 + (7/16384)·n^5 + (-7345/24576)·n^4 + (28525/256)·n^3 + (-5604361/240)·n^2 + (41731985/16)·n - 121303296  →  (1/8)·n^2 | (3/4)·n - 2 | 0.562·n^-2 | read A[i3, i4] (i0=0, i1=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-7/8)·n - 1 | 0.0938/n | read A[i3, i1] (i0=0, i4=1) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 - n | 0.0938/n | read A[i3, i1] (i0=0, i4=0); write A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 + (-9/8)·n + 1 | 0.0938/n | write A[i3, i4] (i0=0, i1=0) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 + (-1/4)·n + 1 | 0.0938/n | read A[i1, i4] (i0=0, i3=0, i4=6); read A[i1, i4] (i0=0, i4=6) (+1) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 + (-5/4)·n + 2 | 0.0938/n | read A[i3, i1] (i0=0) |
| n^2 | 0.157 | ramp | 20  →  (1/8)·n^2 | (7/8)·n - 7 | 0.656·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.134 | ramp | 30  →  (1/8)·n^2 + (-1/8)·n | (3/4)·n - 11 | 0.562·n^-2 | read A[i3, i4] (i0=0) |
| n^2 | 0.133 | ramp | 20  →  (1/8)·n^2 + (-3/8)·n + 2 | (3/4)·n - 7 | 0.562·n^-2 | read A[i3, i1] (i0=0) |
| n^2 | 0.125 | level | 1 | (1/8)·n^2 - 1 | 0.0938/n | write A[i1, i2] (i0=0); read A[i3, i4] (i0=0, i4=0) (+1) |
| n^2 | 0.112 | ramp | 55  →  (1/8)·n^2 + (-1/8)·n - 2 | (5/8)·n - 10 | 0.469·n^-2 | read A[i3, i4] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (1/4)·n - 6 | 0.0469/n | read A[i3, i1] (i0=0, i4=6) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-5/8)·n - 6 | 0.0469/n | read A[i3, i1] (i0=0, i4=7) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0469/n | read A[i1, i4] (i0=0, i3=0, i4=0); read A[i1, i4] (i0=0, i4=0) |
| n^2 | 0.0884 | level | 2 | (1/16)·n^2 + (3/8)·n | 0.0469/n | write A[i3, i1] (i0=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-1/2)·n | 0.0469/n | write A[i1, i2] (i0=0); write A[i3, i4] (i0=0, i4=6) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-13/8)·n + 10 | 0.0469/n | write A[i1, i2] (i0=0) |
| n^2 | 0.0547 | level | 1 | (7/128)·n^2 + (5/16)·n | 0.041/n | read A[i3, i4] (i0=0, i4=6) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i1=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i1=0, i3=0) |
| n^2 | 0.0442 | level | (1/8)·n^2 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i1=0) |
| n^2 | 0.0394 | ramp | (1/3932160)·n^6 + (-7/16384)·n^5 + (7345/24576)·n^4 + (-28525/256)·n^3 + (2802203/120)·n^2 + (-41731977/16)·n + 121303296  →  (1/8)·n^2 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i1=0) |
| n^2 | 0.0392 | ramp | (-1/2949120)·n^6 + (7/12288)·n^5 + (-7345/18432)·n^4 + (28525/192)·n^3 + (-44834933/1440)·n^2 + (10432996/3)·n - 161737727  →  (1/8)·n^2 - n + 9 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i1=0) |
| n^2 | 0.0391 | level | 1 | (5/128)·n^2 + (-5/16)·n | 0.0293/n | read A[i3, i4] (i0=0) |
| n^2 | 0.0373 | ramp | (1/3932160)·n^6 + (-7/16384)·n^5 + (7345/24576)·n^4 + (-28525/256)·n^3 + (2802203/120)·n^2 + (-41731975/16)·n + 121303295  →  (1/8)·n^2 + (-7/8)·n + 14 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i1=0) |
| n^2 | 0.037 | ramp | (-1/2949120)·n^6 + (7/12288)·n^5 + (-7345/18432)·n^4 + (28525/192)·n^3 + (-44834933/1440)·n^2 + (10432996/3)·n - 161737727  →  (1/8)·n^2 - n + 9 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i1=0) |
| n^2 | 0.0222 | ramp | 54  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.0222 | ramp | 52  →  (1/8)·n^2 + (-3/4)·n - 2 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.0221 | ramp | 51  →  (1/8)·n^2 + (-7/8)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i4] (i0=0) |
| n^2 | 0.0221 | ramp | 51  →  (1/8)·n^2 + (-7/8)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0221 | ramp | 50  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i1] (i0=0, i3=0) |
| n^2 | 0.0221 | ramp | 50  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.0221 | ramp | 18  →  (1/8)·n^2 - n + 2 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i3, i1] (i0=0) |
| n^2 | 0.022 | ramp | 99  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i3=0, i4=7) |
| n^2 | 0.022 | ramp | 98  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i3=0) |
| n^2 | 0.0216 | ramp | 34  →  (1/8)·n^2 + (-15/8)·n + 7 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i1] (i0=0) |
| n^2 | 0.0215 | ramp | 74  →  (1/8)·n^2 + (-15/8)·n + 6 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i4=7) |
| n^2 | 0.0214 | ramp | 72  →  (1/8)·n^2 - 2·n + 8 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0) |
| n^2 | 0.00781 | level | 1 | (1/128)·n^2 + (-1/16)·n | 0.00586/n | read A[i3, i4] (i0=0, i4=6) |
| n^1.5 | 0.251 | ramp | 6  →  (1/4)·n | (3/4)·n - 12 | 0.562·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=7); read A[i1, i4] (i0=0, i3=0) |
| n^1.5 | 0.247 | ramp | 2  →  (1/8)·n + 1 | n - 1 | 0.75·n^-2 | read A[i1, i1] (i0=0, i3=0) |
| n^1.5 | 0.156 | ramp | 5  →  (1/8)·n + 2 | (5/8)·n - 10 | 0.469·n^-2 | read A[i1, i4] (i0=0, i3=0) |
| n^1.5 | 0.0418 | ramp | 6  →  (1/4)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i3=0) |
| n^1.5 | 0.0418 | ramp | 6  →  (1/4)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i3=0) |
| n^1.5 | 0.0312 | ramp | 5  →  (1/8)·n + 2 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=6) |
| n^1.5 | 0.0312 | ramp | 5  →  (1/8)·n + 2 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^1 | 11.3 | level | 2 | 8·n - 8 | 6·n^-2 | read A[i1, i2] (i0=0, i1=0); read A[i3, i1] (i0=0, i1=0, i3=0) (+4) |
| n^1 | 7 | level | 1 | 7·n + 1 | 5.25·n^-2 | read A[i1, i2] (i0=0, i1=0); write A[i1, i2] (i0=0, i1=0) (+2) |
| n^1 | 3.75 | level | 1 | (15/4)·n - 15 | 2.81·n^-2 | write A[i1, i2] (i0=0) |
| n^1 | 2.81 | ramp | (1/8)·n^2 - n + 11  →  (1/8)·n^2 | 8 | 6·n^-3 | read A[i3, i4] (i0=0, i1=0, i4=7) |
| n^1 | 1.62 | level | 1 | (13/8)·n - 13 | 1.22·n^-2 | write A[i1, i2] (i0=0) |
| n^1 | 1.41 | level | (1/8)·n^2 | 4 | 3·n^-3 | read A[i3, i1] (i0=0, i1=0) |
| n^1 | 1.3 | level | 3 | (3/4)·n - 6 | 0.562·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=6) |
| n^1 | 0.875 | level | 1 | (7/8)·n - 7 | 0.656·n^-2 | write A[i1, i2] (i0=0, i1=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.656·n^-2 | write A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.75 | level | 1 | (3/4)·n | 0.562·n^-2 | write A[i1, i2] (i0=0, i2=1) |
| n^1 | 0.707 | level | (1/8)·n^2 - n + 9 | 2 | 1.5·n^-3 | read A[i3, i4] (i0=0, i1=0, i3=6, i4=7); read A[i3, i4] (i0=0, i1=0, i3=7, i4=7) |
| n^1 | 0.707 | level | (1/8)·n^2 | 2 | 1.5·n^-3 | read A[i3, i4] (i0=0, i1=0, i3=0); read A[i3, i4] (i0=0, i1=0) |
| n^1 | 0.703 | ramp | (1/8)·n^2 - n + 15  →  (1/8)·n^2 | 2 | 1.5·n^-3 | read A[i3, i4] (i0=0, i1=0, i4=7) |
| n^1 | 0.625 | level | 1 | (5/8)·n | 0.469·n^-2 | read A[i3, i4] (i0=0, i3=6) |
| n^1 | 0.625 | level | 1 | (5/8)·n - 5 | 0.469·n^-2 | write A[i1, i2] (i0=0) |
| n^1 | 0.625 | level | 1 | (5/8)·n - 5 | 0.469·n^-2 | write A[i1, i2] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0, i4=7) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 10 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0, i4=7) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0, i3=0, i4=7) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 1 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - 2·n + 33 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0, i3=14, i4=7) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 17 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i1=0, i3=15, i4=7) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 2 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i1=1) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 - n + 9 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i1=0, i3=6) |
| n^1 | 0.25 | level | 1 | (1/4)·n - 3 | 0.188·n^-2 | write A[i1, i2] (i0=0) |
| n^1 | 0.125 | level | 1 | (1/8)·n | 0.0938·n^-2 | read A[i3, i4] (i0=0, i3=6, i4=6) |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0938·n^-2 | write A[i1, i2] (i0=0, i2=7) |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0938·n^-2 | write A[i1, i2] (i0=0, i1=0) |
| n^0 | 122 | level | 6 | 50 | 37.5·n^-3 | read A[i1, i4] (i0=0) |
| n^0 | 65.7 | level | 30 | 12 | 9·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 64.7 | ramp | 36  →  48 | 10 | 7.5·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 58.2 | level | 28 | 11 | 8.25·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 51 | level | 26 | 10 | 7.5·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 44.1 | level | 24 | 9 | 6.75·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 37.5 | level | 22 | 8 | 6·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 36.7 | level | 6 | 15 | 11.2·n^-3 | read A[i1, i4] (i0=0) |
| n^0 | 34.3 | level | 6 | 14 | 10.5·n^-3 | read A[i1, i4] (i0=0, i4=7); read A[i1, i4] (i0=0) |
| n^0 | 31.8 | level | 6 | 13 | 9.75·n^-3 | read A[i1, i4] (i0=0, i4=6) |
| n^0 | 26.8 | level | 20 | 6 | 4.5·n^-3 | read A[i3, i4] (i0=0, i4=0) |
| n^0 | 21.2 | level | 18 | 5 | 3.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 17.1 | level | 6 | 7 | 5.25·n^-3 | read A[i1, i4] (i0=0, i4=0) |
| n^0 | 14.7 | level | 6 | 6 | 4.5·n^-3 | read A[i1, i4] (i0=0, i4=6) |
| n^0 | 14.1 | level | 2 | 10 | 7.5·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 11.3 | level | 8 | 4 | 3·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 10.4 | level | 3 | 6 | 4.5·n^-3 | read A[i1, i1] (i0=0) |
| n^0 | 10 | level | 4 | 5 | 3.75·n^-3 | read A[i1, i4] (i0=0, i3=0) |
| n^0 | 8.49 | level | 18 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 7.94 | level | 7 | 3 | 2.25·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 7 | level | 49 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i3=0, i4=7) |
| n^0 | 6.56 | level | 43 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 6.48 | level | 42 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 6.4 | level | 41 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 5.66 | level | 32 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 5.66 | level | 8 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 5.29 | level | 28 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 5.29 | level | 7 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 5.1 | level | 26 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.9 | level | 6 | 2 | 1.5·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 4.9 | level | 24 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.9 | level | 6 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 4.69 | level | 22 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.47 | level | 20 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.47 | level | 20 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^0 | 4.47 | level | 5 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 4.47 | level | 5 | 2 | 1.5·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0); read A[i1, i1] (i0=0, i3=1) (+1) |
| n^0 | 4.24 | level | 18 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=0) |
| n^0 | 4.12 | level | 17 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^0 | 4 | level | 16 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=6) |
| n^0 | 4 | level | 4 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 4 | level | 16 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=0) |
| n^0 | 3.87 | level | 15 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.74 | level | 14 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.61 | level | 13 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.46 | level | 12 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.32 | level | 11 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.83 | level | 8 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=6) |
| n^0 | 2.83 | level | 2 | 2 | 1.5·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 2.65 | level | 7 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.45 | level | 6 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.45 | level | 6 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=7); read A[i1, i4] (i0=0, i3=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=6) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.75·n^-3 | read A[i1, i1] (i0=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i1=1, i3=0); read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 1 | level | 1 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^0 | 1 | level | 1 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |

Same structure as lu in a different schedule: submatrix re-reads at up to (1/8)n^2 lines, 0.0076·n^4 leading, headroom +1.0.

## lu_decomp — single-shot  [`exact`]

Accesses $A(n) = (4/3)·n^3 + (1/2)·n^2 + (1/6)·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.01·n^4  +  0.0166·n^3.5  +  2.34·n^3  +  0.631·n^2.5  +  22.6·n^2  +  0.831·n^1.5  +  940·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.00762 | ramp | 108  →  (1/8)·n^2 | (1/32)·n^3 + (-129/128)·n^2 + (37/16)·n + 93 | 0.0234 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^4 | 0.00124 | ramp | 104  →  (1/8)·n^2 + (-3/4)·n | (1/192)·n^3 + (-25/128)·n^2 + (71/48)·n + 5 | 0.00391 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^4 | 0.00117 | ramp | 135  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/192)·n^3 + (-35/128)·n^2 + (143/48)·n + 14 | 0.00391 | read A[i3, i4] (i0=0) |
| n^3.5 | 0.0145 | ramp | 10  →  (1/4)·n + 2 | (7/192)·n^3 + (-147/128)·n^2 + (77/48)·n + 119 | 0.0273 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00204 | ramp | 10  →  (1/4)·n + 2 | (1/192)·n^3 + (-25/128)·n^2 + (71/48)·n + 5 | 0.00391 | read A[i1, i4] (i0=0) |
| n^3 | 0.505 | level | 3 | (7/24)·n^3 + (-133/32)·n^2 + (49/12)·n + 84 | 0.219 | read A[i3, i1] (i0=0) |
| n^3 | 0.505 | level | 3 | (7/24)·n^3 + (-7/2)·n^2 + (119/24)·n + 35 | 0.219 | write A[i3, i4] (i0=0, i4=6); write A[i3, i4] (i0=0) |
| n^3 | 0.442 | level | 3 | (49/192)·n^3 + (-7/2)·n^2 + (14/3)·n + 56 | 0.191 | read A[i1, i4] (i0=0) |
| n^3 | 0.255 | level | 1 | (49/192)·n^3 + (-413/128)·n^2 + (245/48)·n + 35 | 0.191 | read A[i3, i4] (i0=0) |
| n^3 | 0.0889 | ramp | 57  →  (1/8)·n^2 | (3/8)·n^2 + (-9/8)·n - 78 | 0.281/n | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^3 | 0.0883 | ramp | 22  →  (1/8)·n^2 + (-1/8)·n + 2 | (3/8)·n^2 + (-15/8)·n - 9 | 0.281/n | read A[i3, i1] (i0=0) |
| n^3 | 0.0833 | level | 4 | (1/24)·n^3 + (-27/32)·n^2 + (7/12)·n + 36 | 0.0312 | read A[i3, i1] (i0=0) |
| n^3 | 0.0738 | ramp | 57  →  (1/8)·n^2 + (-1/8)·n | (5/16)·n^2 + (-5/4)·n - 60 | 0.234/n | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^3 | 0.0722 | level | 3 | (1/24)·n^3 + (-1/16)·n^2 + (-13/6)·n | 0.0312 | write A[i3, i4] (i0=0) |
| n^3 | 0.0631 | level | 3 | (7/192)·n^3 + (-7/32)·n^2 + (-7/12)·n | 0.0273 | read A[i1, i4] (i0=0) |
| n^3 | 0.0365 | level | 1 | (7/192)·n^3 + (-7/128)·n^2 + (-91/48)·n | 0.0273 | read A[i3, i4] (i0=0) |
| n^3 | 0.0151 | ramp | 72  →  (1/8)·n^2 | (1/16)·n^2 + (1/8)·n - 18 | 0.0469/n | read A[i3, i4] (i0=0, i3=0, i4=6); read A[i3, i4] (i0=0, i4=6) |
| n^3 | 0.0144 | ramp | 54  →  (1/8)·n^2 + (-3/4)·n | (1/16)·n^2 + (-5/8)·n - 6 | 0.0469/n | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^3 | 0.0144 | ramp | 54  →  (1/8)·n^2 + (-3/4)·n | (1/16)·n^2 + (-3/4)·n - 4 | 0.0469/n | read A[i3, i4] (i0=0, i4=0) |
| n^3 | 0.0143 | ramp | 20  →  (1/8)·n^2 + (-7/8)·n + 2 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0469/n | read A[i3, i1] (i0=0) |
| n^3 | 0.0141 | ramp | 37  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/16)·n^2 + (-7/8)·n - 2 | 0.0469/n | read A[i3, i1] (i0=0) |
| n^3 | 0.0141 | ramp | 78  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/16)·n^2 + (-7/8)·n - 15 | 0.0469/n | read A[i3, i4] (i0=0, i4=7) |
| n^3 | 0.0141 | ramp | 77  →  (1/8)·n^2 + (-7/8)·n - 3 | (1/16)·n^2 + (-7/8)·n - 15 | 0.0469/n | read A[i3, i4] (i0=0) |
| n^3 | 0.0123 | ramp | 54  →  (1/8)·n^2 | (7/128)·n^2 + (-21/16)·n + 7 | 0.041/n | read A[i1, i2] (i0=0) |
| n^3 | 0.0101 | ramp | 105  →  (1/8)·n^2 - 3 | (3/64)·n^2 + (-15/8)·n + 18 | 0.0352/n | read A[i3, i4] (i0=0) |
| n^3 | 0.00167 | ramp | 101  →  (1/8)·n^2 + (-3/4)·n - 3 | (1/128)·n^2 + (-5/16)·n + 3 | 0.00586/n | read A[i3, i4] (i0=0) |
| n^3 | 0.00167 | ramp | 100  →  (1/8)·n^2 + (-7/8)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.00586/n | read A[i1, i2] (i0=0) |
| n^3 | 0.0016 | ramp | 164  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i3, i4] (i0=0, i3=0) |
| n^3 | 0.00157 | ramp | 130  →  (1/8)·n^2 + (-15/8)·n + 5 | (1/128)·n^2 + (-7/16)·n + 6 | 0.00586/n | read A[i3, i4] (i0=0) |
| n^2.5 | 0.179 | ramp | 8  →  (1/4)·n + 2 | (7/16)·n^2 + (-7/8)·n - 98 | 0.328/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.179 | ramp | 8  →  (1/4)·n + 2 | (7/16)·n^2 + (-7/8)·n - 98 | 0.328/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.178 | ramp | 5  →  (1/4)·n + 1 | (7/16)·n^2 + (-7/8)·n - 21 | 0.328/n | read A[i1, i1] (i0=0) |
| n^2.5 | 0.0251 | ramp | 8  →  (1/4)·n + 2 | (1/16)·n^2 + (-5/8)·n - 6 | 0.0469/n | read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.0251 | ramp | 8  →  (1/4)·n + 2 | (1/16)·n^2 + (-5/8)·n - 6 | 0.0469/n | read A[i1, i4] (i0=0) |
| n^2.5 | 0.0248 | ramp | 5  →  (1/4)·n + 1 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0469/n | read A[i1, i1] (i0=0) |
| n^2.5 | 0.0173 | ramp | 7  →  (1/4)·n - 1 | (7/128)·n^2 + (-35/16)·n + 21 | 0.041/n | read A[i1, i4] (i0=0, i3=0) |
| n^2.5 | 0.00247 | ramp | 7  →  (1/4)·n - 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.00586/n | read A[i1, i4] (i0=0, i3=0) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 - 7·n - 56 | 1.31/n | read A[i1, i4] (i0=0) |
| n^2 | 2.27 | level | 3 | (21/16)·n^2 + (-49/8)·n - 35 | 0.984/n | read A[i3, i1] (i0=0) |
| n^2 | 2.27 | level | 3 | (21/16)·n^2 + (-49/8)·n - 35 | 0.984/n | write A[i3, i4] (i0=0) |
| n^2 | 1.88 | level | 1 | (15/8)·n^2 + (-15/4)·n - 20 | 1.41/n | read A[i3, i4] (i0=0) |
| n^2 | 1.86 | level | 2 | (21/16)·n^2 + (7/2)·n | 0.984/n | read A[i3, i1] (i0=0) |
| n^2 | 1.86 | level | 2 | (21/16)·n^2 + (7/2)·n | 0.984/n | read A[i1, i4] (i0=0) |
| n^2 | 1.31 | level | 1 | (21/16)·n^2 + (7/8)·n | 0.984/n | write A[i3, i4] (i0=0) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.656/n | read A[i1, i4] (i0=0, i4=0); read A[i3, i1] (i0=0, i4=0) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 - 28 | 0.328/n | read A[i3, i1] (i0=0, i4=8) |
| n^2 | 0.758 | level | 3 | (7/16)·n^2 + (-7/8)·n - 21 | 0.328/n | read A[i3, i1] (i0=0, i4=7) |
| n^2 | 0.707 | level | 2 | (1/2)·n^2 + (-1/2)·n | 0.375/n | write A[i3, i1] (i0=0) |
| n^2 | 0.65 | level | 3 | (3/8)·n^2 + (-3/8)·n - 21 | 0.281/n | read A[i3, i1] (i0=0) |
| n^2 | 0.5 | level | 1 | (1/2)·n^2 + (1/2)·n | 0.375/n | write A[i1, i2] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 | 0.328/n | read A[i1, i2] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 | 0.328/n | read A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.375 | level | 1 | (3/8)·n^2 + (-9/8)·n - 15 | 0.281/n | read A[i3, i4] (i0=0, i4=6) |
| n^2 | 0.313 | ramp | 20  →  (1/8)·n^2 + (-1/8)·n + 2 | (7/4)·n - 14 | 1.31·n^-2 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^2 | 0.312 | level | 1 | (5/16)·n^2 + (15/8)·n | 0.234/n | read A[i3, i4] (i0=0) |
| n^2 | 0.312 | level | 1 | (5/16)·n^2 + (15/8)·n | 0.234/n | write A[i3, i4] (i0=0) |
| n^2 | 0.157 | ramp | 20  →  (1/8)·n^2 | (7/8)·n - 7 | 0.656·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.134 | ramp | 54  →  (1/8)·n^2 + (-1/8)·n | (3/4)·n - 12 | 0.562·n^-2 | read A[i3, i4] (i0=0, i4=7); read A[i3, i4] (i0=0) |
| n^2 | 0.134 | ramp | 20  →  (1/8)·n^2 + (-1/4)·n + 2 | (3/4)·n - 6 | 0.562·n^-2 | read A[i3, i1] (i0=0) |
| n^2 | 0.125 | level | 4 | (1/16)·n^2 + (-1/2)·n - 8 | 0.0469/n | read A[i3, i1] (i0=0, i4=8) |
| n^2 | 0.112 | ramp | 55  →  (1/8)·n^2 + (-1/8)·n - 2 | (5/8)·n - 10 | 0.469·n^-2 | read A[i3, i4] (i0=0) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (3/8)·n - 7 | 0.0469/n | read A[i3, i1] (i0=0, i4=7) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-1/2)·n | 0.0469/n | read A[i3, i1] (i0=0, i4=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (3/8)·n | 0.0469/n | read A[i3, i4] (i0=0, i4=6) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (3/8)·n | 0.0469/n | write A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (3/8)·n | 0.0469/n | write A[i3, i4] (i0=0, i4=6) |
| n^2 | 0.0226 | ramp | 70  →  (1/8)·n^2 - 2 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i4=6) |
| n^2 | 0.0222 | ramp | 54  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^2 | 0.0222 | ramp | 52  →  (1/8)·n^2 + (-3/4)·n - 2 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i4=0) |
| n^2 | 0.0221 | ramp | 51  →  (1/8)·n^2 + (-7/8)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i4] (i0=0) |
| n^2 | 0.0221 | ramp | 51  →  (1/8)·n^2 + (-7/8)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0221 | ramp | 50  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i1] (i0=0, i3=0) |
| n^2 | 0.0221 | ramp | 50  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.0221 | ramp | 18  →  (1/8)·n^2 - n + 2 | (1/8)·n - 1 | 0.0938·n^-2 | read A[i3, i1] (i0=0) |
| n^2 | 0.022 | ramp | 99  →  (1/8)·n^2 + (-7/8)·n - 1 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i3=0, i4=7) |
| n^2 | 0.022 | ramp | 98  →  (1/8)·n^2 + (-7/8)·n - 2 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i3=0) |
| n^2 | 0.0216 | ramp | 34  →  (1/8)·n^2 + (-15/8)·n + 7 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i3, i1] (i0=0) |
| n^2 | 0.0215 | ramp | 74  →  (1/8)·n^2 + (-15/8)·n + 6 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0, i4=7) |
| n^2 | 0.0214 | ramp | 72  →  (1/8)·n^2 - 2·n + 8 | (1/8)·n - 3 | 0.0938·n^-2 | read A[i3, i4] (i0=0) |
| n^1.5 | 0.293 | ramp | 6  →  (1/4)·n | (7/8)·n - 14 | 0.656·n^-2 | read A[i1, i4] (i0=0, i3=0) |
| n^1.5 | 0.247 | ramp | 2  →  (1/8)·n + 1 | n - 1 | 0.75·n^-2 | read A[i1, i1] (i0=0, i3=0) |
| n^1.5 | 0.218 | ramp | 5  →  (1/8)·n + 2 | (7/8)·n - 14 | 0.656·n^-2 | read A[i1, i4] (i0=0, i3=0) |
| n^1.5 | 0.0418 | ramp | 6  →  (1/4)·n | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i3=0) |
| n^1.5 | 0.0312 | ramp | 5  →  (1/8)·n + 2 | (1/8)·n - 2 | 0.0938·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 189 | level | 6 | 77 | 57.8·n^-3 | read A[i1, i4] (i0=0) |
| n^0 | 73.5 | level | 32 | 13 | 9.75·n^-3 | read A[i3, i4] (i0=0, i3=0, i4=6); read A[i3, i4] (i0=0, i4=6) |
| n^0 | 65.7 | level | 30 | 12 | 9·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 64.7 | ramp | 36  →  48 | 10 | 7.5·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 58.2 | level | 28 | 11 | 8.25·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 51.4 | level | 6 | 21 | 15.8·n^-3 | read A[i1, i4] (i0=0) |
| n^0 | 51 | level | 26 | 10 | 7.5·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 44.1 | level | 24 | 9 | 6.75·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 37.5 | level | 22 | 8 | 6·n^-3 | read A[i3, i4] (i0=0, i3=0); read A[i3, i4] (i0=0) |
| n^0 | 36.4 | level | 3 | 21 | 15.8·n^-3 | read A[i1, i1] (i0=0) |
| n^0 | 26.8 | level | 20 | 6 | 4.5·n^-3 | read A[i3, i4] (i0=0, i4=0) |
| n^0 | 21.2 | level | 18 | 5 | 3.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 17.1 | level | 6 | 7 | 5.25·n^-3 | read A[i1, i4] (i0=0, i4=0) |
| n^0 | 14 | level | 4 | 7 | 5.25·n^-3 | read A[i1, i4] (i0=0, i3=0) |
| n^0 | 11.3 | level | 8 | 4 | 3·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 8.49 | level | 18 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 7.94 | level | 7 | 3 | 2.25·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 7 | level | 49 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i3=0, i4=7) |
| n^0 | 6.56 | level | 43 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 6.48 | level | 42 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 6.4 | level | 41 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=7) |
| n^0 | 5.66 | level | 32 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=7); read A[i3, i4] (i0=0) |
| n^0 | 5.66 | level | 8 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 5.48 | level | 30 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=6) |
| n^0 | 5.29 | level | 28 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 5.29 | level | 7 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 5.1 | level | 26 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.9 | level | 6 | 2 | 1.5·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 4.9 | level | 24 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.9 | level | 6 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 4.69 | level | 22 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.47 | level | 20 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0) |
| n^0 | 4.47 | level | 20 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^0 | 4.47 | level | 5 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 4.24 | level | 18 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i4=0) |
| n^0 | 4.12 | level | 17 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^0 | 4 | level | 16 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=6) |
| n^0 | 4 | level | 4 | 2 | 1.5·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 4 | level | 16 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=0) |
| n^0 | 3.87 | level | 15 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.74 | level | 14 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.61 | level | 13 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.46 | level | 12 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 3.32 | level | 11 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.83 | level | 8 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=6) |
| n^0 | 2.65 | level | 7 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.45 | level | 6 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.45 | level | 6 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i4=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.75·n^-3 | read A[i1, i1] (i0=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i3, i4] (i0=0, i3=0, i4=0) |
| n^0 | 2 | level | 4 | 1 | 0.75·n^-3 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0); read A[i3, i1] (i0=0, i3=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=0) |
| n^0 | 1.41 | level | 2 | 1 | 0.75·n^-3 | read A[i3, i1] (i0=0, i3=0) |
| n^0 | 1 | level | 1 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^0 | 1 | level | 1 | 1 | 0.75·n^-3 | read A[i1, i2] (i0=0, i2=0) |

Same structure as lu in a different schedule: submatrix re-reads at up to (1/8)n^2 lines, 0.0076·n^4 leading, headroom +1.0.

## mvt — infinite-repeat  [`exact`]

Accesses $A(n) = 8·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0721·n^3  +  1.12·n^2.5  +  11.5·n^2  +  5.91·n^1.5  +  9.4·n^1  +  2.35·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0316 | ramp | (5/2)·n - 1  →  (1/8)·n^2 + (1/2)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0137 | read B[i4, i3] (i0=0) |
| n^3 | 0.0316 | ramp | (5/2)·n - 1  →  (1/8)·n^2 + (1/2)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0137 | read B[i1, i2] (i0=0) |
| n^3 | 0.00442 | ramp | (27/8)·n - 14  →  (1/8)·n^2 + (1/2)·n - 8 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read B[i1, i2] (i0=0) |
| n^3 | 0.00442 | ramp | (27/8)·n - 14  →  (1/8)·n^2 + (1/2)·n - 8 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read B[i4, i3] (i0=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 + (-7/4)·n | 0.109 | read B[i4, i3] (i0=0) |
| n^2.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read E[i4] (i0=0) |
| n^2.5 | 0.0547 | level | (1/4)·n + 1 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read C[i2] (i0=0) |
| n^2.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read E[i4] (i0=0) |
| n^2.5 | 0.00781 | level | (1/4)·n + 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read C[i2] (i0=0) |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 + (-7/4)·n | 0.328 | read B[i1, i2] (i0=0); read C[i2] (i0=0) (+1) |
| n^2 | 3.03 | level | 3 | (7/4)·n^2 | 0.219 | write A[i1] (i0=0); write D[i3] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 | 0.219 | read A[i1] (i0=0); read D[i3] (i0=0, i4=0) (+1) |
| n^2 | 0.433 | level | 3 | (1/4)·n^2 | 0.0312 | write A[i1] (i0=0); write D[i3] (i0=0, i4=0) (+1) |
| n^2 | 0.31 | ramp | (1/8)·n^2 + (3/8)·n + 1  →  (1/8)·n^2 + (1/2)·n | (7/8)·n - 1 | 0.109/n | read B[i4, i3] (i0=0) |
| n^2 | 0.31 | ramp | (1/8)·n^2 + (3/8)·n + 1  →  (1/8)·n^2 + (1/2)·n | (7/8)·n - 1 | 0.109/n | read B[i1, i2] (i0=0, i2=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (-1/4)·n | 0.0312 | read A[i1] (i0=0); read D[i3] (i0=0) |
| n^2 | 0.212 | ramp | (3/2)·n  →  (1/8)·n^2 + (1/8)·n + 3 | (7/8)·n - 1 | 0.109/n | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.183 | ramp | (3/2)·n  →  (1/8)·n^2 + (1/8)·n + 3 | (3/4)·n | 0.0938/n | read B[i1, i2] (i0=0) |
| n^2 | 0.0884 | level | (1/8)·n^2 + (1/2)·n | (1/4)·n - 4 | 0.0312/n | read A[i1] (i0=0); read D[i3] (i0=0, i4=0) |
| n^2 | 0.0884 | level | (1/8)·n^2 + (3/8)·n + 2 | (1/4)·n - 4 | 0.0312/n | read C[i2] (i0=0); read E[i4] (i0=0, i3=0) |
| n^2 | 0.0595 | ramp | (9/4)·n - 6  →  (1/8)·n^2 + (-5/8)·n + 9 | (1/4)·n - 2 | 0.0312/n | read B[i1, i2] (i0=0) |
| n^2 | 0.0433 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n | (1/8)·n - 1 | 0.0156/n | read B[i4, i3] (i0=0) |
| n^2 | 0.0433 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n | (1/8)·n - 1 | 0.0156/n | read B[i1, i2] (i0=0, i2=0) |
| n^2 | 0.0422 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n - 1 | (1/8)·n - 2 | 0.0156/n | read B[i4, i3] (i0=0, i4=0) |
| n^2 | 0.0422 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n - 1 | (1/8)·n - 2 | 0.0156/n | read B[i1, i2] (i0=0) |
| n^2 | 0.0298 | ramp | (19/8)·n - 7  →  (1/8)·n^2 + (-5/8)·n + 9 | (1/8)·n - 1 | 0.0156/n | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.029 | ramp | (19/8)·n  →  (1/8)·n^2 + (-3/4)·n + 3 | (1/8)·n - 2 | 0.0156/n | read B[i4, i3] (i0=0) |
| n^2 | 0.029 | ramp | (19/8)·n - 1  →  (1/8)·n^2 + (-3/4)·n + 2 | (1/8)·n - 2 | 0.0156/n | read B[i1, i2] (i0=0, i1=0) |
| n^1.5 | 0.928 | level | (9/8)·n + (15/8) | (7/8)·n + (-7/8) | 0.109/n | read B[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.109/n | read B[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.109/n | read B[i4, i3] (i0=0, i4=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.109/n | read E[i4] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.109/n | read E[i4] (i0=0, i4=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n | 0.109/n | read C[i2] (i0=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n | 0.109/n | read C[i2] (i0=0, i2=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.0156/n | read E[i4] (i0=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.0156/n | read E[i4] (i0=0, i4=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.0156/n | read C[i2] (i0=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.0156/n | read C[i2] (i0=0, i2=0) |
| n^1 | 3.03 | level | 3 | (7/4)·n | 0.219/n | read C[i2] (i0=0); read E[i4] (i0=0, i3=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (9/8)·n + (15/4) | 2 | 0.25·n^-2 | read B[i1, i2] (i0=0); read B[i4, i3] (i0=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (1/4)·n + 1 | 2 | 0.25·n^-2 | read B[i1, i2] (i0=0); read B[i4, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (1/2)·n | 2 | 0.25·n^-2 | read A[i1] (i0=0); read D[i3] (i0=0, i4=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (3/8)·n + 2 | 2 | 0.25·n^-2 | read C[i2] (i0=0); read E[i4] (i0=0, i3=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + (1/2)·n | 2 | 0.25·n^-2 | read A[i1] (i0=0); read D[i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | 1 | 0.125·n^-2 | read B[i1, i2] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (11/8)·n + (7/2) | 1 | 0.125·n^-2 | read B[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.125·n^-2 | read B[i1, i2] (i0=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/4)·n + (37/8) | 1 | 0.125·n^-2 | read E[i4] (i0=0, i3=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 2 | 1 | 0.125·n^-2 | read C[i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/8)·n + 2 | 1 | 0.125·n^-2 | read E[i4] (i0=0, i3=0, i4=0) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.125·n^-2 | read B[i1, i2] (i0=0, i1=0) |
| n^0.5 | 1.17 | level | (11/8)·n + 1 | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0, i3=0) |

Unlike atax, mvt multiplies by A and A^T in one pass, so the matrix is re-touched *within* a single invocation: the transposed walk `read B[i4,i3]` forms ramp families from (5/2)n up to (1/8)n^2 + (1/2)n lines with population n^2/8 — d = 3.0, headroom +1.0 already single-shot (the earlier +0.5 reading came from the corrupted iterator terms). The n^2.5 levels below it are the row/vector reuses at (9/8)n and (1/4)n lines. Repetition changes only constants (0.036 → 0.072: the second pass doubles the far reuses).

## mvt — single-shot  [`exact`]

Accesses $A(n) = 8·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.036·n^3  +  1.12·n^2.5  +  10.7·n^2  +  5.91·n^1.5  +  3.16·n^1  +  1.17·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0316 | ramp | (5/2)·n - 1  →  (1/8)·n^2 + (1/2)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0137 | read B[i4, i3] (i0=0) |
| n^3 | 0.00442 | ramp | (27/8)·n - 14  →  (1/8)·n^2 + (1/2)·n - 8 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read B[i4, i3] (i0=0) |
| n^2.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n^2 + (-7/4)·n | 0.109 | read B[i4, i3] (i0=0) |
| n^2.5 | 0.116 | level | (9/8)·n - 6 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read E[i4] (i0=0) |
| n^2.5 | 0.0547 | level | (1/4)·n + 1 | (7/64)·n^2 + (-7/4)·n | 0.0137 | read C[i2] (i0=0) |
| n^2.5 | 0.0166 | level | (9/8)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read E[i4] (i0=0) |
| n^2.5 | 0.00781 | level | (1/4)·n + 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00195 | read C[i2] (i0=0) |
| n^2 | 4.55 | level | 3 | (21/8)·n^2 | 0.328 | read B[i1, i2] (i0=0); read C[i2] (i0=0) (+1) |
| n^2 | 3.46 | level | 3 | 2·n^2 | 0.25 | read C[i2] (i0=0); write A[i1] (i0=0) (+2) |
| n^2 | 2 | level | 1 | 2·n^2 - 2·n | 0.25 | read A[i1] (i0=0); read D[i3] (i0=0) |
| n^2 | 0.31 | ramp | (1/8)·n^2 + (3/8)·n + 1  →  (1/8)·n^2 + (1/2)·n | (7/8)·n - 1 | 0.109/n | read B[i4, i3] (i0=0) |
| n^2 | 0.212 | ramp | (3/2)·n  →  (1/8)·n^2 + (1/8)·n + 3 | (7/8)·n - 1 | 0.109/n | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0433 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n | (1/8)·n - 1 | 0.0156/n | read B[i4, i3] (i0=0) |
| n^2 | 0.0422 | ramp | (1/8)·n^2 + (3/8)·n + 2  →  (1/8)·n^2 + (1/2)·n - 1 | (1/8)·n - 2 | 0.0156/n | read B[i4, i3] (i0=0, i4=0) |
| n^2 | 0.0298 | ramp | (19/8)·n - 7  →  (1/8)·n^2 + (-5/8)·n + 9 | (1/8)·n - 1 | 0.0156/n | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.029 | ramp | (19/8)·n  →  (1/8)·n^2 + (-3/4)·n + 3 | (1/8)·n - 2 | 0.0156/n | read B[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + (15/8) | (7/8)·n + (-7/8) | 0.109/n | read B[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.109/n | read B[i4, i3] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n + 1 | (7/8)·n | 0.109/n | read B[i4, i3] (i0=0, i4=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.109/n | read E[i4] (i0=0) |
| n^1.5 | 0.928 | level | (9/8)·n - 6 | (7/8)·n | 0.109/n | read E[i4] (i0=0, i4=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n | 0.109/n | read C[i2] (i0=0) |
| n^1.5 | 0.438 | level | (1/4)·n + 1 | (7/8)·n | 0.109/n | read C[i2] (i0=0, i2=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.0156/n | read E[i4] (i0=0) |
| n^1.5 | 0.133 | level | (9/8)·n - 5 | (1/8)·n - 1 | 0.0156/n | read E[i4] (i0=0, i4=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.0156/n | read C[i2] (i0=0) |
| n^1.5 | 0.0625 | level | (1/4)·n + 2 | (1/8)·n - 1 | 0.0156/n | read C[i2] (i0=0, i2=0) |
| n^1 | 1.75 | level | 1 | (7/4)·n | 0.219/n | read A[i1] (i0=0); read D[i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (9/8)·n + (15/4) | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 2 | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/4)·n + 1 | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0, i3=0, i4=0) |
| n^0.5 | 1.17 | level | (11/8)·n + 1 | 1 | 0.125·n^-2 | read B[i4, i3] (i0=0, i3=0) |

Unlike atax, mvt multiplies by A and A^T in one pass, so the matrix is re-touched *within* a single invocation: the transposed walk `read B[i4,i3]` forms ramp families from (5/2)n up to (1/8)n^2 + (1/2)n lines with population n^2/8 — d = 3.0, headroom +1.0 already single-shot (the earlier +0.5 reading came from the corrupted iterator terms). The n^2.5 levels below it are the row/vector reuses at (9/8)n and (1/4)n lines. Repetition changes only constants (0.036 → 0.072: the second pass doubles the far reuses).

## seidel-2d — infinite-repeat  [`exact`]

Accesses $A(n) = 10·n^3 - 40·n^2 + 40·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0884·n^4  +  0.306·n^3.5  +  15.4·n^3  +  6.43·n^2.5  +  25·n^2  +  4.29·n^1.5  +  14.3·n^1

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n^3 + (-23/8)·n^2 + (107/8)·n + (-85/8) | 0.0125 | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0) |
| n^4 | 0.0442 | level | (1/8)·n^2 | (1/8)·n^3 + (-11/4)·n^2 + (101/8)·n - 10 | 0.0125 | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0) |
| n^3.5 | 0.153 | level | (3/8)·n + (13/8) | (1/4)·n^3 + (-21/4)·n^2 + (71/4)·n + (-51/4) | 0.025 | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^3.5 | 0.153 | level | (3/8)·n - 1 | (1/4)·n^3 - 5·n^2 + (67/4)·n - 12 | 0.025 | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^3 | 2.38 | level | 3 | (11/8)·n^3 + (-15/4)·n^2 + 2·n | 0.138 | read A[i2 + 1, i3 - 1] (i0=0, i1=0); read A[i2 - 1, i3 - 1] (i0=0) (+1) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (-33/4)·n^2 + (25/2)·n - 6 | 0.175 | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^3 | 1.5 | level | 1 | (3/2)·n^3 + (-17/4)·n^2 + (1/2)·n + 4 | 0.15 | read A[i2 - 1, i3] (i0=0, i1=0); read A[i2, i3] (i0=0, i1=0) (+3) |
| n^3 | 1.22 | level | 6 | (1/2)·n^3 + (-39/8)·n^2 + (27/4)·n + 2 | 0.05 | read A[i2 - 1, i3 - 1] (i0=0, i1=0); read A[i2 - 1, i3] (i0=0, i1=0) (+7) |
| n^3 | 0.884 | level | 2 | (5/8)·n^3 + (-5/4)·n^2 | 0.0625 | read A[i2, i3 - 1] (i0=0, i1=0); read A[i2, i3 - 1] (i0=0) |
| n^3 | 0.884 | level | 2 | (5/8)·n^3 + (-5/4)·n^2 | 0.0625 | write A[i2, i3] (i0=0, i1=0); write A[i2, i3] (i0=0) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 + (-37/8)·n^2 + (31/4)·n - 4 | 0.0875 | read A[i2 + 1, i3 + 1] (i0=0, i2=5); read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 + (-29/8)·n^2 + (19/4)·n - 2 | 0.0875 | read A[i2, i3] (i0=0) |
| n^3 | 0.559 | level | 5 | (1/4)·n^3 + (-5/2)·n^2 + 4·n | 0.025 | read A[i2 + 1, i3 - 1] (i0=0, i1=0); read A[i2 - 1, i3 - 1] (i0=0) (+1) |
| n^3 | 0.433 | level | 3 | (1/4)·n^3 + (-5/2)·n^2 + 4·n | 0.025 | read A[i2, i3 - 1] (i0=0, i1=0); read A[i2, i3 - 1] (i0=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n^2 - 2·n | 0.1/n | read A[i2 - 1, i3 - 1] (i0=0, i1=0, i2=0, i3=0); read A[i2 + 1, i3 - 1] (i0=0, i1=0, i2=1, i3=0) (+5) |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 2·n | 0.1/n | read A[i2 - 1, i3 - 1] (i0=0, i1=0, i2=0, i3=0); read A[i2 + 1, i3 - 1] (i0=0, i1=0, i3=0) (+3) |
| n^3 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n^2 - 5·n + 4 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0, i2=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 5·n + 4 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0, i2=0) (+1) |
| n^3 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n^2 - 6·n + 5 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0, i2=1) (+1) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/4)·n | n^2 - 6·n + 5 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.306 | level | 6 | (1/8)·n^3 + (-11/8)·n^2 + (13/4)·n - 2 | 0.0125 | read A[i2, i3] (i0=0) |
| n^3 | 0.25 | level | 1 | (1/4)·n^3 + (-19/8)·n^2 + (11/4)·n + 2 | 0.025 | read A[i2 - 1, i3] (i0=0, i1=0); read A[i2, i3] (i0=0, i1=0) (+3) |
| n^3 | 0.25 | level | 4 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | write A[i2, i3] (i0=0, i1=0); write A[i2, i3] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | read A[i2 + 1, i3 - 1] (i0=0, i1=0); read A[i2 + 1, i3 - 1] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | write A[i2, i3] (i0=0, i1=0); write A[i2, i3] (i0=0) |
| n^3 | 0.177 | level | 2 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | read A[i2, i3 - 1] (i0=0, i1=0); read A[i2, i3 - 1] (i0=0) |
| n^3 | 0.177 | level | 2 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | write A[i2, i3] (i0=0, i1=0); write A[i2, i3] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n^2 + (-9/4)·n + (17/8) | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n^2 + (-21/8)·n + (17/2) | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i1=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 | (1/8)·n^2 + (-5/2)·n + 8 | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i1=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^3 | 0.0418 | ramp | (1/8)·n^2 + (-1/8)·n + 2  →  (1/8)·n^2 - 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.0418 | ramp | (1/8)·n^2 + (-1/8)·n + 2  →  (1/8)·n^2 - 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0417 | ramp | (1/8)·n^2 + (-1/4)·n + 4  →  (1/8)·n^2 - 2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 + 1, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0417 | ramp | (1/8)·n^2 + (-1/4)·n + 4  →  (1/8)·n^2 - 2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2.5 | 1.22 | level | (3/8)·n + (13/8) | 2·n^2 - 6·n | 0.2/n | read A[i2 - 1, i3 - 1] (i0=0, i1=0, i3=0); read A[i2, i3 - 1] (i0=0, i1=0, i3=0) (+2) |
| n^2.5 | 1.22 | level | (3/8)·n - 1 | 2·n^2 - 6·n | 0.2/n | read A[i2 - 1, i3 - 1] (i0=0, i1=0, i3=0); read A[i2, i3 - 1] (i0=0, i1=0, i3=0) (+2) |
| n^2.5 | 1.22 | level | (3/8)·n + (1/4) | 2·n^2 - 8·n + 6 | 0.2/n | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n - 1 | n^2 - 4·n + 3 | 0.1/n | read A[i2, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n - 2 | n^2 - 4·n + 3 | 0.1/n | read A[i2 - 1, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n + (-3/8) | n^2 - 4·n + 3 | 0.1/n | read A[i2, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n + (13/8) | n^2 - 4·n + 3 | 0.1/n | read A[i2 - 1, i3 + 1] (i0=0) |
| n^2.5 | 0.0765 | level | (3/8)·n + (13/8) | (1/8)·n^2 + (-5/2)·n + (51/8) | 0.0125/n | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^2.5 | 0.0765 | level | (3/8)·n - 1 | (1/8)·n^2 + (-19/8)·n + 6 | 0.0125/n | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^2.5 | 0.0765 | level | (3/8)·n + (13/8) | (1/8)·n^2 + (-5/2)·n + (51/8) | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i1=0) |
| n^2.5 | 0.0765 | level | (3/8)·n - 1 | (1/8)·n^2 + (-19/8)·n + 6 | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i1=0) |
| n^2 | 7 | level | 1 | 7·n^2 - 21·n + 14 | 0.7/n | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 4 | level | 1 | 4·n^2 - 5·n - 6 | 0.4/n | read A[i2 - 1, i3 + 1] (i0=0, i1=0, i2=0); read A[i2 - 1, i3 - 1] (i0=0, i1=0, i3=0) (+14) |
| n^2 | 1.41 | level | 2 | n^2 - 2·n | 0.1/n | read A[i2, i3 - 1] (i0=0, i1=0, i2=0, i3=0); read A[i2, i3 - 1] (i0=0, i1=0, i3=0) (+5) |
| n^2 | 1.25 | level | 1 | (5/4)·n^2 + (-5/2)·n | 0.125/n | read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=5); read A[i2 - 1, i3 + 1] (i0=0, i1=0) (+1) |
| n^2 | 0.707 | level | (1/8)·n^2 + (7/8)·n | 2·n - 2 | 0.2·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0); read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.707 | level | (1/8)·n^2 | 2·n - 2 | 0.2·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0); read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.625 | level | 1 | (5/8)·n^2 + (-5/4)·n | 0.0625/n | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (1/4) | n | 0.1·n^-2 | read A[i2 + 1, i3 - 1] (i0=0, i1=0, i2=0, i3=0); read A[i2 + 1, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 2 | n | 0.1·n^-2 | read A[i2 + 1, i3 - 1] (i0=0, i1=0, i2=0, i3=0); read A[i2 + 1, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (9/4) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/2)·n + (5/2) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 2 | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (1/8) | n | 0.1·n^-2 | read A[i2, i3 - 1] (i0=0, i1=0, i2=0, i3=0); read A[i2, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 1 | n | 0.1·n^-2 | read A[i2, i3 - 1] (i0=0, i1=0, i2=0, i3=0); read A[i2, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (5/4) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (-15/8) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n - 3 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=0) (+2) |
| n^2 | 0.354 | level | (1/8)·n^2 | n - 3 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=0) (+1) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (9/4) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n - 2 | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/2)·n + (5/2) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n - 2 | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=1); read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n - 1 | n - 1 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n - 1 | n - 1 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n - 1 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n | n - 1 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n - 4 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=1) (+1) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n | n - 4 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (-5/2)·n + 4 | 0.025/n | read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=5); read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (-19/4)·n + (17/2) | 0.025/n | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (-5/2)·n + 4 | 0.025/n | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.25 | level | 1 | (1/4)·n^2 + (-5/2)·n + 4 | 0.025/n | read A[i2 - 1, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.0419 | ramp | (1/8)·n^2 + (-1/8)·n + 2  →  (1/8)·n^2 - 1 | (1/8)·n - 2 | 0.0125·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.0419 | ramp | (1/8)·n^2 + (-1/8)·n + 2  →  (1/8)·n^2 - 1 | (1/8)·n - 2 | 0.0125·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0) |
| n^2 | 0.0419 | ramp | (1/8)·n^2 + (-1/4)·n + 4  →  (1/8)·n^2 - 2 | (1/8)·n - 2 | 0.0125·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^2 | 0.0419 | ramp | (1/8)·n^2 + (-1/4)·n + 4  →  (1/8)·n^2 - 2 | (1/8)·n - 2 | 0.0125·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=0) |
| n^1.5 | 0.612 | level | (3/8)·n + (1/4) | n - 3 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^1.5 | 0.612 | level | (3/8)·n + (1/4) | n - 3 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0) |
| n^1.5 | 0.612 | level | (3/8)·n - 1 | n - 3 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^1.5 | 0.612 | level | (3/8)·n + (-19/8) | n - 3 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0) |
| n^1.5 | 0.612 | level | (3/8)·n - 2 | n - 3 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0) |
| n^1.5 | 0.612 | level | (3/8)·n + (13/8) | n - 3 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i1=0) |
| n^1.5 | 0.612 | level | (3/8)·n + (-3/8) | n - 3 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^1 | 5 | level | 1 | 5·n - 10 | 0.5·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 1 | level | 1 | n - 2 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 1 | level | 1 | n - 2 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 1 | level | 1 | n - 2 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^1 | 1 | level | 1 | n - 2 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (9/4) | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n + (5/2) | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 2 | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (5/4) | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (-15/8) | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (-1/8)·n | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n | 1 | 0.1·n^-3 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 | 1 | 0.1·n^-3 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (9/4) | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n - 2 | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/2)·n + (5/2) | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n - 2 | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | 1 | 0.1·n^-3 | read A[i2 + 1, i3 + 1] (i0=0, i1=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (7/8)·n - 1 | 1 | 0.1·n^-3 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (3/4)·n - 1 | 1 | 0.1·n^-3 | read A[i2, i3 + 1] (i0=0, i1=0, i2=0) |

In-place 2-D sweep: the south-east neighbor `read A[i2+1,i3+1]` is the value written a full sweep earlier — distance (1/8)n^2 (+ (7/8)n at the row seam), one array's footprint, 0.0884·n^4 total. The three-row working window shows up as n^3.5 terms at (3/8)n lines. Headroom +1.0 with a single-array boundary, half of jacobi-2d's.

## seidel-2d — single-shot  [`exact`]

Accesses $A(n) = 10·n^3 - 40·n^2 + 40·n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0884·n^4  +  0.306·n^3.5  +  15.3·n^3  +  6.12·n^2.5  +  17.1·n^2

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n^3 + (-23/8)·n^2 + (107/8)·n + (-85/8) | 0.0125 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^4 | 0.0442 | level | (1/8)·n^2 | (1/8)·n^3 + (-11/4)·n^2 + (101/8)·n - 10 | 0.0125 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^3.5 | 0.153 | level | (3/8)·n + (13/8) | (1/4)·n^3 - 5·n^2 + (51/4)·n | 0.025 | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^3.5 | 0.153 | level | (3/8)·n - 1 | (1/4)·n^3 + (-19/4)·n^2 + 12·n | 0.025 | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^3 | 2.38 | level | 3 | (11/8)·n^3 + (-15/4)·n^2 + 2·n | 0.138 | read A[i2 - 1, i3 - 1] (i0=0); read A[i2 + 1, i3 - 1] (i0=0) |
| n^3 | 1.75 | level | 1 | (7/4)·n^3 + (-11/2)·n^2 + 4·n | 0.175 | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^3 | 1.5 | level | 1 | (3/2)·n^3 - 5·n^2 + 4·n | 0.15 | read A[i2 - 1, i3] (i0=0); read A[i2 + 1, i3] (i0=0) |
| n^3 | 1.22 | level | 6 | (1/2)·n^3 - 5·n^2 + 8·n | 0.05 | read A[i2 + 1, i3 - 1] (i0=0, i2=0); read A[i2 - 1, i3 - 1] (i0=0) (+3) |
| n^3 | 1.06 | level | 2 | (3/4)·n^3 + (-5/2)·n^2 + 2·n | 0.075 | read A[i2, i3 - 1] (i0=0, i2=0, i3=0); read A[i2, i3 - 1] (i0=0, i3=0) (+1) |
| n^3 | 0.884 | level | 2 | (5/8)·n^3 + (-5/4)·n^2 | 0.0625 | write A[i2, i3] (i0=0) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 + (-11/4)·n^2 + 2·n | 0.0875 | read A[i2 + 1, i3 + 1] (i0=0, i2=5); read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.875 | level | 1 | (7/8)·n^3 + (-11/4)·n^2 + 2·n | 0.0875 | read A[i2, i3] (i0=0) |
| n^3 | 0.559 | level | 5 | (1/4)·n^3 + (-5/2)·n^2 + 4·n | 0.025 | read A[i2 - 1, i3 - 1] (i0=0); read A[i2 + 1, i3 - 1] (i0=0) |
| n^3 | 0.433 | level | 3 | (1/4)·n^3 + (-5/2)·n^2 + 4·n | 0.025 | read A[i2 - 1, i3 - 1] (i0=0, i2=0, i3=0); read A[i2 + 1, i3 - 1] (i0=0, i2=0, i3=0) (+2) |
| n^3 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n^2 - 3·n + 2 | 0.1/n | read A[i2 - 1, i3 - 1] (i0=0, i2=0, i3=0); read A[i2 + 1, i3 - 1] (i0=0, i2=1, i3=0) (+1) |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 3·n + 2 | 0.1/n | read A[i2 - 1, i3 - 1] (i0=0, i2=0, i3=0); read A[i2 + 1, i3 - 1] (i0=0, i3=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n^2 - 5·n + 4 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0, i2=1) (+1) |
| n^3 | 0.354 | level | (1/8)·n^2 | n^2 - 5·n + 4 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0, i2=0); read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n^2 - 6·n + 5 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0, i2=1); read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.354 | level | (1/8)·n^2 + (3/4)·n | n^2 - 6·n + 5 | 0.1/n | read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.306 | level | 6 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | read A[i2, i3] (i0=0) |
| n^3 | 0.25 | level | 4 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | write A[i2, i3] (i0=0) |
| n^3 | 0.25 | level | 1 | (1/4)·n^3 + (-5/2)·n^2 + 4·n | 0.025 | read A[i2 - 1, i3] (i0=0); read A[i2 + 1, i3] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | read A[i2 + 1, i3 - 1] (i0=0) |
| n^3 | 0.217 | level | 3 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0125 | write A[i2, i3] (i0=0) |
| n^3 | 0.177 | level | 2 | (1/8)·n^3 + (-1/4)·n^2 | 0.0125 | write A[i2, i3] (i0=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 + (7/8)·n | (1/8)·n^2 + (-9/4)·n + (17/8) | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0442 | level | (1/8)·n^2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0418 | ramp | (1/8)·n^2 + (-1/8)·n + 2  →  (1/8)·n^2 - 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 + 1, i3 + 1] (i0=0) |
| n^3 | 0.0418 | ramp | (1/8)·n^2 + (-1/8)·n + 2  →  (1/8)·n^2 - 1 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0417 | ramp | (1/8)·n^2 + (-1/4)·n + 4  →  (1/8)·n^2 - 2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 + 1, i3 + 1] (i0=0, i2=0) |
| n^3 | 0.0417 | ramp | (1/8)·n^2 + (-1/4)·n + 4  →  (1/8)·n^2 - 2 | (1/8)·n^2 + (-17/8)·n + 2 | 0.0125/n | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2.5 | 1.22 | level | (3/8)·n + (13/8) | 2·n^2 - 6·n | 0.2/n | read A[i2 - 1, i3 - 1] (i0=0, i3=0); read A[i2, i3 - 1] (i0=0, i3=0) |
| n^2.5 | 1.22 | level | (3/8)·n - 1 | 2·n^2 - 6·n | 0.2/n | read A[i2 - 1, i3 - 1] (i0=0, i3=0); read A[i2, i3 - 1] (i0=0, i3=0) |
| n^2.5 | 1.22 | level | (3/8)·n + (1/4) | 2·n^2 - 6·n | 0.2/n | read A[i2 - 1, i3 + 1] (i0=0); read A[i2, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n - 1 | n^2 - 3·n | 0.1/n | read A[i2, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n - 2 | n^2 - 3·n | 0.1/n | read A[i2 - 1, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n + (-3/8) | n^2 - 3·n | 0.1/n | read A[i2, i3 + 1] (i0=0) |
| n^2.5 | 0.612 | level | (3/8)·n + (13/8) | n^2 - 3·n | 0.1/n | read A[i2 - 1, i3 + 1] (i0=0) |
| n^2 | 7 | level | 1 | 7·n^2 - 14·n | 0.7/n | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 2 | level | 1 | 2·n^2 - 4·n | 0.2/n | read A[i2 - 1, i3 - 1] (i0=0, i3=0); read A[i2 - 1, i3] (i0=0, i3=0) (+1) |
| n^2 | 0.707 | level | (1/8)·n^2 + (7/8)·n | 2·n - 2 | 0.2·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0); read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.707 | level | (1/8)·n^2 | 2·n - 2 | 0.2·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0); read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (1/4) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 2 | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (9/4) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/2)·n + (5/2) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/4)·n + 2 | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (1/8) | n - 1 | 0.1·n^-2 | read A[i2, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n + 1 | n - 1 | 0.1·n^-2 | read A[i2, i3 - 1] (i0=0, i2=0, i3=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (5/4) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (-15/8) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (-1/8)·n | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (5/8)·n + (9/4) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n - 2 | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (1/2)·n + (5/2) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n - 2 | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n + (9/8) | n - 1 | 0.1·n^-2 | read A[i2 + 1, i3 + 1] (i0=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n - 1 | n - 1 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n - 1 | n - 1 | 0.1·n^-2 | read A[i2, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (7/8)·n | n - 1 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |
| n^2 | 0.354 | level | (1/8)·n^2 + (3/4)·n | n - 1 | 0.1·n^-2 | read A[i2 - 1, i3 + 1] (i0=0, i2=0) |

In-place 2-D sweep: the south-east neighbor `read A[i2+1,i3+1]` is the value written a full sweep earlier — distance (1/8)n^2 (+ (7/8)n at the row seam), one array's footprint, 0.0884·n^4 total. The three-row working window shows up as n^3.5 terms at (3/8)n lines. Headroom +1.0 with a single-array boundary, half of jacobi-2d's.

## symm — infinite-repeat  [`exact`]

Accesses $A(n) = (5/2)·n^3 + (3/2)·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0466·n^4  +  1.08·n^3.5  +  4.13·n^3  +  6.28·n^2.5  +  15.8·n^2  +  0.333·n^1.5  +  43.6·n^1  +  13.6·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0176 | ramp | n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/64)·n^3 + (-63/64)·n^2 + (31/8)·n - 2 | 0.0187 | read A[i3, i2] (i0=0) |
| n^4 | 0.0155 | ramp | n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (21/512)·n^3 + (-53/64)·n^2 + (11/4)·n | 0.0164 | read B[i3, i2] (i0=0) |
| n^4 | 0.00286 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/128)·n^3 + (-7/32)·n^2 + (7/4)·n - 4 | 0.00313 | read A[i3, i2] (i0=0) |
| n^4 | 0.00283 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/128)·n^3 + (-15/64)·n^2 + (17/8)·n - 6 | 0.00313 | read B[i3, i2] (i0=0) |
| n^4 | 0.00283 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/128)·n^3 + (-15/64)·n^2 + (17/8)·n - 6 | 0.00313 | read A[i3, i2] (i0=0) |
| n^4 | 0.00251 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (7/1024)·n^3 + (-23/128)·n^2 + (5/4)·n - 2 | 0.00273 | read B[i3, i2] (i0=0) |
| n^4 | 0.00217 | ramp | (11/4)·n + 7  →  (1/4)·n^2 + (1/4)·n - 3 | (3/512)·n^3 + (-5/32)·n^2 + (9/8)·n - 2 | 0.00234 | read B[i3, i2] (i0=0) |
| n^4 | 0.000345 | ramp | (17/4)·n + 9  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/1024)·n^3 + (-5/128)·n^2 + (1/2)·n - 2 | 0.000391 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.443 | ramp | 9  →  (17/8)·n | (49/128)·n^3 + (-49/32)·n^2 + (7/8)·n | 0.153 | read A[i3, i2] (i0=0) |
| n^3.5 | 0.389 | ramp | 9  →  (17/8)·n | (343/1024)·n^3 + (-133/128)·n^2 | 0.134 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.0613 | ramp | 20  →  (17/8)·n - 14 | (7/128)·n^3 + (-21/32)·n^2 + (7/4)·n | 0.0219 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.0613 | ramp | 20  →  (17/8)·n - 14 | (7/128)·n^3 + (-21/32)·n^2 + (7/4)·n | 0.0219 | read A[i3, i2] (i0=0) |
| n^3.5 | 0.0544 | ramp | 24  →  (17/8)·n | (49/1024)·n^3 + (-63/128)·n^2 + (7/8)·n | 0.0191 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.0516 | ramp | 26  →  (17/8)·n - 13 | (49/1024)·n^3 + (-147/128)·n^2 + (49/8)·n | 0.0191 | read C[i1, i3] (i0=0) |
| n^3.5 | 0.00719 | ramp | 27  →  (17/8)·n - 12 | (7/1024)·n^3 + (-7/32)·n^2 + (35/16)·n - 7 | 0.00273 | read C[i1, i3] (i0=0) |
| n^3.5 | 0.00697 | ramp | 41  →  (17/8)·n - 27 | (7/1024)·n^3 + (-35/128)·n^2 + (21/8)·n | 0.00273 | read C[i1, i3] (i0=0) |
| n^3.5 | 0.000971 | ramp | 42  →  (17/8)·n - 26 | (1/1024)·n^3 + (-3/64)·n^2 + (11/16)·n - 3 | 0.000391 | read C[i1, i3] (i0=0) |
| n^3 | 1.12 | level | 5 | (1/2)·n^3 + (-3/2)·n^2 + (15/8)·n | 0.2 | read A[i3, i2] (i0=0, i1=1, i3=0); read B[i1, i2] (i0=0) |
| n^3 | 0.978 | level | 5 | (7/16)·n^3 + (-21/16)·n^2 + (7/8)·n | 0.175 | read C[i1, i3] (i0=0) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (7/16)·n^2 + (7/8)·n | 0.175 | read B[i1, i2] (i0=0, i1=0); write A[i3, i2] (i0=0) (+1) |
| n^3 | 0.323 | ramp | n + 3  →  (1/4)·n^2 + 1 | (105/128)·n^2 + (-77/16)·n + 6 | 0.328/n | read A[i3, i2] (i0=0, i2=0); read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.149 | ramp | (5/4)·n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | (3/8)·n^2 + (-15/8)·n + 1 | 0.15/n | read A[i3, i2] (i0=0) |
| n^3 | 0.13 | ramp | (5/4)·n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | (21/64)·n^2 + (-11/8)·n | 0.131/n | read B[i3, i2] (i0=0) |
| n^3 | 0.108 | level | 3 | (1/16)·n^3 + (1/16)·n^2 | 0.025 | write A[i3, i2] (i0=0); write A[i1, i2] (i0=0) |
| n^3 | 0.0465 | ramp | (39/8)·n - 9  →  (5/16)·n^2 + (3/8)·n - 4 | (7/64)·n^2 + (-17/8)·n + 6 | 0.0437/n | read A[i1, i2] (i0=0) |
| n^3 | 0.0463 | ramp | (41/8)·n - 13  →  (5/16)·n^2 + (3/8)·n - 5 | (7/64)·n^2 + (-9/4)·n + 8 | 0.0437/n | read B[i1, i2] (i0=0, i3=0) |
| n^3 | 0.0452 | ramp | (9/4)·n + 4  →  (1/4)·n^2 + (-3/2)·n | (15/128)·n^2 + (-21/16)·n + 3 | 0.0469/n | read A[i3, i2] (i0=0, i2=0); read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0306 | level | (5/16)·n^2 + (1/2)·n | (7/128)·n^2 + (-23/16)·n + 9 | 0.0219/n | read C[i1, i3] (i0=0, i2=0) |
| n^3 | 0.0298 | ramp | (1/2)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/32)·n^2 + (-3/2)·n | 0.0375/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0298 | ramp | (1/2)·n + 4  →  (1/4)·n^2 + (1/4)·n - 4 | (3/32)·n^2 + (-3/2)·n | 0.0375/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0298 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/32)·n^2 + (-13/8)·n + 2 | 0.0375/n | read B[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0298 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/32)·n^2 + (-13/8)·n + 2 | 0.0375/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0298 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + (1/4)·n - 4 | (3/32)·n^2 + (-13/8)·n + 2 | 0.0375/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0249 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (5/64)·n^2 + (-5/4)·n | 0.0312/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0241 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-5/4)·n - 1 | (1/16)·n^2 + (-3/4)·n + 2 | 0.025/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0239 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-3/2)·n - 2 | (1/16)·n^2 + (-7/8)·n + 3 | 0.025/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0212 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-5/4)·n - 1 | (7/128)·n^2 + (-9/16)·n + 1 | 0.0219/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0212 | ramp | (11/4)·n + 5  →  (1/4)·n^2 + 1 | (7/128)·n^2 + (-11/16)·n + 2 | 0.0219/n | read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.021 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-3/2)·n - 2 | (7/128)·n^2 + (-11/16)·n + 2 | 0.0219/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0183 | ramp | 3·n + 2  →  (1/4)·n^2 + (1/4)·n - 2 | (3/64)·n^2 + (-1/2)·n + 1 | 0.0187/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00654 | ramp | (51/8)·n - 26  →  (5/16)·n^2 + (-1/4)·n - 14 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i1, i2] (i0=0) |
| n^3 | 0.00654 | ramp | (51/8)·n - 27  →  (5/16)·n^2 + (-1/4)·n - 15 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i1, i2] (i0=0, i3=0) |
| n^3 | 0.0049 | ramp | (5/2)·n + 7  →  (1/4)·n^2 - n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 6  →  (1/4)·n^2 + (-5/4)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 6  →  (1/4)·n^2 + (-5/4)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 5  →  (1/4)·n^2 + (-3/2)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00486 | ramp | 2·n + 5  →  (1/4)·n^2 + (-3/2)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00437 | level | (5/16)·n^2 + 2·n + (3/4) | (1/128)·n^2 + (-7/32)·n + (45/32) | 0.00313/n | read C[i1, i3] (i0=0, i2=0) |
| n^3 | 0.00437 | level | (5/16)·n^2 + (1/2)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.00313/n | read C[i1, i3] (i0=0, i2=0) |
| n^3 | 0.00291 | ramp | (9/2)·n + 4  →  (1/4)·n^2 + (-5/4)·n - 2 | (1/128)·n^2 + (-3/16)·n + 1 | 0.00313/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00291 | ramp | (17/4)·n + 6  →  (1/4)·n^2 + (-3/2)·n | (1/128)·n^2 + (-3/16)·n + 1 | 0.00313/n | read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.00291 | ramp | (17/4)·n + 3  →  (1/4)·n^2 + (-3/2)·n - 3 | (1/128)·n^2 + (-3/16)·n + 1 | 0.00313/n | read B[i3, i2] (i0=0) |
| n^2.5 | 1.6 | ramp | 5  →  (17/8)·n | (105/64)·n^2 + (-7/4)·n | 0.656/n | read A[i3, i2] (i0=0, i3=0); read A[i1, i2] (i0=0) |
| n^2.5 | 0.853 | ramp | 7  →  (17/8)·n | (7/8)·n^2 + (-7/4)·n | 0.35/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.837 | ramp | 21  →  (17/8)·n - 13 | (7/8)·n^2 + (-63/8)·n + 7 | 0.35/n | read C[i1, i3] (i0=0) |
| n^2.5 | 0.748 | ramp | 7  →  (17/8)·n | (49/64)·n^2 + (-7/8)·n | 0.306/n | read B[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.726 | ramp | 9  →  (17/8)·n - 13 | (49/64)·n^2 + (-49/8)·n | 0.306/n | read C[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.643 | ramp | 7  →  (17/8)·n | (21/32)·n^2 | 0.263/n | read B[i3, i2] (i0=0) |
| n^2.5 | 0.209 | ramp | 20  →  (17/8)·n - 14 | (7/32)·n^2 + (-7/4)·n | 0.0875/n | read A[i3, i2] (i0=0, i3=0); read B[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.105 | ramp | 22  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/8)·n | 0.0437/n | read B[i3, i2] (i0=0) |
| n^2.5 | 0.105 | ramp | 20  →  (17/8)·n - 14 | (7/64)·n^2 + (-7/8)·n | 0.0437/n | read B[i3, i2] (i0=0) |
| n^2.5 | 0.105 | ramp | 20  →  (17/8)·n - 14 | (7/64)·n^2 + (-7/8)·n | 0.0437/n | read C[i1, i1] (i0=0) |
| n^2.5 | 0.102 | ramp | 24  →  (17/8)·n - 27 | (7/64)·n^2 + (-7/4)·n | 0.0437/n | read C[i1, i3] (i0=0) |
| n^2.5 | 0.102 | ramp | 24  →  (17/8)·n - 27 | (7/64)·n^2 + (-7/4)·n | 0.0437/n | read C[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.101 | ramp | 10  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/4)·n + 7 | 0.0437/n | read C[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.0146 | ramp | 21  →  (17/8)·n - 13 | (1/64)·n^2 + (-1/4)·n + 1 | 0.00625/n | read C[i1, i1] (i0=0) |
| n^2.5 | 0.0142 | ramp | 25  →  (17/8)·n - 26 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read C[i1, i3] (i0=0) |
| n^2.5 | 0.0142 | ramp | 25  →  (17/8)·n - 26 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read C[i1, i3] (i0=0, i3=0) |
| n^2 | 2.24 | level | 5 | n^2 - n | 0.4/n | read A[i1, i2] (i0=0, i1=0, i2=0); read B[i1, i2] (i0=0, i1=0, i2=0) (+3) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 + (-7/8)·n | 0.35/n | read C[i1, i1] (i0=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 + (-7/8)·n | 0.35/n | read B[i1, i2] (i0=0, i3=0) |
| n^2 | 0.978 | level | 5 | (7/16)·n^2 + (-7/8)·n | 0.175/n | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.978 | level | 5 | (7/16)·n^2 + (-7/8)·n | 0.175/n | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.515 | ramp | (1/4)·n^2 + (1/8)·n + 3  →  (5/16)·n^2 + (3/8)·n | n - 4 | 0.4·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.513 | ramp | (1/4)·n^2 + (1/8)·n + 2  →  (5/16)·n^2 + (1/4)·n - 1 | n - 5 | 0.4·n^-2 | read B[i1, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.489 | level | (5/16)·n^2 + (1/2)·n | (7/8)·n - 8 | 0.35·n^-2 | read C[i1, i3] (i0=0, i2=0, i3=0) |
| n^2 | 0.489 | level | (5/16)·n^2 + (1/2)·n | (7/8)·n - 8 | 0.35·n^-2 | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.462 | ramp | n + 1  →  (1/4)·n^2 + (1/8)·n | (11/8)·n - 1 | 0.55·n^-2 | read A[i3, i2] (i0=0); read B[i3, i2] (i0=0) |
| n^2 | 0.318 | ramp | (23/8)·n - 3  →  (5/16)·n^2 + (-1/4)·n + 5 | (7/8)·n - 3 | 0.35·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.317 | ramp | (25/8)·n - 5  →  (5/16)·n^2 + (-1/4)·n + 4 | (7/8)·n - 4 | 0.35·n^-2 | read B[i1, i2] (i0=0, i3=0) |
| n^2 | 0.292 | ramp | (1/2)·n + 3  →  (1/4)·n^2 + 1 | (7/8)·n - 1 | 0.35·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.292 | ramp | (1/2)·n + 2  →  (1/4)·n^2 | (7/8)·n - 1 | 0.35·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + 1 | (7/8)·n - 2 | 0.35·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + (-1/8)·n + 2 | (7/8)·n - 2 | 0.35·n^-2 | read B[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 1  →  (1/4)·n^2 + (-1/8)·n | (7/8)·n - 2 | 0.35·n^-2 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.252 | ramp | n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | (3/4)·n - 1 | 0.3·n^-2 | read B[i3, i2] (i0=0, i3=0) |
| n^2 | 0.252 | ramp | n  →  (1/4)·n^2 + (1/4)·n - 2 | (3/4)·n - 1 | 0.3·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.252 | ramp | (3/4)·n + 1  →  (1/4)·n^2 + (1/8)·n | (3/4)·n | 0.3·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.252 | ramp | (3/4)·n - 1  →  (1/4)·n^2 + (1/8)·n - 2 | (3/4)·n | 0.3·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.251 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + 1 | (3/4)·n - 1 | 0.3·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0824 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-11/8)·n | (1/4)·n - 2 | 0.1·n^-2 | read A[i3, i2] (i0=0); read B[i3, i2] (i0=0) |
| n^2 | 0.082 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-13/8)·n | (1/4)·n - 2 | 0.1·n^-2 | read A[i3, i2] (i0=0); read B[i3, i2] (i0=0) |
| n^2 | 0.0815 | ramp | (35/8)·n - 3  →  (1/4)·n^2 + (-15/8)·n + 3 | (1/4)·n - 4 | 0.1·n^-2 | read A[i1, i2] (i0=0, i1=0); read B[i1, i2] (i0=0, i1=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (9/4)·n + (7/16) | (1/8)·n + (-9/8) | 0.05·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.05·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (9/4)·n + (7/16) | (1/8)·n + (-9/8) | 0.05·n^-2 | read B[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.05·n^-2 | read B[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n + 4 | 0.05·n^-2 | read C[i1, i3] (i0=0, i2=0, i3=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (7/4)·n + (9/4) | (1/8)·n + (-5/4) | 0.05·n^-2 | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + 2·n + (27/16) | (1/8)·n + (-17/8) | 0.05·n^-2 | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/4)·n + 3 | (1/8)·n - 2 | 0.05·n^-2 | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + 2·n + (3/4) | (1/8)·n + (-9/4) | 0.05·n^-2 | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 3 | 0.05·n^-2 | read C[i1, i3] (i0=0, i2=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (9/4)·n + (7/16) | (1/8)·n + (-9/8) | 0.05·n^-2 | read C[i1, i1] (i0=0, i2=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.05·n^-2 | read C[i1, i1] (i0=0, i2=0) |
| n^2 | 0.0664 | ramp | (5/16)·n^2 + (1/4)·n + 4  →  (5/16)·n^2 + (1/2)·n - 2 | (1/8)·n - 2 | 0.05·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.0664 | ramp | (5/16)·n^2 + (1/4)·n + 3  →  (5/16)·n^2 + (1/2)·n - 3 | (1/8)·n - 2 | 0.05·n^-2 | read B[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0449 | ramp | (35/8)·n - 9  →  (5/16)·n^2 + (-17/8)·n + 15 | (1/8)·n - 1 | 0.05·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.0449 | ramp | (35/8)·n - 10  →  (5/16)·n^2 + (-17/8)·n + 14 | (1/8)·n - 1 | 0.05·n^-2 | read B[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0414 | ramp | (11/4)·n + 2  →  (1/4)·n^2 + (-9/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 5  →  (1/4)·n^2 + (-5/4)·n + 1 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-5/4)·n - 1 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-5/4)·n - 2 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-11/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.0412 | ramp | (5/2)·n  →  (1/4)·n^2 + (-11/8)·n - 2 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 5  →  (1/4)·n^2 + (-3/2)·n + 1 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 4  →  (1/4)·n^2 + (-3/2)·n | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 4  →  (1/4)·n^2 + (-3/2)·n | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 3  →  (1/4)·n^2 + (-3/2)·n - 1 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-3/2)·n - 2 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 1  →  (1/4)·n^2 + (-3/2)·n - 3 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.041 | ramp | (9/4)·n + 4  →  (1/4)·n^2 + (-13/8)·n + 2 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.041 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-13/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.041 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-13/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.041 | ramp | (9/4)·n  →  (1/4)·n^2 + (-13/8)·n - 2 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.0409 | ramp | (39/8)·n - 10  →  (1/4)·n^2 + (-15/8)·n + 8 | (1/8)·n - 2 | 0.05·n^-2 | read B[i1, i2] (i0=0, i1=2, i3=0) |
| n^2 | 0.0408 | ramp | (37/8)·n - 6  →  (1/4)·n^2 + (-15/8)·n + 6 | (1/8)·n - 2 | 0.05·n^-2 | read A[i1, i2] (i0=0, i1=1) |
| n^2 | 0.0408 | ramp | (37/8)·n - 7  →  (1/4)·n^2 + (-15/8)·n + 5 | (1/8)·n - 2 | 0.05·n^-2 | read B[i1, i2] (i0=0, i1=1, i3=0) |
| n^1.5 | 0.0941 | ramp | (1/2)·n + 5  →  (3/4)·n - 1 | (1/8)·n - 2 | 0.05·n^-2 | read B[i3, i2] (i0=0, i1=2, i3=0) |
| n^1.5 | 0.0939 | ramp | (1/2)·n + 4  →  (3/4)·n - 2 | (1/8)·n - 2 | 0.05·n^-2 | read A[i3, i2] (i0=0, i1=2, i3=0) |
| n^1.5 | 0.0729 | ramp | (1/4)·n + 5  →  (1/2)·n - 1 | (1/8)·n - 2 | 0.05·n^-2 | read B[i3, i2] (i0=0, i1=1, i3=0) |
| n^1.5 | 0.0723 | ramp | (1/4)·n + 3  →  (1/2)·n - 3 | (1/8)·n - 2 | 0.05·n^-2 | read A[i3, i2] (i0=0, i1=1, i3=0) |
| n^1 | 14.2 | level | 4 | (57/8)·n - 8 | 2.85·n^-2 | read C[i1, i1] (i0=0, i1=0, i2=0); read C[i1, i1] (i0=0, i1=0) (+1) |
| n^1 | 2.32 | level | 7 | (7/8)·n | 0.35·n^-2 | read B[i1, i2] (i0=0, i1=1, i2=0, i3=0); read B[i3, i2] (i0=0, i1=2, i2=0, i3=0) (+1) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.35·n^-2 | read B[i3, i2] (i0=0, i1=1, i3=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n + (-7/8) | 0.35·n^-2 | read C[i1, i1] (i0=0, i2=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.35·n^-2 | read C[i1, i1] (i0=0, i2=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.35·n^-2 | read B[i1, i2] (i0=0, i1=0, i2=0); read C[i1, i1] (i0=0, i1=0, i2=0) (+1) |
| n^1 | 1.12 | level | (5/16)·n^2 + (1/2)·n | 2 | 0.8·n^-3 | read C[i1, i3] (i0=0, i1=1, i2=0, i3=0); read C[i1, i3] (i0=0, i1=8, i2=0, i3=0) |
| n^1 | 1.12 | level | (5/16)·n^2 + (9/4)·n + (7/16) | 2 | 0.8·n^-3 | read B[i1, i2] (i0=0, i2=0, i3=0); read A[i1, i2] (i0=0, i2=0) |
| n^1 | 1.12 | level | (5/16)·n^2 + (1/2)·n | 2 | 0.8·n^-3 | read B[i1, i2] (i0=0, i2=0, i3=0); read A[i1, i2] (i0=0, i2=0) |
| n^1 | 1 | level | (1/4)·n^2 + (15/8)·n + (7/8) | 2 | 0.8·n^-3 | read A[i1, i2] (i0=0, i1=0, i2=0); read B[i1, i2] (i0=0, i1=1, i2=0, i3=0) |
| n^1 | 1 | level | (1/4)·n^2 + (1/8)·n | 2 | 0.8·n^-3 | read A[i1, i2] (i0=0, i1=0, i2=0); read B[i1, i2] (i0=0, i1=1, i2=0, i3=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.35·n^-2 | read A[i1, i2] (i0=0, i1=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/8)·n + 3 | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/8)·n + 2 | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i3=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i3=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + 2·n + (27/16) | 1 | 0.4·n^-3 | read C[i1, i3] (i0=0, i2=0, i3=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/4)·n + 3 | 1 | 0.4·n^-3 | read C[i1, i3] (i0=0, i2=0, i3=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (17/8)·n + (-7/16) | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i2=0, i3=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (3/8)·n - 1 | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i2=0, i3=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (9/4)·n + (7/16) | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (7/4)·n + (5/4) | 1 | 0.4·n^-3 | read C[i1, i3] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/4)·n + 2 | 1 | 0.4·n^-3 | read C[i1, i3] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.4·n^-3 | read C[i1, i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + 2·n + (3/4) | 1 | 0.4·n^-3 | read C[i1, i3] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.4·n^-3 | read C[i1, i3] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + 2·n + (27/16) | 1 | 0.4·n^-3 | read C[i1, i3] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.4·n^-3 | read C[i1, i1] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + 2·n + (27/16) | 1 | 0.4·n^-3 | read C[i1, i1] (i0=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (15/8) | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i1=2, i2=0, i3=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 1 | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i1=2, i2=0, i3=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (23/8) | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0, i1=1, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 2 | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0, i1=1, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (15/8)·n + (7/8) | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.354 | level | 8 | (1/8)·n - 1 | 0.05·n^-2 | read C[i1, i3] (i0=0, i1=8, i3=0) |
| n^0.5 | 3.08 | level | (19/8)·n - 1 | 2 | 0.8·n^-3 | read A[i1, i2] (i0=0, i1=0); read B[i1, i2] (i0=0, i1=0) |
| n^0.5 | 1.7 | level | (23/8)·n - 4 | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i1=2, i3=0) |
| n^0.5 | 1.62 | level | (21/8)·n - 2 | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0, i1=1) |
| n^0.5 | 1.62 | level | (21/8)·n - 3 | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i1=1, i3=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=2, i3=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.4·n^-3 | read A[i3, i2] (i0=0, i1=2, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 3 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=2, i2=0, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=1, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n - 2 | 1 | 0.4·n^-3 | read A[i3, i2] (i0=0, i1=1, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.4·n^-3 | read B[i1, i2] (i0=0, i1=0, i2=0); read A[i3, i2] (i0=0, i1=2, i2=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.4·n^-3 | read A[i1, i2] (i0=0, i1=0, i2=0); read A[i3, i2] (i0=0, i1=1, i2=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 3 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=1, i2=0, i3=0) |

Symmetric multiply: the symmetric operand and the accumulating output are re-read across output rows — ramps to (1/4)n^2 + O(n) lines with population n^3/16, coefficient 0.0466·n^4 (numerically identical to syr2k's two-source structure), headroom +1.0. The old headroom-0 reading was the rendering artifact.

## symm — single-shot  [`exact`]

Accesses $A(n) = (5/2)·n^3 + (3/2)·n^2$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0466·n^4  +  1.08·n^3.5  +  3.99·n^3  +  6.28·n^2.5  +  7.71·n^2  +  0.333·n^1.5  +  23.2·n^1  +  5.56·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0176 | ramp | n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/64)·n^3 + (-63/64)·n^2 + (31/8)·n - 2 | 0.0187 | read A[i3, i2] (i0=0) |
| n^4 | 0.0155 | ramp | n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (21/512)·n^3 + (-53/64)·n^2 + (11/4)·n | 0.0164 | read B[i3, i2] (i0=0) |
| n^4 | 0.00286 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/128)·n^3 + (-7/32)·n^2 + (7/4)·n - 4 | 0.00313 | read A[i3, i2] (i0=0) |
| n^4 | 0.00283 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/128)·n^3 + (-15/64)·n^2 + (17/8)·n - 6 | 0.00313 | read B[i3, i2] (i0=0) |
| n^4 | 0.00283 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/128)·n^3 + (-15/64)·n^2 + (17/8)·n - 6 | 0.00313 | read A[i3, i2] (i0=0) |
| n^4 | 0.00251 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (7/1024)·n^3 + (-23/128)·n^2 + (5/4)·n - 2 | 0.00273 | read B[i3, i2] (i0=0) |
| n^4 | 0.00217 | ramp | (11/4)·n + 7  →  (1/4)·n^2 + (1/4)·n - 3 | (3/512)·n^3 + (-5/32)·n^2 + (9/8)·n - 2 | 0.00234 | read B[i3, i2] (i0=0) |
| n^4 | 0.000345 | ramp | (17/4)·n + 9  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/1024)·n^3 + (-5/128)·n^2 + (1/2)·n - 2 | 0.000391 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.443 | ramp | 9  →  (17/8)·n | (49/128)·n^3 + (-49/32)·n^2 + (7/8)·n | 0.153 | read A[i3, i2] (i0=0) |
| n^3.5 | 0.389 | ramp | 9  →  (17/8)·n | (343/1024)·n^3 + (-133/128)·n^2 | 0.134 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.0613 | ramp | 20  →  (17/8)·n - 14 | (7/128)·n^3 + (-21/32)·n^2 + (7/4)·n | 0.0219 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.0613 | ramp | 20  →  (17/8)·n - 14 | (7/128)·n^3 + (-21/32)·n^2 + (7/4)·n | 0.0219 | read A[i3, i2] (i0=0) |
| n^3.5 | 0.0544 | ramp | 24  →  (17/8)·n | (49/1024)·n^3 + (-63/128)·n^2 + (7/8)·n | 0.0191 | read B[i3, i2] (i0=0) |
| n^3.5 | 0.0516 | ramp | 26  →  (17/8)·n - 13 | (49/1024)·n^3 + (-147/128)·n^2 + (49/8)·n | 0.0191 | read C[i1, i3] (i0=0) |
| n^3.5 | 0.00719 | ramp | 27  →  (17/8)·n - 12 | (7/1024)·n^3 + (-7/32)·n^2 + (35/16)·n - 7 | 0.00273 | read C[i1, i3] (i0=0) |
| n^3.5 | 0.00697 | ramp | 41  →  (17/8)·n - 27 | (7/1024)·n^3 + (-35/128)·n^2 + (21/8)·n | 0.00273 | read C[i1, i3] (i0=0) |
| n^3.5 | 0.000971 | ramp | 42  →  (17/8)·n - 26 | (1/1024)·n^3 + (-3/64)·n^2 + (11/16)·n - 3 | 0.000391 | read C[i1, i3] (i0=0) |
| n^3 | 1.12 | level | 5 | (1/2)·n^3 + (-1/2)·n^2 | 0.2 | read B[i3, i2] (i0=0, i1=1, i2=0, i3=0); read A[i3, i2] (i0=0, i1=2, i2=0, i3=0) (+1) |
| n^3 | 0.978 | level | 5 | (7/16)·n^3 + (-7/8)·n^2 | 0.175 | read C[i1, i3] (i0=0) |
| n^3 | 0.866 | level | 3 | (1/2)·n^3 + (1/2)·n^2 + (7/8)·n | 0.2 | read C[i1, i1] (i0=0, i1=0); write A[i3, i2] (i0=0) (+1) |
| n^3 | 0.323 | ramp | n + 3  →  (1/4)·n^2 + 1 | (105/128)·n^2 + (-77/16)·n + 6 | 0.328/n | read A[i3, i2] (i0=0, i2=0); read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.149 | ramp | (5/4)·n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | (3/8)·n^2 + (-15/8)·n + 1 | 0.15/n | read A[i3, i2] (i0=0) |
| n^3 | 0.13 | ramp | (5/4)·n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | (21/64)·n^2 + (-11/8)·n | 0.131/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0452 | ramp | (9/4)·n + 4  →  (1/4)·n^2 + (-3/2)·n | (15/128)·n^2 + (-21/16)·n + 3 | 0.0469/n | read A[i3, i2] (i0=0, i2=0); read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0298 | ramp | (1/2)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/32)·n^2 + (-3/2)·n | 0.0375/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0298 | ramp | (1/2)·n + 4  →  (1/4)·n^2 + (1/4)·n - 4 | (3/32)·n^2 + (-3/2)·n | 0.0375/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0298 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/32)·n^2 + (-13/8)·n + 2 | 0.0375/n | read B[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0298 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (3/32)·n^2 + (-13/8)·n + 2 | 0.0375/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0298 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + (1/4)·n - 4 | (3/32)·n^2 + (-13/8)·n + 2 | 0.0375/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.0249 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (5/64)·n^2 + (-5/4)·n | 0.0312/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0241 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-5/4)·n - 1 | (1/16)·n^2 + (-3/4)·n + 2 | 0.025/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0239 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-3/2)·n - 2 | (1/16)·n^2 + (-7/8)·n + 3 | 0.025/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0212 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-5/4)·n - 1 | (7/128)·n^2 + (-9/16)·n + 1 | 0.0219/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0212 | ramp | (11/4)·n + 5  →  (1/4)·n^2 + 1 | (7/128)·n^2 + (-11/16)·n + 2 | 0.0219/n | read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.021 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-3/2)·n - 2 | (7/128)·n^2 + (-11/16)·n + 2 | 0.0219/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0183 | ramp | 3·n + 2  →  (1/4)·n^2 + (1/4)·n - 2 | (3/64)·n^2 + (-1/2)·n + 1 | 0.0187/n | read B[i3, i2] (i0=0) |
| n^3 | 0.0049 | ramp | (5/2)·n + 7  →  (1/4)·n^2 - n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 6  →  (1/4)·n^2 + (-5/4)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 6  →  (1/4)·n^2 + (-5/4)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read B[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00486 | ramp | 2·n + 6  →  (1/4)·n^2 + (-3/2)·n - 4 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 5  →  (1/4)·n^2 + (-3/2)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00486 | ramp | 2·n + 5  →  (1/4)·n^2 + (-3/2)·n - 5 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read A[i3, i2] (i0=0, i3=0) |
| n^3 | 0.00291 | ramp | (9/2)·n + 4  →  (1/4)·n^2 + (-5/4)·n - 2 | (1/128)·n^2 + (-3/16)·n + 1 | 0.00313/n | read B[i3, i2] (i0=0) |
| n^3 | 0.00291 | ramp | (17/4)·n + 6  →  (1/4)·n^2 + (-3/2)·n | (1/128)·n^2 + (-3/16)·n + 1 | 0.00313/n | read B[i3, i2] (i0=0, i2=0) |
| n^3 | 0.00291 | ramp | (17/4)·n + 3  →  (1/4)·n^2 + (-3/2)·n - 3 | (1/128)·n^2 + (-3/16)·n + 1 | 0.00313/n | read B[i3, i2] (i0=0) |
| n^2.5 | 0.853 | ramp | 5  →  (17/8)·n | (7/8)·n^2 + (-7/8)·n | 0.35/n | read A[i1, i2] (i0=0) |
| n^2.5 | 0.853 | ramp | 7  →  (17/8)·n | (7/8)·n^2 + (-7/4)·n | 0.35/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.837 | ramp | 21  →  (17/8)·n - 13 | (7/8)·n^2 + (-63/8)·n + 7 | 0.35/n | read C[i1, i3] (i0=0) |
| n^2.5 | 0.748 | ramp | 7  →  (17/8)·n | (49/64)·n^2 + (-7/8)·n | 0.306/n | read B[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.748 | ramp | 7  →  (17/8)·n | (49/64)·n^2 + (-7/8)·n | 0.306/n | read A[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.726 | ramp | 9  →  (17/8)·n - 13 | (49/64)·n^2 + (-49/8)·n | 0.306/n | read C[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.643 | ramp | 7  →  (17/8)·n | (21/32)·n^2 | 0.263/n | read B[i3, i2] (i0=0) |
| n^2.5 | 0.209 | ramp | 20  →  (17/8)·n - 14 | (7/32)·n^2 + (-7/4)·n | 0.0875/n | read A[i3, i2] (i0=0, i3=0); read B[i3, i2] (i0=0, i3=0) |
| n^2.5 | 0.105 | ramp | 22  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/8)·n | 0.0437/n | read B[i3, i2] (i0=0) |
| n^2.5 | 0.105 | ramp | 20  →  (17/8)·n - 14 | (7/64)·n^2 + (-7/8)·n | 0.0437/n | read B[i3, i2] (i0=0) |
| n^2.5 | 0.105 | ramp | 20  →  (17/8)·n - 14 | (7/64)·n^2 + (-7/8)·n | 0.0437/n | read C[i1, i1] (i0=0) |
| n^2.5 | 0.102 | ramp | 24  →  (17/8)·n - 27 | (7/64)·n^2 + (-7/4)·n | 0.0437/n | read C[i1, i3] (i0=0) |
| n^2.5 | 0.102 | ramp | 24  →  (17/8)·n - 27 | (7/64)·n^2 + (-7/4)·n | 0.0437/n | read C[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.101 | ramp | 10  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/4)·n + 7 | 0.0437/n | read C[i1, i3] (i0=0, i3=0) |
| n^2.5 | 0.0146 | ramp | 21  →  (17/8)·n - 13 | (1/64)·n^2 + (-1/4)·n + 1 | 0.00625/n | read C[i1, i1] (i0=0) |
| n^2.5 | 0.0142 | ramp | 25  →  (17/8)·n - 26 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read C[i1, i3] (i0=0) |
| n^2.5 | 0.0142 | ramp | 25  →  (17/8)·n - 26 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00625/n | read C[i1, i3] (i0=0, i3=0) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 | 0.35/n | read C[i1, i1] (i0=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 + (-7/8)·n | 0.35/n | read B[i1, i2] (i0=0, i3=0) |
| n^2 | 0.462 | ramp | n + 1  →  (1/4)·n^2 + (1/8)·n | (11/8)·n - 1 | 0.55·n^-2 | read A[i3, i2] (i0=0); read B[i3, i2] (i0=0) |
| n^2 | 0.293 | ramp | (3/4)·n - 1  →  (1/4)·n^2 + (1/8)·n - 2 | (7/8)·n - 1 | 0.35·n^-2 | read A[i3, i2] (i0=0) |
| n^2 | 0.292 | ramp | (1/2)·n + 3  →  (1/4)·n^2 + 1 | (7/8)·n - 1 | 0.35·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.292 | ramp | (1/2)·n + 2  →  (1/4)·n^2 | (7/8)·n - 1 | 0.35·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + 1 | (7/8)·n - 2 | 0.35·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + (-1/8)·n + 2 | (7/8)·n - 2 | 0.35·n^-2 | read B[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 1  →  (1/4)·n^2 + (-1/8)·n | (7/8)·n - 2 | 0.35·n^-2 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.252 | ramp | n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | (3/4)·n - 1 | 0.3·n^-2 | read B[i3, i2] (i0=0, i3=0) |
| n^2 | 0.252 | ramp | n  →  (1/4)·n^2 + (1/4)·n - 2 | (3/4)·n - 1 | 0.3·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.252 | ramp | (3/4)·n + 1  →  (1/4)·n^2 + (1/8)·n | (3/4)·n | 0.3·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.251 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + 1 | (3/4)·n - 1 | 0.3·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0824 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-11/8)·n | (1/4)·n - 2 | 0.1·n^-2 | read A[i3, i2] (i0=0); read B[i3, i2] (i0=0) |
| n^2 | 0.0821 | ramp | (9/4)·n + 4  →  (1/4)·n^2 + (-3/2)·n | (1/4)·n - 2 | 0.1·n^-2 | read A[i3, i2] (i0=0, i2=0); read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.082 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-13/8)·n | (1/4)·n - 2 | 0.1·n^-2 | read A[i3, i2] (i0=0); read B[i3, i2] (i0=0) |
| n^2 | 0.0414 | ramp | (11/4)·n + 2  →  (1/4)·n^2 + (-9/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 5  →  (1/4)·n^2 + (-5/4)·n + 1 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-5/4)·n - 1 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-5/4)·n - 2 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-11/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 5  →  (1/4)·n^2 + (-3/2)·n + 1 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 3  →  (1/4)·n^2 + (-3/2)·n - 1 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-3/2)·n - 2 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 1  →  (1/4)·n^2 + (-3/2)·n - 3 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.041 | ramp | (9/4)·n + 4  →  (1/4)·n^2 + (-13/8)·n + 2 | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.041 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-13/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read B[i3, i2] (i0=0) |
| n^2 | 0.041 | ramp | (9/4)·n + 2  →  (1/4)·n^2 + (-13/8)·n | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.041 | ramp | (9/4)·n  →  (1/4)·n^2 + (-13/8)·n - 2 | (1/8)·n - 1 | 0.05·n^-2 | read A[i3, i2] (i0=0) |
| n^1.5 | 0.0941 | ramp | (1/2)·n + 5  →  (3/4)·n - 1 | (1/8)·n - 2 | 0.05·n^-2 | read B[i3, i2] (i0=0, i1=2, i3=0) |
| n^1.5 | 0.0939 | ramp | (1/2)·n + 4  →  (3/4)·n - 2 | (1/8)·n - 2 | 0.05·n^-2 | read A[i3, i2] (i0=0, i1=2, i3=0) |
| n^1.5 | 0.0729 | ramp | (1/4)·n + 5  →  (1/2)·n - 1 | (1/8)·n - 2 | 0.05·n^-2 | read B[i3, i2] (i0=0, i1=1, i3=0) |
| n^1.5 | 0.0723 | ramp | (1/4)·n + 3  →  (1/2)·n - 3 | (1/8)·n - 2 | 0.05·n^-2 | read A[i3, i2] (i0=0, i1=1, i3=0) |
| n^1 | 14.2 | level | 4 | (57/8)·n - 8 | 2.85·n^-2 | read C[i1, i1] (i0=0, i1=0); read C[i1, i3] (i0=0, i3=0) |
| n^1 | 2.32 | level | 7 | (7/8)·n | 0.35·n^-2 | read B[i3, i2] (i0=0, i1=2, i2=0, i3=0); read C[i1, i3] (i0=0, i1=8, i3=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.35·n^-2 | read B[i3, i2] (i0=0, i1=1, i3=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.35·n^-2 | read A[i3, i2] (i0=0, i1=1, i3=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.35·n^-2 | read B[i1, i2] (i0=0, i1=0) |
| n^1 | 0.875 | level | 1 | (7/8)·n | 0.35·n^-2 | read A[i1, i2] (i0=0, i1=0) |
| n^1 | 0.354 | level | 8 | (1/8)·n - 1 | 0.05·n^-2 | read C[i1, i3] (i0=0, i1=8, i3=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=2, i3=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.4·n^-3 | read A[i3, i2] (i0=0, i1=2, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=1, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 3 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=2, i2=0, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.4·n^-3 | read A[i3, i2] (i0=0, i1=2, i2=0, i3=0) |
| n^0.5 | 0.707 | level | (1/2)·n - 2 | 1 | 0.4·n^-3 | read A[i3, i2] (i0=0, i1=1, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.4·n^-3 | read A[i3, i2] (i0=0, i1=1, i2=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 3 | 1 | 0.4·n^-3 | read B[i3, i2] (i0=0, i1=1, i2=0, i3=0) |

Symmetric multiply: the symmetric operand and the accumulating output are re-read across output rows — ramps to (1/4)n^2 + O(n) lines with population n^3/16, coefficient 0.0466·n^4 (numerically identical to syr2k's two-source structure), headroom +1.0. The old headroom-0 reading was the rendering artifact.

## syr2k — infinite-repeat  [`exact`]

Accesses $A(n) = 3·n^3 + 4·n^2 + n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0466·n^4  +  1.08·n^3.5  +  4.98·n^3  +  5.47·n^2.5  +  20.9·n^2  +  0.572·n^1.5  +  57.4·n^1  +  70.9·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0409 | ramp | n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^3 + (-77/32)·n^2 + (45/4)·n - 12 | 0.0365 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^4 | 0.00572 | ramp | (11/4)·n + 8  →  (1/4)·n^2 + (1/4)·n - 2 | (1/64)·n^3 + (-15/32)·n^2 + (17/4)·n - 12 | 0.00521 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3.5 | 0.885 | ramp | 9  →  (17/8)·n | (49/64)·n^3 + (-49/16)·n^2 + (7/4)·n | 0.255 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3.5 | 0.124 | ramp | 24  →  (17/8)·n | (7/64)·n^3 + (-21/16)·n^2 + (7/2)·n | 0.0365 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3.5 | 0.0583 | ramp | 23  →  (17/8)·n - 16 | (7/128)·n^3 + (-91/64)·n^2 + (35/4)·n | 0.0182 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00814 | ramp | 25  →  (17/8)·n - 14 | (1/128)·n^3 + (-17/64)·n^2 + (23/8)·n - 10 | 0.0026 | read A[i1, i4] (i0=0) |
| n^3 | 1.96 | level | 5 | (7/8)·n^3 + (-13/8)·n^2 - n | 0.292 | read C[i4, i3] (i0=0, i1=1, i3=0, i4=0); read C[i1, i3] (i0=0) (+1) |
| n^3 | 0.978 | level | 5 | (7/16)·n^3 + (-7/8)·n^2 | 0.146 | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^3 | 0.438 | level | 1 | (7/16)·n^3 | 0.146 | read A[i1, i4] (i0=0) |
| n^3 | 0.393 | ramp | n + 4  →  (1/4)·n^2 + 2 | n^2 - 7·n + 12 | 0.333/n | read B[i4, i3] (i0=0, i3=0); read C[i4, i3] (i0=0, i3=0) |
| n^3 | 0.306 | level | 6 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0417 | read B[i4, i3] (i0=0, i1=0, i3=0, i4=0); read B[i1, i3] (i0=0, i1=1, i3=0, i4=0) (+3) |
| n^3 | 0.148 | ramp | (5/4)·n + 2  →  (1/4)·n^2 + (1/4)·n | (3/8)·n^2 + (-15/8)·n + 1 | 0.125/n | read C[i4, i3] (i0=0) |
| n^3 | 0.148 | ramp | (5/4)·n + 2  →  (1/4)·n^2 + (1/4)·n | (3/8)·n^2 + (-15/8)·n + 1 | 0.125/n | read B[i4, i3] (i0=0) |
| n^3 | 0.14 | level | 5 | (1/16)·n^3 + (3/8)·n^2 | 0.0208 | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^3 | 0.0511 | ramp | (67/8)·n - 41  →  (5/16)·n^2 + (3/8)·n - 4 | (1/8)·n^2 + (-17/4)·n + 36 | 0.0417/n | read B[i1, i3] (i0=0, i4=0) |
| n^3 | 0.0511 | ramp | (67/8)·n - 42  →  (5/16)·n^2 + (3/8)·n - 5 | (1/8)·n^2 + (-17/4)·n + 36 | 0.0417/n | read C[i1, i3] (i0=0, i4=0) |
| n^3 | 0.0485 | ramp | (11/4)·n + 4  →  (1/4)·n^2 + (1/4)·n | (1/8)·n^2 + (-3/2)·n + 4 | 0.0417/n | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3 | 0.0483 | ramp | 3·n + 3  →  (1/4)·n^2 + (1/4)·n - 1 | (1/8)·n^2 + (-7/4)·n + 6 | 0.0417/n | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3 | 0.0396 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (1/8)·n^2 + (-19/8)·n + 6 | 0.0417/n | read C[i4, i3] (i0=0, i4=0) |
| n^3 | 0.0396 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + (1/4)·n - 4 | (1/8)·n^2 + (-19/8)·n + 6 | 0.0417/n | read B[i4, i3] (i0=0, i4=0) |
| n^3 | 0.0349 | level | (5/16)·n^2 + (1/2)·n | (1/16)·n^2 + (-13/8)·n + 10 | 0.0208/n | read A[i1, i2] (i0=0) |
| n^3 | 0.0347 | ramp | (1/2)·n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0365/n | read C[i4, i3] (i0=0) |
| n^3 | 0.0347 | ramp | (1/2)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0365/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0347 | ramp | (3/4)·n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^2 - 2·n + 4 | 0.0365/n | read C[i4, i3] (i0=0) |
| n^3 | 0.0347 | ramp | (3/4)·n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^2 - 2·n + 4 | 0.0365/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0049 | ramp | (5/2)·n + 8  →  (1/4)·n^2 - n - 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read C[i4, i3] (i0=0) |
| n^3 | 0.0049 | ramp | (5/2)·n + 8  →  (1/4)·n^2 - n - 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read B[i4, i3] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 8  →  (1/4)·n^2 + (-5/4)·n - 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read C[i4, i3] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read B[i4, i3] (i0=0) |
| n^2.5 | 1.71 | ramp | 7  →  (17/8)·n | (7/4)·n^2 + (-7/2)·n | 0.583/n | read B[i4, i3] (i0=0, i4=0); read C[i4, i3] (i0=0, i4=0) |
| n^2.5 | 0.958 | ramp | 20  →  (17/8)·n - 14 | n^2 - 8·n | 0.333/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.937 | ramp | 6  →  (17/8)·n - 16 | n^2 - 10·n + 9 | 0.333/n | read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.747 | ramp | 7  →  (17/8)·n | (49/64)·n^2 + (-7/8)·n | 0.255/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.747 | ramp | 7  →  (17/8)·n | (49/64)·n^2 + (-7/8)·n | 0.255/n | read C[i4, i3] (i0=0) |
| n^2.5 | 0.149 | ramp | 21  →  (17/8)·n - 30 | (11/64)·n^2 + (-13/4)·n + 8 | 0.0573/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.105 | ramp | 22  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/8)·n | 0.0365/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.105 | ramp | 22  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/8)·n | 0.0365/n | read C[i4, i3] (i0=0) |
| n^2.5 | 0.0141 | ramp | 23  →  (17/8)·n - 28 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read A[i1, i4] (i0=0) |
| n^2 | 2.32 | level | 2 | (105/64)·n^2 + (7/8)·n | 0.547/n | read B[i4, i3] (i0=0, i1=0, i4=0); read B[i4, i3] (i0=0) (+1) |
| n^2 | 2.14 | level | 6 | (7/8)·n^2 - 7·n | 0.292/n | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 - 7·n | 0.292/n | read C[i1, i3] (i0=0, i4=0) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 + n - 1 | 0.292/n | read A[i1, i4] (i0=0, i1=8, i4=0); read C[i1, i3] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 | 0.583/n | read A[i1, i2] (i0=0); write A[i1, i2] (i0=0) (+1) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 | 0.292/n | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (1/2)·n | n - 9 | 0.333·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.559 | level | (5/16)·n^2 + (1/2)·n | n - 9 | 0.333·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.505 | ramp | (1/4)·n^2 + (1/8)·n + 10  →  (5/16)·n^2 + (3/8)·n | n - 10 | 0.333·n^-2 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.505 | ramp | (1/4)·n^2 + (1/8)·n + 9  →  (5/16)·n^2 + (3/8)·n - 1 | n - 10 | 0.333·n^-2 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.334 | ramp | (51/8)·n - 26  →  (1/4)·n^2 + (-15/8)·n + 56 | n - 16 | 0.333·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.334 | ramp | (51/8)·n - 27  →  (1/4)·n^2 + (-15/8)·n + 55 | n - 16 | 0.333·n^-2 | read C[i1, i3] (i0=0, i4=0) |
| n^2 | 0.334 | ramp | n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | n - 3 | 0.333·n^-2 | read C[i4, i3] (i0=0, i4=0) |
| n^2 | 0.334 | ramp | n  →  (1/4)·n^2 + (1/4)·n - 2 | n - 3 | 0.333·n^-2 | read B[i4, i3] (i0=0, i4=0) |
| n^2 | 0.333 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + 2 | n - 3 | 0.333·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.333 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + 1 | n - 3 | 0.333·n^-2 | read C[i4, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.333 | ramp | (3/4)·n + 2  →  (1/4)·n^2 | n - 3 | 0.333·n^-2 | read B[i4, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.331 | level | 2 | (15/64)·n^2 | 0.0781/n | read B[i1, i3] (i0=0, i1=0, i4=0); read B[i4, i3] (i0=0) (+1) |
| n^2 | 0.313 | ramp | (51/8)·n - 9  →  (5/16)·n^2 + (-1/4)·n + 5 | (7/8)·n - 15 | 0.292·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.313 | ramp | (51/8)·n - 10  →  (5/16)·n^2 + (-1/4)·n + 4 | (7/8)·n - 15 | 0.292·n^-2 | read C[i1, i3] (i0=0, i4=0) |
| n^2 | 0.293 | ramp | (3/4)·n + 2  →  (1/4)·n^2 + (1/8)·n + 1 | (7/8)·n - 1 | 0.292·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.293 | ramp | (3/4)·n + 1  →  (1/4)·n^2 + (1/8)·n | (7/8)·n - 1 | 0.292·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + 2 | (7/8)·n - 2 | 0.292·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.251 | ramp | n + 2  →  (1/4)·n^2 + (1/8)·n + 1 | (3/4)·n - 1 | 0.25·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.251 | ramp | n + 2  →  (1/4)·n^2 + (1/8)·n + 1 | (3/4)·n - 1 | 0.25·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.251 | ramp | (1/2)·n + 4  →  (1/4)·n^2 + 2 | (3/4)·n | 0.25·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.251 | ramp | (1/2)·n + 3  →  (1/4)·n^2 + 1 | (3/4)·n | 0.25·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.247 | ramp | (39/8)·n - 9  →  (1/4)·n^2 + (-15/8)·n + 24 | (3/4)·n - 12 | 0.25·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.247 | ramp | (39/8)·n - 10  →  (1/4)·n^2 + (-15/8)·n + 23 | (3/4)·n - 12 | 0.25·n^-2 | read C[i1, i3] (i0=0, i4=0) |
| n^2 | 0.217 | level | 3 | (1/8)·n^2 | 0.0417/n | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^2 | 0.189 | level | 3 | (7/64)·n^2 + (-7/8)·n | 0.0365/n | read B[i4, i3] (i0=0) |
| n^2 | 0.188 | level | 1 | (3/16)·n^2 + (1/2)·n | 0.0625/n | read C[i4, i3] (i0=0, i1=0, i4=0); write A[i1, i2] (i0=0) (+1) |
| n^2 | 0.14 | level | (5/16)·n^2 + (1/2)·n | (1/4)·n - 4 | 0.0833·n^-2 | read C[i1, i3] (i0=0, i4=0); read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.0824 | ramp | (5/2)·n + 4  →  (1/4)·n^2 + (-11/8)·n + 2 | (1/4)·n - 2 | 0.0833·n^-2 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^2 | 0.0815 | ramp | (35/8)·n - 3  →  (1/4)·n^2 + (-15/8)·n + 3 | (1/4)·n - 4 | 0.0833·n^-2 | read B[i4, i3] (i0=0, i1=0, i4=0); read C[i1, i3] (i0=0, i1=0, i4=0) |
| n^2 | 0.0699 | level | (5/16)·n^2 + (1/4)·n + 2 | (1/8)·n - 2 | 0.0417·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.0664 | ramp | (5/16)·n^2 + (1/4)·n + 4  →  (5/16)·n^2 + (1/2)·n - 2 | (1/8)·n - 2 | 0.0417·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.0664 | ramp | (5/16)·n^2 + (1/4)·n + 3  →  (5/16)·n^2 + (1/2)·n - 3 | (1/8)·n - 2 | 0.0417·n^-2 | read C[i1, i3] (i0=0, i4=0) |
| n^2 | 0.0439 | ramp | (65/8)·n - 2  →  (5/16)·n^2 + (-5/2)·n + 18 | (1/8)·n - 3 | 0.0417·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.0439 | ramp | (65/8)·n - 3  →  (5/16)·n^2 + (-5/2)·n + 17 | (1/8)·n - 3 | 0.0417·n^-2 | read C[i1, i3] (i0=0, i4=0) |
| n^2 | 0.0414 | ramp | (11/4)·n + 3  →  (1/4)·n^2 + (-9/8)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.0414 | ramp | (11/4)·n + 3  →  (1/4)·n^2 + (-9/8)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 6  →  (1/4)·n^2 + (-5/4)·n + 2 | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-11/8)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-11/8)·n | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 6  →  (1/4)·n^2 + (-3/2)·n + 2 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 5  →  (1/4)·n^2 + (-3/2)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0409 | ramp | 2·n + 5  →  (1/4)·n^2 + (-7/4)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0409 | ramp | 2·n + 4  →  (1/4)·n^2 + (-7/4)·n | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0408 | ramp | (37/8)·n - 6  →  (1/4)·n^2 + (-15/8)·n + 6 | (1/8)·n - 2 | 0.0417·n^-2 | read B[i1, i3] (i0=0, i1=1, i4=0) |
| n^2 | 0.0408 | ramp | (37/8)·n - 7  →  (1/4)·n^2 + (-15/8)·n + 5 | (1/8)·n - 2 | 0.0417·n^-2 | read C[i1, i3] (i0=0, i1=1, i4=0) |
| n^2 | 0.0271 | level | 3 | (1/64)·n^2 + (-1/8)·n | 0.00521/n | read B[i4, i3] (i0=0) |
| n^1.5 | 0.238 | ramp | 2  →  (1/8)·n | n - 8 | 0.333·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^1.5 | 0.0941 | ramp | (1/2)·n + 5  →  (3/4)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i1=2, i4=0) |
| n^1.5 | 0.0939 | ramp | (1/2)·n + 4  →  (3/4)·n - 2 | (1/8)·n - 2 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i1=2, i4=0) |
| n^1.5 | 0.0729 | ramp | (1/4)·n + 5  →  (1/2)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i1=1, i4=0) |
| n^1.5 | 0.0723 | ramp | (1/4)·n + 3  →  (1/2)·n - 3 | (1/8)·n - 2 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 13.7 | level | 5 | (49/8)·n | 2.04·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^1 | 12.2 | level | 4 | (49/8)·n | 2.04·n^-2 | read C[i1, i3] (i0=0, i4=0) |
| n^1 | 8 | level | 1 | 8·n | 2.67·n^-2 | read A[i1, i2] (i0=0, i1=0); read A[i1, i4] (i0=0, i3=0, i4=0) (+1) |
| n^1 | 3.91 | level | (5/16)·n^2 + (1/2)·n | 7 | 2.33·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.292·n^-2 | read C[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.292·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.292·n^-2 | read C[i1, i3] (i0=0, i1=0, i4=0) |
| n^1 | 1.12 | level | (5/16)·n^2 + (1/2)·n | 2 | 0.667·n^-3 | read C[i1, i3] (i0=0, i4=0); read B[i1, i3] (i0=0, i4=0) |
| n^1 | 1.12 | level | (5/16)·n^2 + (1/2)·n | 2 | 0.667·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0); read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 1 | level | (1/4)·n^2 + (1/8)·n + 1 | 2 | 0.667·n^-3 | read C[i1, i3] (i0=0, i1=0, i3=0, i4=0); read C[i1, i3] (i0=0, i1=1, i3=0, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/8)·n + 2 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/4)·n + 2 | 1 | 0.333·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/8)·n + 3 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n - 1 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/4)·n | 1 | 0.333·n^-3 | read A[i1, i2] (i0=0) |
| n^1 | 0.559 | level | (5/16)·n^2 + (1/2)·n | 1 | 0.333·n^-3 | read A[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 3 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 4 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 5 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 6 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 7 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 8 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 2 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i1=1, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 2 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 3 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 4 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 5 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 6 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 7 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.5 | level | (1/4)·n^2 + (1/8)·n + 1 | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=0, i3=0, i4=0) |
| n^0.5 | 3.08 | level | (19/8)·n - 1 | 2 | 0.667·n^-3 | read B[i4, i3] (i0=0, i1=0, i4=0); read C[i1, i3] (i0=0, i1=0, i4=0) |
| n^0.5 | 2.47 | level | (49/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.47 | level | (49/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.42 | level | (47/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.42 | level | (47/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.37 | level | (45/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.37 | level | (45/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.32 | level | (43/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.32 | level | (43/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.26 | level | (41/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.26 | level | (41/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.21 | level | (39/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.21 | level | (39/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.15 | level | (37/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.15 | level | (37/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.09 | level | (35/8)·n - 10 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.09 | level | (35/8)·n - 9 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 2.03 | level | (33/8)·n - 9 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i1=7, i4=0) |
| n^0.5 | 2.03 | level | (33/8)·n - 8 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i1=7, i4=0) |
| n^0.5 | 1.97 | level | (31/8)·n - 8 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.97 | level | (31/8)·n - 7 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.9 | level | (29/8)·n - 7 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.9 | level | (29/8)·n - 6 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.84 | level | (27/8)·n - 6 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.84 | level | (27/8)·n - 5 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.77 | level | (25/8)·n - 5 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.77 | level | (25/8)·n - 4 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.7 | level | (23/8)·n - 4 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.7 | level | (23/8)·n - 3 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.62 | level | (21/8)·n - 3 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 1.62 | level | (21/8)·n - 2 | 1 | 0.333·n^-3 | read B[i1, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=2, i4=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=2, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n - 1 | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 3 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=2, i3=0, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=0, i3=0, i4=0); read B[i4, i3] (i0=0, i1=2, i3=0, i4=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 3 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=1, i3=0, i4=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 1 | 1 | 0.333·n^-3 | read C[i1, i3] (i0=0, i1=0, i3=0, i4=0); read B[i4, i3] (i0=0, i1=1, i3=0, i4=0) |

Two rank-k sources (B and C) double syrk's structure: ramps to (1/4)n^2 + (1/4)n lines with combined 0.0466·n^4 (≈ 2.8× syrk, two matrices and a wider window), headroom +1.0.

## syr2k — single-shot  [`exact`]

Accesses $A(n) = 3·n^3 + 4·n^2 + n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0466·n^4  +  1.08·n^3.5  +  4.84·n^3  +  5.47·n^2.5  +  16.4·n^2  +  0.572·n^1.5  +  18.9·n^1  +  5.56·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0409 | ramp | n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^3 + (-77/32)·n^2 + (45/4)·n - 12 | 0.0365 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^4 | 0.00572 | ramp | (11/4)·n + 8  →  (1/4)·n^2 + (1/4)·n - 2 | (1/64)·n^3 + (-15/32)·n^2 + (17/4)·n - 12 | 0.00521 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3.5 | 0.885 | ramp | 9  →  (17/8)·n | (49/64)·n^3 + (-49/16)·n^2 + (7/4)·n | 0.255 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3.5 | 0.124 | ramp | 24  →  (17/8)·n | (7/64)·n^3 + (-21/16)·n^2 + (7/2)·n | 0.0365 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3.5 | 0.0583 | ramp | 23  →  (17/8)·n - 16 | (7/128)·n^3 + (-91/64)·n^2 + (35/4)·n | 0.0182 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00814 | ramp | 25  →  (17/8)·n - 14 | (1/128)·n^3 + (-17/64)·n^2 + (23/8)·n - 10 | 0.0026 | read A[i1, i4] (i0=0) |
| n^3 | 3.07 | level | 5 | (11/8)·n^3 + (-17/8)·n^2 - n | 0.458 | read C[i4, i3] (i0=0, i1=1, i3=0, i4=0); read C[i1, i3] (i0=0) (+2) |
| n^3 | 0.438 | level | 1 | (7/16)·n^3 | 0.146 | read A[i1, i4] (i0=0) |
| n^3 | 0.393 | ramp | n + 4  →  (1/4)·n^2 + 2 | n^2 - 7·n + 12 | 0.333/n | read B[i4, i3] (i0=0, i3=0); read C[i4, i3] (i0=0, i3=0) |
| n^3 | 0.306 | level | 6 | (1/8)·n^3 + (-5/4)·n^2 + 2·n | 0.0417 | read B[i4, i3] (i0=0, i1=2, i3=0, i4=0); read C[i1, i3] (i0=0) (+1) |
| n^3 | 0.148 | ramp | (5/4)·n + 2  →  (1/4)·n^2 + (1/4)·n | (3/8)·n^2 + (-15/8)·n + 1 | 0.125/n | read C[i4, i3] (i0=0) |
| n^3 | 0.148 | ramp | (5/4)·n + 2  →  (1/4)·n^2 + (1/4)·n | (3/8)·n^2 + (-15/8)·n + 1 | 0.125/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0485 | ramp | (11/4)·n + 4  →  (1/4)·n^2 + (1/4)·n | (1/8)·n^2 + (-3/2)·n + 4 | 0.0417/n | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3 | 0.0483 | ramp | 3·n + 3  →  (1/4)·n^2 + (1/4)·n - 1 | (1/8)·n^2 + (-7/4)·n + 6 | 0.0417/n | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^3 | 0.0396 | ramp | (3/4)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (1/8)·n^2 + (-19/8)·n + 6 | 0.0417/n | read C[i4, i3] (i0=0, i4=0) |
| n^3 | 0.0396 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + (1/4)·n - 4 | (1/8)·n^2 + (-19/8)·n + 6 | 0.0417/n | read B[i4, i3] (i0=0, i4=0) |
| n^3 | 0.0347 | ramp | (1/2)·n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0365/n | read C[i4, i3] (i0=0) |
| n^3 | 0.0347 | ramp | (1/2)·n + 5  →  (1/4)·n^2 + (1/4)·n - 3 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0365/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0347 | ramp | (3/4)·n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^2 - 2·n + 4 | 0.0365/n | read C[i4, i3] (i0=0) |
| n^3 | 0.0347 | ramp | (3/4)·n + 6  →  (1/4)·n^2 + (1/4)·n - 2 | (7/64)·n^2 - 2·n + 4 | 0.0365/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0049 | ramp | (5/2)·n + 8  →  (1/4)·n^2 - n - 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read C[i4, i3] (i0=0) |
| n^3 | 0.0049 | ramp | (5/2)·n + 8  →  (1/4)·n^2 - n - 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read B[i4, i3] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 8  →  (1/4)·n^2 + (-5/4)·n - 2 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read C[i4, i3] (i0=0) |
| n^3 | 0.00488 | ramp | (9/4)·n + 7  →  (1/4)·n^2 + (-5/4)·n - 3 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read B[i4, i3] (i0=0) |
| n^2.5 | 1.71 | ramp | 7  →  (17/8)·n | (7/4)·n^2 + (-7/2)·n | 0.583/n | read B[i4, i3] (i0=0, i4=0); read C[i4, i3] (i0=0, i4=0) |
| n^2.5 | 0.958 | ramp | 20  →  (17/8)·n - 14 | n^2 - 8·n | 0.333/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.937 | ramp | 6  →  (17/8)·n - 16 | n^2 - 10·n + 9 | 0.333/n | read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.747 | ramp | 7  →  (17/8)·n | (49/64)·n^2 + (-7/8)·n | 0.255/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.747 | ramp | 7  →  (17/8)·n | (49/64)·n^2 + (-7/8)·n | 0.255/n | read C[i4, i3] (i0=0) |
| n^2.5 | 0.149 | ramp | 21  →  (17/8)·n - 30 | (11/64)·n^2 + (-13/4)·n + 8 | 0.0573/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.105 | ramp | 22  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/8)·n | 0.0365/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.105 | ramp | 22  →  (17/8)·n - 12 | (7/64)·n^2 + (-7/8)·n | 0.0365/n | read C[i4, i3] (i0=0) |
| n^2.5 | 0.0141 | ramp | 23  →  (17/8)·n - 28 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00521/n | read A[i1, i4] (i0=0) |
| n^2 | 2.65 | level | 2 | (15/8)·n^2 | 0.625/n | read B[i4, i3] (i0=0); read B[i1, i3] (i0=0) |
| n^2 | 2.14 | level | 6 | (7/8)·n^2 - 7·n | 0.292/n | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 1.96 | level | 5 | (7/8)·n^2 + (-7/8)·n | 0.292/n | read C[i1, i3] (i0=0, i4=0); read B[i1, i3] (i0=0, i4=0) |
| n^2 | 1.95 | level | 3 | (9/8)·n^2 - n | 0.375/n | read B[i4, i3] (i0=0, i1=1, i3=0, i4=0); read B[i4, i3] (i0=0) (+1) |
| n^2 | 1.75 | level | 4 | (7/8)·n^2 + n - 1 | 0.292/n | read A[i1, i4] (i0=0, i1=8, i4=0); read C[i1, i3] (i0=0) |
| n^2 | 1.5 | level | 1 | (3/2)·n^2 + (17/2)·n | 0.5/n | write A[i1, i2] (i0=0); read A[i1, i4] (i0=0, i3=0, i4=0) (+2) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 | 0.146/n | read A[i1, i2] (i0=0) |
| n^2 | 0.334 | ramp | n + 1  →  (1/4)·n^2 + (1/4)·n - 1 | n - 3 | 0.333·n^-2 | read C[i4, i3] (i0=0, i4=0) |
| n^2 | 0.334 | ramp | n  →  (1/4)·n^2 + (1/4)·n - 2 | n - 3 | 0.333·n^-2 | read B[i4, i3] (i0=0, i4=0) |
| n^2 | 0.333 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + 2 | n - 3 | 0.333·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.333 | ramp | (3/4)·n + 3  →  (1/4)·n^2 + 1 | n - 3 | 0.333·n^-2 | read C[i4, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.333 | ramp | (3/4)·n + 2  →  (1/4)·n^2 | n - 3 | 0.333·n^-2 | read B[i4, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.293 | ramp | (3/4)·n + 2  →  (1/4)·n^2 + (1/8)·n + 1 | (7/8)·n - 1 | 0.292·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.293 | ramp | (3/4)·n + 1  →  (1/4)·n^2 + (1/8)·n | (7/8)·n - 1 | 0.292·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.292 | ramp | (3/4)·n + 4  →  (1/4)·n^2 + 2 | (7/8)·n - 2 | 0.292·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.251 | ramp | n + 2  →  (1/4)·n^2 + (1/8)·n + 1 | (3/4)·n - 1 | 0.25·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.251 | ramp | n + 2  →  (1/4)·n^2 + (1/8)·n + 1 | (3/4)·n - 1 | 0.25·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.251 | ramp | (1/2)·n + 4  →  (1/4)·n^2 + 2 | (3/4)·n | 0.25·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.251 | ramp | (1/2)·n + 3  →  (1/4)·n^2 + 1 | (3/4)·n | 0.25·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0824 | ramp | (5/2)·n + 4  →  (1/4)·n^2 + (-11/8)·n + 2 | (1/4)·n - 2 | 0.0833·n^-2 | read B[i4, i3] (i0=0); read C[i4, i3] (i0=0) |
| n^2 | 0.0414 | ramp | (11/4)·n + 3  →  (1/4)·n^2 + (-9/8)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.0414 | ramp | (11/4)·n + 3  →  (1/4)·n^2 + (-9/8)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 6  →  (1/4)·n^2 + (-5/4)·n + 2 | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 3  →  (1/4)·n^2 + (-11/8)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0) |
| n^2 | 0.0412 | ramp | (5/2)·n + 2  →  (1/4)·n^2 + (-11/8)·n | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 6  →  (1/4)·n^2 + (-3/2)·n + 2 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0411 | ramp | (9/4)·n + 5  →  (1/4)·n^2 + (-3/2)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0409 | ramp | 2·n + 5  →  (1/4)·n^2 + (-7/4)·n + 1 | (1/8)·n - 1 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0409 | ramp | 2·n + 4  →  (1/4)·n^2 + (-7/4)·n | (1/8)·n - 1 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^1.5 | 0.238 | ramp | 2  →  (1/8)·n | n - 8 | 0.333·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^1.5 | 0.0941 | ramp | (1/2)·n + 5  →  (3/4)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i1=2, i4=0) |
| n^1.5 | 0.0939 | ramp | (1/2)·n + 4  →  (3/4)·n - 2 | (1/8)·n - 2 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i1=2, i4=0) |
| n^1.5 | 0.0729 | ramp | (1/4)·n + 5  →  (1/2)·n - 1 | (1/8)·n - 2 | 0.0417·n^-2 | read C[i4, i3] (i0=0, i1=1, i4=0) |
| n^1.5 | 0.0723 | ramp | (1/4)·n + 3  →  (1/2)·n - 3 | (1/8)·n - 2 | 0.0417·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 12.2 | level | 4 | (49/8)·n | 2.04·n^-2 | read C[i1, i3] (i0=0, i4=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.292·n^-2 | read C[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 1.96 | level | 5 | (7/8)·n | 0.292·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.292·n^-2 | read C[i1, i3] (i0=0, i1=0, i4=0) |
| n^1 | 1.24 | level | 2 | (7/8)·n | 0.292·n^-2 | read B[i4, i3] (i0=0, i1=0, i4=0) |
| n^0.5 | 0.866 | level | (3/4)·n + 1 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=2, i4=0) |
| n^0.5 | 0.866 | level | (3/4)·n | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=2, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 1 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n - 1 | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 3 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=2, i3=0, i4=0) |
| n^0.5 | 0.707 | level | (1/2)·n + 2 | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=2, i3=0, i4=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 3 | 1 | 0.333·n^-3 | read C[i4, i3] (i0=0, i1=1, i3=0, i4=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 1 | 1 | 0.333·n^-3 | read B[i4, i3] (i0=0, i1=1, i3=0, i4=0) |

Two rank-k sources (B and C) double syrk's structure: ramps to (1/4)n^2 + (1/4)n lines with combined 0.0466·n^4 (≈ 2.8× syrk, two matrices and a wider window), headroom +1.0.

## syrk — infinite-repeat  [`exact`]

Accesses $A(n) = 2·n^3 + 3·n^2 + n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0166·n^4  +  0.416·n^3.5  +  2.59·n^3  +  2.74·n^2.5  +  7.9·n^2  +  0.291·n^1.5  +  27.4·n^1  +  24.3·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0146 | ramp | (3/8)·n + 4  →  (1/8)·n^2 + (1/4)·n - 1 | (7/128)·n^3 + (-35/32)·n^2 + (29/8)·n - 2 | 0.0273 | read B[i4, i3] (i0=0) |
| n^4 | 0.00204 | ramp | (5/4)·n + 6  →  (1/8)·n^2 + (1/4)·n - 1 | (1/128)·n^3 + (-7/32)·n^2 + (7/4)·n - 4 | 0.00391 | read B[i4, i3] (i0=0) |
| n^3.5 | 0.322 | ramp | 5  →  (9/8)·n | (49/128)·n^3 + (-49/32)·n^2 + (7/8)·n | 0.191 | read B[i4, i3] (i0=0) |
| n^3.5 | 0.045 | ramp | 13  →  (9/8)·n | (7/128)·n^3 + (-21/32)·n^2 + (7/4)·n | 0.0273 | read B[i4, i3] (i0=0) |
| n^3.5 | 0.0425 | ramp | 13  →  (9/8)·n - 8 | (7/128)·n^3 + (-91/64)·n^2 + (35/4)·n | 0.0273 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00593 | ramp | 14  →  (9/8)·n - 7 | (1/128)·n^3 + (-17/64)·n^2 + (23/8)·n - 10 | 0.00391 | read A[i1, i4] (i0=0) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + n - 1 | 0.219 | read A[i1, i4] (i0=0, i1=8, i4=0); read B[i1, i3] (i0=0) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (-7/8)·n^2 | 0.219 | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^3 | 0.438 | level | 1 | (7/16)·n^3 | 0.219 | read A[i1, i4] (i0=0) |
| n^3 | 0.14 | ramp | (3/8)·n + 3  →  (1/8)·n^2 + (1/8)·n + 1 | (1/2)·n^2 + (-5/2)·n + 3 | 0.25/n | read B[i4, i3] (i0=0, i3=0) |
| n^3 | 0.125 | level | 4 | (1/16)·n^3 + (-1/2)·n^2 | 0.0312 | read B[i4, i3] (i0=0, i1=1, i3=0, i4=0); read B[i1, i3] (i0=0) |
| n^3 | 0.108 | level | 3 | (1/16)·n^3 + (3/8)·n^2 | 0.0312 | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^3 | 0.106 | ramp | (1/2)·n + 2  →  (1/8)·n^2 + (1/4)·n | (3/8)·n^2 + (-9/8)·n | 0.188/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0381 | ramp | (17/4)·n - 7  →  (3/16)·n^2 + (1/2)·n - 1 | (1/8)·n^2 + (-33/8)·n + 34 | 0.0625/n | read B[i1, i3] (i0=0, i4=0) |
| n^3 | 0.028 | ramp | (1/4)·n + 4  →  (1/8)·n^2 + (1/4)·n - 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0625/n | read B[i4, i3] (i0=0, i4=0) |
| n^3 | 0.0271 | level | (3/16)·n^2 + (1/2)·n | (1/16)·n^2 + (-13/8)·n + 10 | 0.0312/n | read A[i1, i2] (i0=0) |
| n^3 | 0.0246 | ramp | (1/4)·n + 4  →  (1/8)·n^2 + (1/4)·n - 1 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0547/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0173 | ramp | (5/4)·n + 4  →  (1/8)·n^2 + (1/4)·n | (1/16)·n^2 + (-5/8)·n + 1 | 0.0312/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0172 | ramp | (11/8)·n + 3  →  (1/8)·n^2 + (1/4)·n - 1 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0312/n | read B[i4, i3] (i0=0) |
| n^3 | 0.00346 | ramp | (9/8)·n + 6  →  (1/8)·n^2 + (-1/2)·n - 1 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00781/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.698 | ramp | 11  →  (9/8)·n - 7 | n^2 - 8·n | 0.5/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.683 | ramp | 4  →  (9/8)·n - 8 | n^2 - 10·n + 9 | 0.5/n | read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.621 | ramp | 4  →  (9/8)·n | (7/8)·n^2 + (-7/4)·n | 0.438/n | read B[i4, i3] (i0=0, i4=0) |
| n^2.5 | 0.544 | ramp | 4  →  (9/8)·n | (49/64)·n^2 + (-7/8)·n | 0.383/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.11 | ramp | 12  →  (9/8)·n - 15 | (11/64)·n^2 + (-13/4)·n + 8 | 0.0859/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.0766 | ramp | 12  →  (9/8)·n - 6 | (7/64)·n^2 + (-7/8)·n | 0.0547/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.0103 | ramp | 13  →  (9/8)·n - 14 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00781/n | read A[i1, i4] (i0=0) |
| n^2 | 1.75 | level | 1 | (7/4)·n^2 | 0.875/n | read A[i1, i2] (i0=0); write A[i1, i2] (i0=0) (+1) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 - 7·n | 0.438/n | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 1.24 | level | 2 | (7/8)·n^2 | 0.438/n | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^2 | 0.433 | level | (3/16)·n^2 + (1/2)·n | n - 9 | 0.5·n^-2 | read A[i1, i2] (i0=0, i2=0) |
| n^2 | 0.433 | level | (3/16)·n^2 + (1/2)·n | n - 9 | 0.5·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.372 | ramp | (1/8)·n^2 + (1/8)·n + 10  →  (3/16)·n^2 + (1/2)·n | n - 9 | 0.5·n^-2 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.237 | ramp | (13/4)·n + 8  →  (3/16)·n^2 + (1/4)·n + 2 | (7/8)·n - 14 | 0.438·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.237 | ramp | (13/4)·n - 8  →  (1/8)·n^2 + (-7/8)·n + 40 | n - 16 | 0.5·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.237 | ramp | (3/8)·n + 2  →  (1/8)·n^2 + (1/4)·n | n - 2 | 0.5·n^-2 | read B[i4, i3] (i0=0, i4=0) |
| n^2 | 0.236 | ramp | (1/4)·n + 3  →  (1/8)·n^2 + (1/8)·n + 1 | n - 2 | 0.5·n^-2 | read B[i4, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.207 | ramp | (3/8)·n + 2  →  (1/8)·n^2 + (1/8)·n + 1 | (7/8)·n - 1 | 0.438·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.204 | ramp | (19/8)·n - 2  →  (1/8)·n^2 + (-7/8)·n + 16 | (7/8)·n - 14 | 0.438·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.188 | level | 1 | (3/16)·n^2 + (1/2)·n | 0.0938/n | read B[i4, i3] (i0=0, i1=0, i4=0); write A[i1, i2] (i0=0) (+1) |
| n^2 | 0.178 | ramp | (1/4)·n + 3  →  (1/8)·n^2 + (1/8)·n + 1 | (3/4)·n | 0.375·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.177 | level | 2 | (1/8)·n^2 | 0.0625/n | write A[i1, i4] (i0=0, i3=0); write A[i1, i4] (i0=0) |
| n^2 | 0.0541 | level | (3/16)·n^2 + (1/2)·n | (1/8)·n - 2 | 0.0625·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.0541 | level | (3/16)·n^2 + (3/8)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i1, i2] (i0=0) |
| n^2 | 0.033 | ramp | (33/8)·n + 22  →  (3/16)·n^2 + (-3/2)·n + 10 | (1/8)·n - 3 | 0.0625·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 0.0292 | ramp | (5/4)·n + 3  →  (1/8)·n^2 + (-5/8)·n + 1 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.0291 | ramp | (9/8)·n + 5  →  (1/8)·n^2 + (-5/8)·n + 1 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.029 | ramp | n + 4  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 1 | 0.0625·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0289 | ramp | (9/4)·n - 1  →  (1/8)·n^2 + (-7/8)·n + 2 | (1/8)·n - 2 | 0.0625·n^-2 | read B[i1, i3] (i0=0, i1=0, i4=0) |
| n^1.5 | 0.238 | ramp | 2  →  (1/8)·n | n - 8 | 0.5·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^1.5 | 0.0522 | ramp | (1/8)·n + 4  →  (1/4)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 9.9 | level | 2 | 7·n | 3.5·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^1 | 8 | level | 1 | 8·n | 4·n^-2 | read A[i1, i2] (i0=0, i1=0); read A[i1, i4] (i0=0, i3=0, i4=0) (+1) |
| n^1 | 3.03 | level | (3/16)·n^2 + (1/2)·n | 7 | 3.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.438·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (1/2)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (3/8)·n + 1 | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (3/8)·n | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (1/2)·n | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (1/2)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 2 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 3 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 4 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 5 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 6 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 7 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 8 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i3=0, i4=0) |
| n^1 | 0.354 | level | (1/8)·n^2 + (1/8)·n + 1 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i1=0, i3=0, i4=0) |
| n^0.5 | 1.77 | level | (25/8)·n + 7 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.73 | level | 3·n + 6 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.7 | level | (23/8)·n + 5 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.66 | level | (11/4)·n + 4 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.62 | level | (21/8)·n + 3 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.58 | level | (5/2)·n + 2 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.54 | level | (19/8)·n + 1 | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.5 | level | (9/4)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.46 | level | (17/8)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i1=7, i4=0) |
| n^0.5 | 1.41 | level | 2·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.37 | level | (15/8)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.32 | level | (7/4)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.27 | level | (13/8)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.22 | level | (3/2)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.17 | level | (11/8)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i4=0) |
| n^0.5 | 1.12 | level | (5/4)·n | 1 | 0.5·n^-3 | read B[i1, i3] (i0=0, i1=0, i4=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 2 | 1 | 0.5·n^-3 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.354 | level | (1/8)·n + 3 | 1 | 0.5·n^-3 | read B[i4, i3] (i0=0, i1=1, i3=0, i4=0) |

The rank-k source `read B[i4,i3]` is re-read for every output row: ramp families from (3/8)n + 4 up to (1/8)n^2 + (1/4)n lines (the growing triangular window) with population n^3/16, giving 0.0166·n^4 — headroom +1.0. The old headroom-0 reading was an artifact: the ramp values collapsed to constants under the rendering bug and the scale approximation could not see the whole-matrix distances. The n^3.5 terms are row-window reuses (up to (9/8)n lines) of both A and B. The cache cliff sits at 64·((1/8)n^2) bytes, a constant factor past gemm's — matching the measured 0.04% → 1.9% jump at N≈560 in the empirical section.

## syrk — single-shot  [`exact`]

Accesses $A(n) = 2·n^3 + 3·n^2 + n$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0166·n^4  +  0.416·n^3.5  +  2.52·n^3  +  2.74·n^2.5  +  5.81·n^2  +  0.291·n^1.5  +  11.4·n^1  +  0.854·n^0.5

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0146 | ramp | (3/8)·n + 4  →  (1/8)·n^2 + (1/4)·n - 1 | (7/128)·n^3 + (-35/32)·n^2 + (29/8)·n - 2 | 0.0273 | read B[i4, i3] (i0=0) |
| n^4 | 0.00204 | ramp | (5/4)·n + 6  →  (1/8)·n^2 + (1/4)·n - 1 | (1/128)·n^3 + (-7/32)·n^2 + (7/4)·n - 4 | 0.00391 | read B[i4, i3] (i0=0) |
| n^3.5 | 0.322 | ramp | 5  →  (9/8)·n | (49/128)·n^3 + (-49/32)·n^2 + (7/8)·n | 0.191 | read B[i4, i3] (i0=0) |
| n^3.5 | 0.045 | ramp | 13  →  (9/8)·n | (7/128)·n^3 + (-21/32)·n^2 + (7/4)·n | 0.0273 | read B[i4, i3] (i0=0) |
| n^3.5 | 0.0425 | ramp | 13  →  (9/8)·n - 8 | (7/128)·n^3 + (-91/64)·n^2 + (35/4)·n | 0.0273 | read A[i1, i4] (i0=0) |
| n^3.5 | 0.00593 | ramp | 14  →  (9/8)·n - 7 | (1/128)·n^3 + (-17/64)·n^2 + (23/8)·n - 10 | 0.00391 | read A[i1, i4] (i0=0) |
| n^3 | 1.62 | level | 3 | (15/16)·n^3 + (-1/2)·n^2 + n - 1 | 0.469 | read A[i1, i4] (i0=0, i1=8, i4=0); read B[i1, i3] (i0=0) (+1) |
| n^3 | 0.438 | level | 1 | (7/16)·n^3 | 0.219 | read A[i1, i4] (i0=0) |
| n^3 | 0.14 | ramp | (3/8)·n + 3  →  (1/8)·n^2 + (1/8)·n + 1 | (1/2)·n^2 + (-5/2)·n + 3 | 0.25/n | read B[i4, i3] (i0=0, i3=0) |
| n^3 | 0.125 | level | 4 | (1/16)·n^3 + (-1/2)·n^2 | 0.0312 | read B[i4, i3] (i0=0, i1=1, i3=0, i4=0); read B[i1, i3] (i0=0) |
| n^3 | 0.106 | ramp | (1/2)·n + 2  →  (1/8)·n^2 + (1/4)·n | (3/8)·n^2 + (-9/8)·n | 0.188/n | read B[i4, i3] (i0=0) |
| n^3 | 0.028 | ramp | (1/4)·n + 4  →  (1/8)·n^2 + (1/4)·n - 1 | (1/8)·n^2 + (-9/4)·n + 4 | 0.0625/n | read B[i4, i3] (i0=0, i4=0) |
| n^3 | 0.0246 | ramp | (1/4)·n + 4  →  (1/8)·n^2 + (1/4)·n - 1 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0547/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0173 | ramp | (5/4)·n + 4  →  (1/8)·n^2 + (1/4)·n | (1/16)·n^2 + (-5/8)·n + 1 | 0.0312/n | read B[i4, i3] (i0=0) |
| n^3 | 0.0172 | ramp | (11/8)·n + 3  →  (1/8)·n^2 + (1/4)·n - 1 | (1/16)·n^2 + (-3/4)·n + 2 | 0.0312/n | read B[i4, i3] (i0=0) |
| n^3 | 0.00346 | ramp | (9/8)·n + 6  →  (1/8)·n^2 + (-1/2)·n - 1 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00781/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.698 | ramp | 11  →  (9/8)·n - 7 | n^2 - 8·n | 0.5/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.683 | ramp | 4  →  (9/8)·n - 8 | n^2 - 10·n + 9 | 0.5/n | read A[i1, i4] (i0=0, i4=0) |
| n^2.5 | 0.621 | ramp | 4  →  (9/8)·n | (7/8)·n^2 + (-7/4)·n | 0.438/n | read B[i4, i3] (i0=0, i4=0) |
| n^2.5 | 0.544 | ramp | 4  →  (9/8)·n | (49/64)·n^2 + (-7/8)·n | 0.383/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.11 | ramp | 12  →  (9/8)·n - 15 | (11/64)·n^2 + (-13/4)·n + 8 | 0.0859/n | read A[i1, i4] (i0=0, i3=0); read A[i1, i4] (i0=0) |
| n^2.5 | 0.0766 | ramp | 12  →  (9/8)·n - 6 | (7/64)·n^2 + (-7/8)·n | 0.0547/n | read B[i4, i3] (i0=0) |
| n^2.5 | 0.0103 | ramp | 13  →  (9/8)·n - 14 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00781/n | read A[i1, i4] (i0=0) |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 - 7·n | 0.438/n | read B[i1, i3] (i0=0, i4=0) |
| n^2 | 1.5 | level | 1 | (3/2)·n^2 + (17/2)·n | 0.75/n | write A[i1, i2] (i0=0); read A[i1, i4] (i0=0, i3=0, i4=0) (+2) |
| n^2 | 1.41 | level | 2 | n^2 | 0.5/n | write A[i1, i4] (i0=0) |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 | 0.219/n | read A[i1, i2] (i0=0) |
| n^2 | 0.237 | ramp | (3/8)·n + 2  →  (1/8)·n^2 + (1/4)·n | n - 2 | 0.5·n^-2 | read B[i4, i3] (i0=0, i4=0) |
| n^2 | 0.236 | ramp | (1/4)·n + 3  →  (1/8)·n^2 + (1/8)·n + 1 | n - 2 | 0.5·n^-2 | read B[i4, i3] (i0=0, i3=0, i4=0) |
| n^2 | 0.207 | ramp | (3/8)·n + 2  →  (1/8)·n^2 + (1/8)·n + 1 | (7/8)·n - 1 | 0.438·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.178 | ramp | (1/4)·n + 3  →  (1/8)·n^2 + (1/8)·n + 1 | (3/4)·n | 0.375·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.0292 | ramp | (5/4)·n + 3  →  (1/8)·n^2 + (-5/8)·n + 1 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i4, i3] (i0=0) |
| n^2 | 0.0291 | ramp | (9/8)·n + 5  →  (1/8)·n^2 + (-5/8)·n + 1 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^2 | 0.029 | ramp | n + 4  →  (1/8)·n^2 + (-3/4)·n | (1/8)·n - 1 | 0.0625·n^-2 | read B[i4, i3] (i0=0, i3=0) |
| n^1.5 | 0.238 | ramp | 2  →  (1/8)·n | n - 8 | 0.5·n^-2 | read A[i1, i4] (i0=0, i3=0, i4=0) |
| n^1.5 | 0.0522 | ramp | (1/8)·n + 4  →  (1/4)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^1 | 9.9 | level | 2 | 7·n | 3.5·n^-2 | read B[i1, i3] (i0=0, i4=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.438·n^-2 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 2 | 1 | 0.5·n^-3 | read B[i4, i3] (i0=0, i1=1, i4=0) |
| n^0.5 | 0.354 | level | (1/8)·n + 3 | 1 | 0.5·n^-3 | read B[i4, i3] (i0=0, i1=1, i3=0, i4=0) |

The rank-k source `read B[i4,i3]` is re-read for every output row: ramp families from (3/8)n + 4 up to (1/8)n^2 + (1/4)n lines (the growing triangular window) with population n^3/16, giving 0.0166·n^4 — headroom +1.0. The old headroom-0 reading was an artifact: the ramp values collapsed to constants under the rendering bug and the scale approximation could not see the whole-matrix distances. The n^3.5 terms are row-window reuses (up to (9/8)n lines) of both A and B. The cache cliff sits at 64·((1/8)n^2) bytes, a constant factor past gemm's — matching the measured 0.04% → 1.9% jump at N≈560 in the empirical section.

## trisolve — infinite-repeat  [`exact`]

Accesses $A(n) = 2·n^2 + 3·n$ (exact on n ≡ 0 mod 8); DMD order $n^{3}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0156·n^3  +  0.0226·n^2.5  +  3.52·n^2  +  1.02·n^1.5  +  38.7·n^1  +  1·n^0.5  +  50.8·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^3 | 0.0137 | level | (1/16)·n^2 + (3/4)·n | (7/128)·n^2 + (-23/16)·n + 9 | 0.0273 | read C[i0, i1] |
| n^3 | 0.00195 | level | (1/16)·n^2 + (3/4)·n | (1/128)·n^2 + (-5/16)·n + 3 | 0.00391 | read C[i0, i1] |
| n^2.5 | 0.0171 | ramp | 9  →  (1/4)·n + 1 | (3/64)·n^2 + (-15/8)·n + 18 | 0.0234 | read B[i1] |
| n^2.5 | 0.00285 | ramp | 9  →  (1/4)·n + 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.00391 | read B[i1] |
| n^2.5 | 0.00269 | ramp | 10  →  (1/4)·n | (1/128)·n^2 + (-7/16)·n + 6 | 0.00391 | read B[i1] |
| n^2 | 1.52 | level | 3 | (7/8)·n^2 + (-105/8)·n + 49 | 0.438 | read C[i0, i1]; read B[i1] |
| n^2 | 0.568 | level | 3 | (21/64)·n^2 + (-21/4)·n + 21 | 0.164 | write B[i0] |
| n^2 | 0.438 | level | 1 | (7/16)·n^2 + (7/8)·n - 7 | 0.219 | read B[i0] (i1=0); read B[i0] |
| n^2 | 0.219 | level | (1/16)·n^2 + (3/4)·n | (7/8)·n - 8 | 0.438/n | read C[i0, i1] |
| n^2 | 0.219 | level | (1/16)·n^2 + (3/4)·n | (7/8)·n - 8 | 0.438/n | read C[i0, i1] (i1=0) |
| n^2 | 0.0947 | level | 3 | (7/128)·n^2 + (-21/16)·n + 7 | 0.0273 | write B[i0] |
| n^2 | 0.0947 | level | 3 | (7/128)·n^2 + (-5/16)·n - 1 | 0.0273 | write B[i0] |
| n^2 | 0.0812 | level | 3 | (3/64)·n^2 + (-3/8)·n | 0.0234 | write B[i0] |
| n^2 | 0.0625 | level | 1 | (1/16)·n^2 + (-3/8)·n - 1 | 0.0312 | read B[i0] (i1=0); read B[i0] |
| n^2 | 0.0312 | level | (1/16)·n^2 + (3/4)·n + (35/16) | (1/8)·n + (-17/8) | 0.0625/n | read C[i0, i1] |
| n^2 | 0.0312 | level | (1/16)·n^2 + (3/4)·n | (1/8)·n - 2 | 0.0625/n | read C[i0, i1] |
| n^2 | 0.0312 | level | (1/16)·n^2 + (3/4)·n | (1/8)·n - 3 | 0.0625/n | read C[i0, i1] |
| n^2 | 0.0312 | level | (1/16)·n^2 + (3/4)·n | (1/8)·n - 2 | 0.0625/n | read C[i0, i1] (i1=0) |
| n^2 | 0.0312 | level | (1/16)·n^2 + (3/4)·n | (1/8)·n - 2 | 0.0625/n | read C[i0, i0] |
| n^2 | 0.0291 | ramp | (1/16)·n^2 + (-1/8)·n + 7  →  (1/16)·n^2 + (3/4)·n - 21 | (1/8)·n - 3 | 0.0625/n | read A[i0] |
| n^2 | 0.0271 | level | 3 | (1/64)·n^2 + (-3/8)·n + 5 | 0.00781 | read A[i0] (i0=0); read C[i0, i0] (i0=0) (+3) |
| n^2 | 0.0146 | ramp | (1/4)·n + 8  →  (1/16)·n^2 + (-9/4)·n + 24 | (1/8)·n - 3 | 0.0625/n | write B[i0] |
| n^1.5 | 0.257 | ramp | 5  →  (1/4)·n + 1 | (3/4)·n - 6 | 0.375/n | read A[i0] |
| n^1.5 | 0.254 | ramp | 7  →  (1/4)·n + 1 | (3/4)·n - 12 | 0.375/n | read B[i1] (i1=0) |
| n^1.5 | 0.254 | ramp | 7  →  (1/4)·n + 1 | (3/4)·n - 12 | 0.375/n | read B[i1] |
| n^1.5 | 0.0428 | ramp | 5  →  (1/4)·n + 1 | (1/8)·n - 1 | 0.0625/n | read A[i0] |
| n^1.5 | 0.0424 | ramp | 7  →  (1/4)·n + 1 | (1/8)·n - 2 | 0.0625/n | read B[i1] (i1=0) |
| n^1.5 | 0.0424 | ramp | 7  →  (1/4)·n + 1 | (1/8)·n - 2 | 0.0625/n | read B[i1] |
| n^1.5 | 0.0418 | ramp | 6  →  (1/4)·n | (1/8)·n - 2 | 0.0625/n | read B[i1] |
| n^1.5 | 0.0414 | ramp | 8  →  (1/4)·n | (1/8)·n - 3 | 0.0625/n | read B[i1] |
| n^1.5 | 0.0414 | ramp | 8  →  (1/4)·n | (1/8)·n - 3 | 0.0625/n | read B[i1] (i1=0) |
| n^1 | 10.6 | level | 3 | (49/8)·n - 49 | 3.06/n | read C[i0, i1]; read B[i1] |
| n^1 | 7.42 | level | 2 | (21/4)·n | 2.62/n | read C[i0, i1]; read B[i1] |
| n^1 | 2.47 | level | 2 | (7/4)·n | 0.875/n | write B[i0] |
| n^1 | 2.17 | level | 3 | (5/4)·n - 10 | 0.625/n | write B[i0] |
| n^1 | 1.41 | level | 2 | n - 1 | 0.5/n | read C[i0, i0]; write B[i0] |
| n^1 | 1.38 | level | 1 | (11/8)·n | 0.688/n | write B[i0] |
| n^1 | 1.3 | level | 3 | (3/4)·n - 6 | 0.375/n | write B[i0] |
| n^1 | 1.25 | level | (1/16)·n^2 + (3/4)·n | 5 | 2.5·n^-2 | read C[i0, i1] (i1=0) |
| n^1 | 1.25 | level | 1 | (5/4)·n | 0.625/n | write B[i0] |
| n^1 | 1.08 | level | 3 | (5/8)·n - 5 | 0.312/n | write B[i0] |
| n^1 | 1.06 | level | 2 | (3/4)·n - 6 | 0.375/n | read B[i1] |
| n^1 | 0.5 | level | (1/16)·n^2 + (3/4)·n | 2 | 1·n^-2 | read C[i0, i1] (i0=7, i1=0); read C[i0, i1] (i0=8, i1=0) |
| n^1 | 0.5 | level | 1 | (1/2)·n | 0.25/n | write B[i0] |
| n^1 | 0.433 | level | 3 | (1/4)·n - 2 | 0.125/n | write B[i0] |
| n^1 | 0.433 | level | 3 | (1/4)·n - 2 | 0.125/n | write B[i0] |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n | 1 | 0.5·n^-2 | read C[i0, i1] |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n | 1 | 0.5·n^-2 | read C[i0, i1] |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (35/16) | 1 | 0.5·n^-2 | read C[i0, i1] |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (35/16) | 1 | 0.5·n^-2 | read C[i0, i1] (i1=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n | 1 | 0.5·n^-2 | read C[i0, i1] (i1=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n | 1 | 0.5·n^-2 | read C[i0, i1] (i0=1, i1=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n | 1 | 0.5·n^-2 | read C[i0, i0] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (35/16) | 1 | 0.5·n^-2 | read A[i0] |
| n^1 | 0.25 | level | (1/16)·n^2 + (-1/8)·n | 1 | 0.5·n^-2 | read A[i0] |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 14 | 1 | 0.5·n^-2 | read A[i0] (i0=8) |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n | 1 | 0.5·n^-2 | read C[i0, i0] |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n + (35/16) | 1 | 0.5·n^-2 | read C[i0, i0] |
| n^1 | 0.25 | level | (1/16)·n^2 + (-5/4)·n + 8 | 1 | 0.5·n^-2 | write B[i0] |
| n^1 | 0.25 | level | (1/16)·n^2 + (-1/4)·n | 1 | 0.5·n^-2 | write B[i0] |
| n^1 | 0.25 | level | (1/16)·n^2 + (3/4)·n - 7 | 1 | 0.5·n^-2 | read A[i0] (i0=0) |
| n^1 | 0.25 | level | (1/16)·n^2 + (-3/8)·n + (37/16) | 1 | 0.5·n^-2 | write B[i0] |
| n^1 | 0.217 | level | 3 | (1/8)·n - 2 | 0.0625/n | write B[i0] (i1=0) |
| n^1 | 0.217 | level | 3 | (1/8)·n - 1 | 0.0625/n | write B[i0] |
| n^1 | 0.217 | level | 3 | (1/8)·n - 1 | 0.0625/n | write B[i0] |
| n^1 | 0.217 | level | 3 | (1/8)·n - 2 | 0.0625/n | write B[i0] |
| n^1 | 0.177 | level | 2 | (1/8)·n - 1 | 0.0625/n | read B[i1] |
| n^1 | 0.125 | level | 1 | (1/8)·n | 0.0625/n | write B[i0] |
| n^1 | 0.125 | level | 1 | (1/8)·n | 0.0625/n | write B[i0] |
| n^1 | 0.125 | level | 1 | (1/8)·n - 1 | 0.0625/n | write B[i0] |
| n^0.5 | 0.5 | level | (1/4)·n + (7/4) | 1 | 0.5·n^-2 | write B[i0] (i0=0) |
| n^0.5 | 0.5 | level | (1/4)·n | 1 | 0.5·n^-2 | write B[i0] (i0=0) |
| n^0 | 13.4 | level | 5 | 6 | 3·n^-2 | read B[i1] (i1=0) |
| n^0 | 11.3 | level | 2 | 8 | 4·n^-2 | write B[i0] (i0=0); read B[i1] (i0=1, i1=0) (+1) |
| n^0 | 10.4 | level | 3 | 6 | 3·n^-2 | read A[i0] |
| n^0 | 9 | level | 1 | 9 | 4.5·n^-2 | read B[i0] (i0=0); read B[i0] (i0=1, i1=0) (+2) |
| n^0 | 2.45 | level | 6 | 1 | 0.5·n^-2 | read B[i1] (i0=16, i1=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.5·n^-2 | read B[i1] (i0=9, i1=0) |
| n^0 | 2 | level | 4 | 1 | 0.5·n^-2 | write B[i0] (i0=0); read C[i0, i1] (i0=1, i1=0) (+1) |

The triangular matrix returns across passes: levels at distance (1/16)n^2 + (3/4)n — the triangular footprint in lines — with population ~n^2/16, giving the n^3 order (0.0156). trisolve is the cleanest case of a kernel whose only asymptotic locality lives across invocations.

## trisolve — single-shot  [`exact`]

Accesses $A(n) = 2·n^2 + 3·n$ (exact on n ≡ 0 mod 8); DMD order $n^{2.5}$, headroom **+0.5**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0226·n^2.5  +  2.88·n^2  +  1.02·n^1.5  +  34.5·n^1  +  40.7·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^2.5 | 0.0171 | ramp | 9  →  (1/4)·n + 1 | (3/64)·n^2 + (-15/8)·n + 18 | 0.0234 | read B[i1] |
| n^2.5 | 0.00285 | ramp | 9  →  (1/4)·n + 1 | (1/128)·n^2 + (-5/16)·n + 3 | 0.00391 | read B[i1] |
| n^2.5 | 0.00269 | ramp | 10  →  (1/4)·n | (1/128)·n^2 + (-7/16)·n + 6 | 0.00391 | read B[i1] |
| n^2 | 2.27 | level | 3 | (21/16)·n^2 - 21·n + 84 | 0.656 | read C[i0, i1]; read B[i1] (+1) |
| n^2 | 0.5 | level | 1 | (1/2)·n^2 + (1/2)·n | 0.25 | read B[i0] (i0=0); read B[i0] (i1=0) (+1) |
| n^2 | 0.108 | level | 3 | (1/16)·n^2 + (-5/8)·n + 1 | 0.0312 | write B[i0] |
| n^1.5 | 0.257 | ramp | 5  →  (1/4)·n + 1 | (3/4)·n - 6 | 0.375/n | read A[i0] |
| n^1.5 | 0.254 | ramp | 7  →  (1/4)·n + 1 | (3/4)·n - 12 | 0.375/n | read B[i1] (i1=0) |
| n^1.5 | 0.254 | ramp | 7  →  (1/4)·n + 1 | (3/4)·n - 12 | 0.375/n | read B[i1] |
| n^1.5 | 0.0428 | ramp | 5  →  (1/4)·n + 1 | (1/8)·n - 1 | 0.0625/n | read A[i0] |
| n^1.5 | 0.0424 | ramp | 7  →  (1/4)·n + 1 | (1/8)·n - 2 | 0.0625/n | read B[i1] (i1=0) |
| n^1.5 | 0.0424 | ramp | 7  →  (1/4)·n + 1 | (1/8)·n - 2 | 0.0625/n | read B[i1] |
| n^1.5 | 0.0418 | ramp | 6  →  (1/4)·n | (1/8)·n - 2 | 0.0625/n | read B[i1] |
| n^1.5 | 0.0414 | ramp | 8  →  (1/4)·n | (1/8)·n - 3 | 0.0625/n | read B[i1] |
| n^1.5 | 0.0414 | ramp | 8  →  (1/4)·n | (1/8)·n - 3 | 0.0625/n | read B[i1] (i1=0) |
| n^1 | 6.06 | level | 3 | (7/2)·n - 28 | 1.75/n | read C[i0, i1] |
| n^1 | 4.55 | level | 3 | (21/8)·n - 21 | 1.31/n | read B[i1] |
| n^1 | 4.55 | level | 3 | (21/8)·n - 21 | 1.31/n | write B[i0] |
| n^1 | 3.71 | level | 2 | (21/8)·n | 1.31/n | read C[i0, i1] |
| n^1 | 3.71 | level | 2 | (21/8)·n | 1.31/n | read B[i1] |
| n^1 | 2.62 | level | 1 | (21/8)·n | 1.31/n | write B[i0] |
| n^1 | 2.47 | level | 2 | (7/4)·n | 0.875/n | write B[i0]; read C[i0, i0] |
| n^1 | 1.52 | level | 3 | (7/8)·n - 7 | 0.438/n | write B[i0] |
| n^1 | 1.52 | level | 3 | (7/8)·n - 7 | 0.438/n | write B[i0] |
| n^1 | 1.41 | level | 2 | n + 1 | 0.5/n | read B[i1] (i0=7, i1=0); write B[i0] |
| n^1 | 1.06 | level | 2 | (3/4)·n - 6 | 0.375/n | read B[i1] |
| n^1 | 0.75 | level | 1 | (3/4)·n | 0.375/n | write B[i0] |
| n^1 | 0.217 | level | 3 | (1/8)·n - 1 | 0.0625/n | write B[i0] |
| n^1 | 0.177 | level | 2 | (1/8)·n - 1 | 0.0625/n | read B[i1] |
| n^1 | 0.125 | level | 1 | (1/8)·n | 0.0625/n | write B[i0] |
| n^0 | 13.4 | level | 5 | 6 | 3·n^-2 | read B[i1] (i1=0) |
| n^0 | 10.4 | level | 3 | 6 | 3·n^-2 | read A[i0] |
| n^0 | 8.49 | level | 2 | 6 | 3·n^-2 | read B[i1] (i1=0) |
| n^0 | 2.45 | level | 6 | 1 | 0.5·n^-2 | read B[i1] (i0=16, i1=0) |
| n^0 | 2.24 | level | 5 | 1 | 0.5·n^-2 | read B[i1] (i0=9, i1=0) |
| n^0 | 2 | level | 4 | 1 | 0.5·n^-2 | read B[i1] (i0=8, i1=0) |
| n^0 | 1.73 | level | 3 | 1 | 0.5·n^-2 | read A[i0] (i0=1) |

The solution vector `read B[i1]` is re-read against a growing prefix: ramp families from ~9 lines up to (1/4)n + 1 (the vector footprint), d = 2.5, headroom +0.5. The matrix C is streamed once (triangular row by row) and contributes only line-reuse levels at distance 1–3.

## trmm — infinite-repeat  [`exact`]

Accesses $A(n) = 2·n^3$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0169·n^4  +  1.05·n^3.5  +  1.74·n^3  +  2.67·n^2.5  +  3.68·n^2  +  0.119·n^1.5  +  16.8·n^1  +  5.54·n^0.5  +  2·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0148 | ramp | (3/8)·n + 5  →  (1/8)·n^2 + n - 2 | (7/128)·n^3 + (-63/64)·n^2 + (13/8)·n + 2 | 0.0273 | read A[i3, i2] (i0=0, i3=0); read A[i3, i2] (i0=0) |
| n^4 | 0.00207 | ramp | n + 17  →  (1/8)·n^2 + (9/8)·n - 18 | (1/128)·n^3 + (-13/64)·n^2 + (11/8)·n - 2 | 0.00391 | read A[i3, i2] (i0=0, i3=0); read A[i3, i2] (i0=0) |
| n^3.5 | 0.494 | ramp | 5  →  2·n - 1 | (7/16)·n^3 + (-7/16)·n^2 + (-7/8)·n | 0.219 | read B[i3, i1] (i0=0, i3=0); read B[i3, i1] (i0=0) |
| n^3.5 | 0.487 | ramp | 5  →  2·n - 3 | (7/16)·n^3 + (-35/16)·n^2 + (21/8)·n | 0.219 | read A[i3, i2] (i0=0, i3=0); read A[i3, i2] (i0=0) |
| n^3.5 | 0.0684 | ramp | 6  →  2·n | (1/16)·n^3 + (-11/16)·n^2 + (13/8)·n - 1 | 0.0312 | read B[i3, i1] (i0=0, i3=0); read B[i3, i1] (i0=0) |
| n^3 | 0.758 | level | 3 | (7/16)·n^3 + (-7/16)·n^2 | 0.219 | read A[i3, i2] (i0=0, i1=0, i3=0); write A[i1, i2] (i0=0) |
| n^3 | 0.438 | level | 1 | (7/16)·n^3 + (7/16)·n^2 + (7/8)·n | 0.219 | read A[i1, i2] (i0=0, i3=0); read A[i1, i2] (i0=0) (+1) |
| n^3 | 0.125 | ramp | (1/4)·n + 2  →  (1/8)·n^2 + (7/8)·n - 1 | (7/16)·n^2 + (1/8)·n - 3 | 0.219/n | read A[i1, i2] (i0=0, i3=0); read A[i3, i2] (i0=0, i3=0) (+1) |
| n^3 | 0.125 | ramp | (1/2)·n + 3  →  (1/8)·n^2 + n - 1 | (7/16)·n^2 + (-7/8)·n - 1 | 0.219/n | read A[i3, i2] (i0=0, i2=0, i3=0); read A[i3, i2] (i0=0, i2=0) |
| n^3 | 0.108 | level | 3 | (1/16)·n^3 + (-1/16)·n^2 + (7/8)·n | 0.0312 | read A[i3, i2] (i0=0, i3=0); write A[i1, i2] (i0=0, i3=0) (+1) |
| n^3 | 0.0625 | level | 1 | (1/16)·n^3 + (-1/16)·n^2 + (1/8)·n | 0.0312 | read A[i1, i2] (i0=0); write A[i1, i2] (i0=0) |
| n^3 | 0.0329 | ramp | (27/8)·n - 5  →  (3/16)·n^2 + (-5/8)·n - 7 | (7/64)·n^2 + (-29/8)·n + 30 | 0.0547/n | read A[i3, i2] (i0=0, i1=0) |
| n^3 | 0.0271 | level | (3/16)·n^2 + (3/8)·n | (1/16)·n^2 + (-3/2)·n + 9 | 0.0312/n | read B[i3, i1] (i0=0, i2=0, i3=0); read B[i3, i1] (i0=0, i2=0, i3=7) (+1) |
| n^3 | 0.0248 | ramp | (1/4)·n + 4  →  (1/8)·n^2 + n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0547/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.0172 | ramp | (9/8)·n + 10  →  (1/8)·n^2 + (9/8)·n - 18 | (1/16)·n^2 + (-7/8)·n + 3 | 0.0312/n | read A[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0172 | ramp | n + 9  →  (1/8)·n^2 + n - 19 | (1/16)·n^2 + (-7/8)·n + 3 | 0.0312/n | read A[i3, i2] (i0=0) |
| n^3 | 0.0046 | ramp | (33/8)·n - 4  →  (3/16)·n^2 + (-5/8)·n - 21 | (1/64)·n^2 + (-5/8)·n + 6 | 0.00781/n | read A[i3, i2] (i0=0, i1=0) |
| n^3 | 0.00352 | ramp | n + 17  →  (1/8)·n^2 + (9/8)·n - 18 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00781/n | read A[i1, i2] (i0=0, i3=0) |
| n^2.5 | 1.24 | level | 2·n - 1 | (7/8)·n^2 + (-7/4)·n | 0.438/n | read A[i3, i2] (i0=0, i1=0, i3=0); read A[i3, i2] (i0=0, i1=0) |
| n^2.5 | 0.825 | ramp | 5  →  2·n - 1 | (7/8)·n^2 + (-7/4)·n | 0.438/n | read A[i3, i2] (i0=0, i1=0); read A[i3, i2] (i0=0) |
| n^2.5 | 0.49 | ramp | 6  →  2·n - 2 | (7/16)·n^2 + (-7/4)·n + 1 | 0.219/n | read A[i1, i2] (i0=0, i2=0, i3=0); read B[i3, i1] (i0=0, i2=0, i3=0) (+1) |
| n^2.5 | 0.118 | ramp | 6  →  2·n | (1/8)·n^2 + (-3/8)·n | 0.0625/n | read B[i3, i1] (i0=0, i2=0); read B[i3, i1] (i0=0) |
| n^2 | 1 | level | 1 | n^2 - n | 0.5/n | read A[i1, i2] (i0=0, i1=0, i2=0); read A[i1, i2] (i0=0) |
| n^2 | 0.433 | level | (3/16)·n^2 + (-5/8)·n + 9 | n - 9 | 0.5·n^-2 | read B[i3, i1] (i0=0, i1=0, i2=0) |
| n^2 | 0.375 | ramp | (1/8)·n^2 + n  →  (3/16)·n^2 + (3/8)·n | n - 9 | 0.5·n^-2 | read A[i3, i2] (i0=0, i1=0) |
| n^2 | 0.359 | ramp | (3/16)·n^2 + (-11/8)·n + 22  →  (3/16)·n^2 + (3/8)·n - 8 | (7/8)·n - 14 | 0.438·n^-2 | read A[i3, i2] (i0=0, i1=0) |
| n^2 | 0.325 | level | (3/16)·n^2 + (3/8)·n | (3/4)·n - 7 | 0.375·n^-2 | read B[i3, i1] (i0=0, i2=0) |
| n^2 | 0.257 | ramp | (3/16)·n^2 + (-3/8)·n + 12  →  (3/16)·n^2 + (3/8)·n - 2 | (5/8)·n - 10 | 0.312·n^-2 | read A[i3, i2] (i0=0, i1=0) |
| n^2 | 0.239 | ramp | (3/8)·n + 2  →  (1/8)·n^2 + n - 1 | n - 2 | 0.5·n^-2 | read A[i1, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.226 | ramp | (19/8)·n - 4  →  (3/16)·n^2 + (-29/8)·n + 42 | (7/8)·n - 15 | 0.438·n^-2 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^2 | 0.0541 | level | (3/16)·n^2 + (3/8)·n | (1/8)·n - 2 | 0.0625·n^-2 | read B[i3, i1] (i0=0, i2=0) |
| n^2 | 0.0541 | level | (3/16)·n^2 + (3/8)·n | (1/8)·n - 2 | 0.0625·n^-2 | read A[i1, i2] (i0=0, i1=0, i3=0) |
| n^2 | 0.0515 | ramp | (3/16)·n^2 + (1/4)·n + 2  →  (3/16)·n^2 + (3/8)·n - 1 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i1=0, i3=0) |
| n^2 | 0.0513 | ramp | (3/16)·n^2 + (-1/2)·n + 14  →  (3/16)·n^2 + (3/8)·n - 7 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i1=0, i3=6) |
| n^2 | 0.0511 | ramp | (3/16)·n^2 + (-3/2)·n + 23  →  (3/16)·n^2 + (3/8)·n - 22 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i1=0, i3=14) |
| n^2 | 0.0317 | ramp | (25/8)·n - 3  →  (3/16)·n^2 + (-9/2)·n + 49 | (1/8)·n - 3 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^2 | 0.0297 | ramp | (9/8)·n + 15  →  (1/8)·n^2 + (9/8)·n - 17 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0297 | ramp | (13/4)·n - 4  →  (1/8)·n^2 + 2 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i1=0) |
| n^2 | 0.0296 | ramp | (25/8)·n - 3  →  (1/8)·n^2 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i1=0) |
| n^2 | 0.0296 | ramp | n + 15  →  (1/8)·n^2 + n - 17 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0296 | ramp | n + 14  →  (1/8)·n^2 + n - 18 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0294 | ramp | (9/8)·n + 9  →  (1/8)·n^2 + (1/8)·n - 7 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^1.5 | 0.067 | ramp | (1/4)·n + 4  →  (3/8)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^1.5 | 0.0518 | ramp | (1/8)·n + 3  →  (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | read A[i1, i2] (i0=0, i1=1); read A[i1, i2] (i0=0) |
| n^1 | 2.17 | level | (3/16)·n^2 + (3/8)·n | 5 | 2.5·n^-3 | read A[i3, i2] (i0=0, i1=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.438·n^-2 | read B[i3, i1] (i0=0, i3=0) |
| n^1 | 0.866 | level | (3/16)·n^2 + (3/8)·n | 2 | 1·n^-3 | read A[i1, i2] (i0=0, i1=0, i3=0); read A[i3, i2] (i0=0, i1=0, i3=0) |
| n^1 | 0.707 | level | (1/8)·n^2 + n - 1 | 2 | 1·n^-3 | read A[i3, i2] (i0=0, i1=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (3/8)·n | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (9/8)·n | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i2=0, i3=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-5/2)·n + 24 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0, i3=14) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-19/8)·n + 23 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-9/4)·n + 22 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-17/8)·n + 21 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 - 2·n + 20 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-15/8)·n + 19 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-7/4)·n + 18 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-13/8)·n + 17 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-1/2)·n + 7 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0, i3=6) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-3/8)·n + 6 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-1/4)·n + 5 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-1/8)·n + 4 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + 3 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (1/8)·n + 2 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (1/4)·n + 1 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0, i3=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-5/8)·n + 9 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0, i3=7) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-1/2)·n + 8 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-3/8)·n + 7 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-1/4)·n + 6 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (-1/8)·n + 5 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + 4 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (1/8)·n + 3 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (1/4)·n + 2 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i1=0, i2=0, i3=0) |
| n^1 | 0.433 | level | (3/16)·n^2 + (3/8)·n | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0, i1=0, i2=0, i3=0) |
| n^1 | 0.25 | level | 4 | (1/8)·n - 1 | 0.0625·n^-2 | read B[i3, i1] (i0=0, i3=0) |
| n^0.5 | 1.5 | level | (9/4)·n - 3 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^0.5 | 1.46 | level | (17/8)·n - 2 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i1=0, i2=0) |
| n^0.5 | 0.612 | level | (3/8)·n + (21/4) | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 2 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 1 | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0, i1=0, i2=0, i3=0); read A[i1, i2] (i0=0, i1=1, i2=0) (+1) |
| n^0.5 | 0.354 | level | (1/8)·n + 1 | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0, i1=1); read A[i1, i2] (i0=0) |
| n^0 | 2 | level | 4 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i2=0, i3=0) |

Triangular multiply: `read A[i3,i2]` ramps from (3/8)n + 5 to (1/8)n^2 + n lines (population n^3/16, 0.0148·n^4). The wide n^3.5 band (0.49 + 0.49) is the B-panel and A-row reuse ramping to 2n lines — trmm reaches its matrix boundary with a much larger runner-up coefficient than syrk, which is why its n^4 term anchors earlier.

## trmm — single-shot  [`exact`]

Accesses $A(n) = 2·n^3$ (exact on n ≡ 0 mod 8); DMD order $n^{4}$, headroom **+1**; conservation Σmass/warm = 1 at n=256, 1 at n=264.

**DMD spectrum:**  0.0169·n^4  +  1.05·n^3.5  +  1.68·n^3  +  1.31·n^2.5  +  1.23·n^2  +  0.943·n^1.5  +  1.77·n^1  +  2.58·n^0.5  +  2·n^0

| order | coeff | kind | distance (lines) | population (accesses) | portion | source access |
|---|---|---|---|---|---|---|
| n^4 | 0.0148 | ramp | (3/8)·n + 5  →  (1/8)·n^2 + n - 2 | (7/128)·n^3 + (-63/64)·n^2 + (13/8)·n + 2 | 0.0273 | read A[i3, i2] (i0=0, i3=0); read A[i3, i2] (i0=0) |
| n^4 | 0.00207 | ramp | n + 17  →  (1/8)·n^2 + (9/8)·n - 18 | (1/128)·n^3 + (-13/64)·n^2 + (11/8)·n - 2 | 0.00391 | read A[i3, i2] (i0=0, i3=0); read A[i3, i2] (i0=0) |
| n^3.5 | 0.494 | ramp | 5  →  2·n - 1 | (7/16)·n^3 + (-7/16)·n^2 + (-7/8)·n | 0.219 | read B[i3, i1] (i0=0, i3=0); read B[i3, i1] (i0=0) |
| n^3.5 | 0.491 | ramp | 5  →  2·n - 1 | (7/16)·n^3 + (-21/16)·n^2 + (7/8)·n | 0.219 | read A[i3, i2] (i0=0, i3=0); read A[i3, i2] (i0=0) |
| n^3.5 | 0.0688 | ramp | 6  →  2·n | (1/16)·n^3 + (-9/16)·n^2 + (3/8)·n + 1 | 0.0312 | read A[i3, i2] (i0=0, i2=0, i3=0); read A[i3, i2] (i0=0, i2=0) (+2) |
| n^3 | 0.866 | level | 3 | (1/2)·n^3 + (-1/2)·n^2 + (7/8)·n | 0.25 | read B[i3, i1] (i0=0, i3=0); write A[i1, i2] (i0=0) |
| n^3 | 0.5 | level | 1 | (1/2)·n^3 + (1/2)·n^2 | 0.25 | read A[i1, i2] (i0=0); write A[i1, i2] (i0=0) |
| n^3 | 0.125 | ramp | (1/4)·n + 2  →  (1/8)·n^2 + (7/8)·n - 1 | (7/16)·n^2 + (1/8)·n - 3 | 0.219/n | read A[i1, i2] (i0=0, i3=0); read A[i3, i2] (i0=0, i3=0) (+1) |
| n^3 | 0.125 | ramp | (1/2)·n + 3  →  (1/8)·n^2 + n - 1 | (7/16)·n^2 + (-7/8)·n - 1 | 0.219/n | read A[i3, i2] (i0=0, i2=0, i3=0); read A[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0248 | ramp | (1/4)·n + 4  →  (1/8)·n^2 + n - 2 | (7/64)·n^2 + (-15/8)·n + 2 | 0.0547/n | read A[i1, i2] (i0=0, i3=0) |
| n^3 | 0.0172 | ramp | (9/8)·n + 10  →  (1/8)·n^2 + (9/8)·n - 18 | (1/16)·n^2 + (-7/8)·n + 3 | 0.0312/n | read A[i3, i2] (i0=0, i2=0) |
| n^3 | 0.0172 | ramp | n + 9  →  (1/8)·n^2 + n - 19 | (1/16)·n^2 + (-7/8)·n + 3 | 0.0312/n | read A[i3, i2] (i0=0) |
| n^3 | 0.00352 | ramp | n + 17  →  (1/8)·n^2 + (9/8)·n - 18 | (1/64)·n^2 + (-3/8)·n + 2 | 0.00781/n | read A[i1, i2] (i0=0, i3=0) |
| n^2.5 | 0.825 | ramp | 5  →  2·n - 1 | (7/8)·n^2 + (-7/4)·n | 0.438/n | read A[i3, i2] (i0=0) |
| n^2.5 | 0.49 | ramp | 6  →  2·n - 2 | (7/16)·n^2 + (-7/4)·n + 1 | 0.219/n | read B[i3, i1] (i0=0, i2=0, i3=0); read B[i3, i1] (i0=0, i2=0) |
| n^2 | 0.875 | level | 1 | (7/8)·n^2 | 0.438/n | read A[i1, i2] (i0=0, i3=0); read A[i1, i2] (i0=0) |
| n^2 | 0.239 | ramp | (3/8)·n + 2  →  (1/8)·n^2 + n - 1 | n - 2 | 0.5·n^-2 | read A[i1, i2] (i0=0, i2=0, i3=0) |
| n^2 | 0.0297 | ramp | (9/8)·n + 15  →  (1/8)·n^2 + (9/8)·n - 17 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i2=0) |
| n^2 | 0.0296 | ramp | n + 15  →  (1/8)·n^2 + n - 17 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i1, i2] (i0=0, i3=0) |
| n^2 | 0.0296 | ramp | n + 14  →  (1/8)·n^2 + n - 18 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^2 | 0.0294 | ramp | (9/8)·n + 9  →  (1/8)·n^2 + (1/8)·n - 7 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^1.5 | 0.824 | ramp | 6  →  2·n - 2 | (7/8)·n - 2 | 0.438·n^-2 | read B[i3, i1] (i0=0, i2=0) |
| n^1.5 | 0.067 | ramp | (1/4)·n + 4  →  (3/8)·n + 1 | (1/8)·n - 2 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^1.5 | 0.0518 | ramp | (1/8)·n + 3  →  (1/4)·n | (1/8)·n - 2 | 0.0625·n^-2 | read A[i1, i2] (i0=0) |
| n^1 | 1.52 | level | 3 | (7/8)·n | 0.438·n^-2 | read A[i3, i2] (i0=0, i3=0) |
| n^1 | 0.25 | level | 4 | (1/8)·n - 1 | 0.0625·n^-2 | read A[i3, i2] (i0=0, i2=0, i3=0); read B[i3, i1] (i0=0, i3=0) |
| n^0.5 | 0.612 | level | (3/8)·n + (21/4) | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^0.5 | 0.612 | level | (3/8)·n + 2 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i2=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 2 | 1 | 0.5·n^-3 | read A[i3, i2] (i0=0, i3=0) |
| n^0.5 | 0.5 | level | (1/4)·n + 1 | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0, i2=0) |
| n^0.5 | 0.354 | level | (1/8)·n + 1 | 1 | 0.5·n^-3 | read A[i1, i2] (i0=0) |
| n^0 | 2 | level | 4 | 1 | 0.5·n^-3 | read B[i3, i1] (i0=0, i2=0, i3=0) |

Triangular multiply: `read A[i3,i2]` ramps from (3/8)n + 5 to (1/8)n^2 + n lines (population n^3/16, 0.0148·n^4). The wide n^3.5 band (0.49 + 0.49) is the B-panel and A-row reuse ramping to 2n lines — trmm reaches its matrix boundary with a much larger runner-up coefficient than syrk, which is why its n^4 term anchors earlier.
