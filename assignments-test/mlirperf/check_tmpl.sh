#!/bin/bash
# Usage: ./check.sh your.mlir    (correctness only; does NOT benchmark)
set -e
HERE="$(cd "$(dirname "$0")"&&pwd)"; B=/usr/lib/llvm-22/bin
build(){ "$HERE/../../lower.sh" "$1" "$2.o" && clang -O3 -march=native "$HERE/driver.c" "$2.o" -o "$2"; }
build "$HERE/ref.mlir" /tmp/_ref || { echo "REF build failed"; exit 1; }
build "$1" /tmp/_opt || { echo "OPT build failed (see /tmp/_opt.o.err)"; cat /tmp/_opt.o.err; exit 1; }
for N in 96 130; do /tmp/_ref $N /tmp/_r.bin; /tmp/_opt $N /tmp/_o.bin
 python3 -c "import numpy as np;a=np.fromfile('/tmp/_r.bin');b=np.fromfile('/tmp/_o.bin');import sys;print('N=$N allclose',np.allclose(a,b,rtol=1e-6,atol=1e-9))"; done
