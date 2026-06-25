#!/bin/bash
# lower.sh <in.mlir> <out.o>  -- MLIR affine -> native object (llvm-22)
B=/usr/lib/llvm-22/bin
$B/mlir-opt "$1" -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm \
  -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts 2>"$2.err" \
  | $B/mlir-translate --mlir-to-llvmir 2>>"$2.err" \
  | $B/llc -O3 -filetype=obj -o "$2" 2>>"$2.err"
