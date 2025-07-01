#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROGRAM=$(realpath "$SCRIPT_DIR/../../../../target/release/analyzer")

echo "Program path: $PROGRAM"

testnames=""
for i in "$SCRIPT_DIR"/const_*.mlir; do
  basename=$(basename "$i" .mlir)
  testname="${basename#const_}"
  if [ -z "$testnames" ]; then
    testnames="$testname"
  else
    testnames="$testnames,$testname"
  fi
done
# All testnames: 3mm,atax,bicg,cholesky,convolution,correlation,covariance,doitgen,floyd_warshall,gemm,gemver,gesummv,gramschmidt,lu,mvt,seidel-2d,symm,syr2k,syrk,trisolve,trmm
echo "All testnames: $testnames"
approxmethod="=drop,=scale,=volume,=bernoulli,"
hyperfine --warmup 3 --export-markdown "$SCRIPT_DIR/benchmark.md" \
  --export-csv "$SCRIPT_DIR/benchmark.csv" \
  --export-json "$SCRIPT_DIR/benchmark.json" \
  -L testname "$testnames" \
  -L approxmethod "$approxmethod" \
  -u microsecond \
  "$PROGRAM -i $SCRIPT_DIR/const_{testname}.mlir barvinok --block-size=8 --barvinok-arg='--approximation-method{approxmethod}' -m /dev/null"
