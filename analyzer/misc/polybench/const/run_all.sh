#!/usr/bin/env bash
set -x
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
    testnames="$testnames $testname"
  fi
done

echo "All testnames: $testnames"
for testname in $testnames; do
  echo "Running test: $testname"
  $PROGRAM --json -i "$SCRIPT_DIR/const_$testname.mlir" -o "$SCRIPT_DIR/const_$testname.json" barvinok --block-size=8 --infinite-repeat \
    --barvinok-arg='--approximation-method=scale' -m /dev/null
done
