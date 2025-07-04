#!/usr/bin/env bash
set -x
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROGRAM=$(realpath "$SCRIPT_DIR/../../../../target/release/mrc")

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
echo "Program,Predicted Miss Count,Prediction Time (ms)"> "$SCRIPT_DIR/results.csv"
for testname in $testnames; do
  echo "Running test: $testname"
  $PROGRAM --input "$SCRIPT_DIR/const_$testname.json" -c 32768 -b 64 >> "$SCRIPT_DIR/results.csv" 
done
