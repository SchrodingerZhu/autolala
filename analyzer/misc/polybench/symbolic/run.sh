#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROGRAM=$(realpath "$SCRIPT_DIR/../../../../target/release/analyzer")

echo "Program path: $PROGRAM"

# 组合 testname 列表
testnames=""
while read -r line; do
  name=$(echo "$line" | awk '{print $1}')
  if [ -z "$testnames" ]; then
    testnames="$name"
  else
    testnames="$testnames,$name"
  fi
done < "$SCRIPT_DIR/command.txt"

echo "All testnames: $testnames"

hyperfine -r 1 --export-markdown "$SCRIPT_DIR/benchmark.md" \
  --export-csv "$SCRIPT_DIR/benchmark.csv" \
  --export-json "$SCRIPT_DIR/benchmark.json" \
  -L testname "$testnames" \
  -L SCRIPT_DIR "$SCRIPT_DIR" \
  -L PROGRAM "$PROGRAM" \
  -u microsecond \
  "bash -c '
    set -euo pipefail
    line=\$(grep \"^{testname} \" \"{SCRIPT_DIR}/command.txt\")
    params=\$(echo \"\$line\" | cut -d\" \" -f2-)

    symbols=\"\"
    for p in \$params; do
      symbols=\"\$symbols --symbol-lowerbound=\$p\"
    done

    {PROGRAM} -i {SCRIPT_DIR}/sym_{testname}.mlir barvinok --block-size=8 \$symbols --infinite-repeat -m /dev/null
  '"
