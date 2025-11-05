#!/usr/bin/env bash
CACHE_LIMIT=524288
mkdir -p results
export RUST_LOG=warn
SCRIPT_DIR=$(dirname "$0")
for file in analyzer/misc/polybench/polygeist/constant/*.c; do
    echo "Running $file"
    # echo "$file, fully" >> results/runtimes.txt
    # /usr/bin/time -f "%E" -a -o results/runtimes.txt bash $SCRIPT_DIR/run-single-simulation.sh "$file" -C$CACHE_LIMIT -B64 -c$CACHE_LIMIT -b64 -a16 --database  results/fully-associative.db --batched
    echo "$file, 12-way" >> results/runtimes2.txt
    /usr/bin/time -f "%E" -a -o results/runtimes2.txt bash $SCRIPT_DIR/run-single-simulation.sh "$file" -C$CACHE_LIMIT -B64 -A12 -c$CACHE_LIMIT -b64 -a16 --database  results/12-way2.db --batched
    # echo "$file, 16-way" >> results/runtimes.txt
    # /usr/bin/time -f "%E" -a -o results/runtimes.txt bash $SCRIPT_DIR/run-single-simulation.sh "$file" -C$CACHE_LIMIT -B64 -A16 -c$CACHE_LIMIT -b64 -a16 --database  results/16-way.db --batched
done

