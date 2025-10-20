#!/usr/bin/env bash
CACHE_LIMIT=65536
mkdir -p results
export RUST_LOG=warn
SCRIPT_DIR=$(dirname "$0")
for file in analyzer/misc/einsum/constant_global/*.c; do

    # Skip if this file has already been processed
    if [ -f results/runtimes-einsum.txt ] && grep -q "^$file," results/runtimes-einsum.txt; then
        echo "Skipping $file (already processed)"
        continue
    fi
    
    echo "Running $file"
    echo "$file, fully" >> results/runtimes-einsum.txt
    /usr/bin/time -f "%E" -a -o results/runtimes-einsum.txt bash $SCRIPT_DIR/run-single-simulation.sh "$file" -C$CACHE_LIMIT -B64 -c$CACHE_LIMIT -b64 -a16 --database  results/fully-associative-einsum.db --batched
    echo "$file, 12-way" >> results/runtimes-einsum.txt
    /usr/bin/time -f "%E" -a -o results/runtimes-einsum.txt bash $SCRIPT_DIR/run-single-simulation.sh "$file" -C$CACHE_LIMIT -B64 -A12 -c$CACHE_LIMIT -b64 -a16 --database  results/12-way-einsum.db --batched
done

