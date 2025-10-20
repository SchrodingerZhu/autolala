#!/usr/bin/env bash

touch results/prediction-runtimes-einsum.txt
for file in analyzer/misc/einsum/constant_global/*.c; do

    # Check if file already exists in results
    if grep -Fxq "$file" results/prediction-runtimes-einsum.txt; then
        echo "Skipping $file (already processed)"
        continue
    fi
    
    echo "Running $file"
    echo "$file" >> results/prediction-runtimes-einsum.txt
    /usr/bin/time -f "%E" -a -o results/prediction-runtimes-einsum.txt bash scripts/run-single-prediction-einsum.sh "$file"
done
