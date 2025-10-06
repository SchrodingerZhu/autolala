#!/usr/bin/env bash

touch results/prediction-runtimes.txt
for file in analyzer/misc/polybench/polygeist/constant/*.c; do

    # Check if file already exists in results
    if grep -Fxq "$file" results/prediction-runtimes.txt; then
        echo "Skipping $file (already processed)"
        continue
    fi
    
    echo "Running $file"
    echo "$file" >> results/prediction-runtimes.txt
    /usr/bin/time -f "%E" -a -o results/prediction-runtimes.txt bash scripts/run-single-prediction.sh "$file"
done
