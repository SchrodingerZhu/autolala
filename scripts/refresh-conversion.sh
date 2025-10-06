skewness=2.5
for i in results/fully-associative/*.json; do
    PROGRAM_NAME=$(basename $i .json)
    cargo run --release --quiet --bin assoc-conv -- -o results/12-way/${PROGRAM_NAME}.json -i $i -a 12 -s $skewness
done
