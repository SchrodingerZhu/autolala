skewness=2.71828
for i in results/fully-associative/*.json; do
    PROGRAM_NAME=$(basename $i .json)
    echo "Processing $PROGRAM_NAME with skewness $skewness"
    cargo run --release --quiet --bin assoc-conv -- -o results/12-way/${PROGRAM_NAME}.json -i $i -a 12 -s $skewness -d gaussian,1.0
done
