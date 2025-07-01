SCRIPT_PATH=$(dirname $(realpath $0))
for i in $SCRIPT_PATH/*.mlir; do
    echo "Running $i"
    for n in 1 2 4 8; do
        echo "Number of sets: $n"
        s=$(python3 -c "print(int(2048 / $n))")
        cargo run --release --bin cachegrind-runner -- -i "$i" -N $n -S "$s" -d $SCRIPT_PATH/database.sqlite --skip-existing
    done
done
