SCRIPT_PATH=$(dirname $(realpath $0))
for i in $SCRIPT_PATH/*.mlir; do
    echo "Running $i"
    # cargo run --release --bin cachegrind-runner -- -i $i -A4 -C32768 -B64 -a4 -b64 -c262144 -d $SCRIPT_PATH/database.sqlite 
    # cargo run --release --bin cachegrind-runner -- -i $i -A512 -C32768 -B64 -a4096 -b64 -c262144 -d $SCRIPT_PATH/database.sqlite
    cargo run --release --bin cachegrind-runner -- -i $i -A32 -C32768 -B64 -a16 -b64 -c262144 -d $SCRIPT_PATH/database.sqlite 
done
