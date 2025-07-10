SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo "test,time" > "$SCRIPT_DIR/analysis_time.csv"
for i in $SCRIPT_DIR/const_*.json; do
    basename=$(basename "$i" .json)
    without_prefix="${basename#const_}"
    time=$(jq -r '.analysis_time | (.secs + (.nanos / 1e9))' "$i")
    echo "$without_prefix,$time" >> "$SCRIPT_DIR/analysis_time.csv"
done
