import argparse
import sqlite3
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Process SQLite cache records and merge associtivity data.")
    parser.add_argument("db_path", help="Path to the SQLite database file.")
    parser.add_argument("prediction_path", help="Path to the prediction file (not used in this script).")
    args = parser.parse_args()

    # Connect to SQLite database
    conn = sqlite3.connect(args.db_path)

    # Original query
    query = """
    SELECT 
        program, 
        d1_associativity,
        d1_miss_count,
        total_access,
        process_time/1E6 AS 'process time (ms)'
    FROM records 
    WHERE d1_associativity IN (32, 512)
    """

    df = pd.read_sql_query(query, conn)

    # Clean program names
    df["program"] = df["program"].str.replace(r"^const_", "", regex=True).str.replace(r"\.mlir$", "", regex=True)

    # Convert d1_associativity to categorical type for better sorting
    df_pivot = df.pivot_table(
        index="program",
        columns=["d1_associativity"],
        values=[ "d1_miss_count", "total_access", "process time (ms)"],
    ).reset_index()

    # remove repeated total_access
    df_pivot = df_pivot.drop(columns=[("total_access", 512)])

    # # Flatten MultiIndex columns
    df_pivot.columns = [
        "Program",
        "D1 Miss Count (16-way Associative)",
        "D1 Miss Count (Fully Associative)",
        "Process Time (ms, 16-way Associative)",
        "Process Time (ms, Fully Associative)",
        "Total Access",
    ]

    # Read prediction file (CSV format)
    try:
        df_prediction = pd.read_csv(args.prediction_path)
        # Clean program names in prediction data
        df_prediction["Program"] = df_prediction["Program"].str.replace(r"^const_", "", regex=True).str.replace(r"\.json$", "", regex=True)

        # Merge prediction data with the pivot table
        df_pivot = pd.merge(df_pivot, df_prediction, on="Program", how="left")
    except FileNotFoundError:
        print(f"Prediction file {args.prediction_path} not found. Skipping merge.")
    
    # Calculate percentage error for program
    # Error (Fully Associative) = (Predicted Miss Count - D1 Miss Count (Fully Associative)) / Total Access * 100
    # Error (4-way Associative) = (Predicted Miss Count - D1 Miss Count (4-way Associative)) / Total Access * 100
    if "Predicted Miss Count" in df_pivot.columns:
        df_pivot["Error (Fully Associative)"] = (
            abs(df_pivot["Predicted Miss Count"] - df_pivot["D1 Miss Count (Fully Associative)"]) 
            / df_pivot["Total Access"] * 100
        )
        df_pivot["Error (4-way Associative)"] = (
            abs(df_pivot["Predicted Miss Count"] - df_pivot["D1 Miss Count (16-way Associative)"]) 
            / df_pivot["Total Access"] * 100
        )
    else:
        print("Prediction file does not contain 'Predicted Miss Count'. Skipping error calculation.")

    # Move Predicted Miss Count right after Fully Associative column

    if "Predicted Miss Count" in df_pivot.columns:
        cols = list(df_pivot.columns)
        d1_index = cols.index("D1 Miss Count (Fully Associative)")
        pred_index = cols.index("Predicted Miss Count")
        # Move it to the right after Fully Associative
        new_order = (
            cols[:d1_index+1] + 
            ["Predicted Miss Count"] + 
            cols[d1_index+1:pred_index] + 
            cols[pred_index+1:]
        )
        df_pivot = df_pivot[new_order]

    # Show result
    print(df_pivot)

if __name__ == "__main__":
    main()
