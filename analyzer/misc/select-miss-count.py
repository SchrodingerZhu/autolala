import argparse
import sqlite3
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Process SQLite cache records and merge associtivity data.")
    parser.add_argument("db_path", help="Path to the SQLite database file.")
    args = parser.parse_args()

    # Connect to SQLite database
    conn = sqlite3.connect(args.db_path)

    # Original query
    query = """
    SELECT 
        program, 
        set_size AS associtivity, 
        miss_count, 
        total_access
    FROM records 
    WHERE 
        (set_size=8 AND cache_size=4096) 
        OR (num_sets=1 AND cache_size=4096)
    """

    df = pd.read_sql_query(query, conn)

    # Clean program names
    df["program"] = df["program"].str.replace(r"^const_", "", regex=True).str.replace(r"\.mlir$", "", regex=True)

    # Pivot so associtivity 8 and 64 are on same row
    df_pivot = df.pivot_table(
        index="program",
        columns="associtivity",
        values=["miss_count", "total_access"],
    ).reset_index()

    # remove repeated total_access
    df_pivot = df_pivot.drop(columns=[("total_access", 64)])

    # Flatten MultiIndex columns
    df_pivot.columns = [
        "program",
        "miss_count_8", "miss_count_64",
        "total_access",
    ]

    # Show result
    print(df_pivot)

    # Save to CSV
    df_pivot.to_csv("merged_assoc_data.csv", index=False)
    print("Saved merged data to merged_assoc_data.csv")

if __name__ == "__main__":
    main()
