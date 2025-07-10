import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot grouped bar chart with legend for methods.")
    parser.add_argument('--db', required=True, help='Path to the SQLite DB file')
    parser.add_argument('--output', default='output.png', help='Path to save the output plot')
    args = parser.parse_args()

    DB_FILE = args.db

    # === 连接 DB ===
    conn = sqlite3.connect(DB_FILE)

    # === 查询 ===
    df = pd.read_sql_query('''
        SELECT testname, approxmethod, mean FROM results
    ''', conn)

    conn.close()

    # === 排序 ===
    df = df.sort_values(['testname', 'approxmethod'])
    print(df)

    # === seaborn 分组条形图 ===
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='testname', y='mean', hue='approxmethod')

    plt.xlabel('Program', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Time (seconds)', fontsize=14, fontweight='bold')
    #plt.title('Mean Running Time by Program and Approximation Method', fontsize=16, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    plt.legend(title='Approximation Method', title_fontsize=12, fontsize=12)

    plt.tight_layout()

    plt.savefig(args.output)
    

if __name__ == "__main__":
    main()
