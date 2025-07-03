import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot grouped bar chart with legend for methods.")
    parser.add_argument('--db-const', required=True, help='Path to the SQLite DB file')
    parser.add_argument('--db-symbolic', required=True, help='Path to the SQLite DB file for symbolic')
    parser.add_argument('--output', default='output.png', help='Path to save the output plot')
    args = parser.parse_args()

    DB_FILE_CONST = args.db_const
    DB_FILE_SYMBOLIC = args.db_symbolic

    # === 连接 DB ===
    conn_const = sqlite3.connect(DB_FILE_CONST)
    conn_symbolic = sqlite3.connect(DB_FILE_SYMBOLIC)

    # === 查询 ===
    df_const = pd.read_sql_query('''
        SELECT testname, approxmethod, mean FROM results
    ''', conn_const)

    df_symbolic = pd.read_sql_query('''
        SELECT testname, approxmethod, mean FROM results
    ''', conn_symbolic)

    conn_const.close()
    conn_symbolic.close()

    # === 排序 ===
    df_const = df_const.sort_values(['testname', 'approxmethod'])
    df_symbolic = df_symbolic.sort_values(['testname', 'approxmethod'])

    # === 合并数据 ===
    # 给 df_const 和 df_symbolic 添加一个新列，标记来源
    df_const['source']   = 'const'
    df_symbolic['source'] = 'symbolic'

    # 将两张表合并为一张
    df_all = pd.concat([df_const, df_symbolic], ignore_index=True)

    # （可选）再按需要对合并后的表进行排序
    df_all = df_all.sort_values(['testname', 'approxmethod', 'source'])
    df_all['approxmethod'] = df_all['approxmethod'].replace('none', 'exact')
    df_all['method_source'] = df_all[['approxmethod', 'source']].agg(','.join, axis=1)

    # 删除原来的两列（如果不再需要）
    df_all = df_all.drop(columns=['approxmethod', 'source'])


    # === seaborn 分组条形图 ===
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_all, x='testname', y='mean', hue='method_source')

    plt.xlabel('Program', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Time (seconds)', fontsize=14, fontweight='bold')
    #plt.title('Mean Running Time by Program and Approximation Method', fontsize=16, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    plt.legend(title='Approximation Method and Source', title_fontsize=12, fontsize=12)

    plt.tight_layout()

    plt.savefig(args.output)
    

if __name__ == "__main__":
    main()
