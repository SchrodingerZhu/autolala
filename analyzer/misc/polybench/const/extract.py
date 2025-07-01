import csv
import sqlite3
import os
import re
import argparse

def parse_command(command):
    """解析 -i 和 --approximation-method"""
    input_match = re.search(r'-i\s+(\S+)', command)
    if input_match:
        path = input_match.group(1)
        basename = os.path.basename(path)
        basename = re.sub(r'^const_', '', basename)
        basename = re.sub(r'\.mlir$', '', basename)
    else:
        basename = 'unknown'

    approx_match = re.search(r'--approximation-method(?:=|\s+)?(\w+)?', command)
    approx = approx_match.group(1) if approx_match and approx_match.group(1) else 'none'

    return basename, approx

def main():
    parser = argparse.ArgumentParser(description="Load CSV and update SQLite DB with ON CONFLICT.")
    parser.add_argument('--csv', required=True, help='Path to the input CSV file')
    parser.add_argument('--db', required=True, help='Path to the output SQLite DB file')
    args = parser.parse_args()

    CSV_FILE = args.csv
    DB_FILE = args.db

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # === 建表，所有字段 NOT NULL ===
    cur.execute('''
    CREATE TABLE IF NOT EXISTS results (
        testname TEXT NOT NULL,
        approxmethod TEXT NOT NULL,
        command TEXT NOT NULL,
        mean REAL NOT NULL,
        stddev REAL NOT NULL,
        median REAL NOT NULL,
        user REAL NOT NULL,
        system REAL NOT NULL,
        min REAL NOT NULL,
        max REAL NOT NULL,
        PRIMARY KEY (testname, approxmethod)
    )
    ''')

    # === 读取 CSV 并插入 ===
    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            testname, approxmethod = parse_command(row['command'])
            cur.execute('''
                INSERT INTO results
                (testname, approxmethod, command, mean, stddev, median, user, system, min, max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(testname, approxmethod) DO UPDATE SET
                    command=excluded.command,
                    mean=excluded.mean,
                    stddev=excluded.stddev,
                    median=excluded.median,
                    user=excluded.user,
                    system=excluded.system,
                    min=excluded.min,
                    max=excluded.max
            ''', (
                testname,
                approxmethod,
                row['command'],
                float(row['mean']),
                float(row['stddev']),
                float(row['median']),
                float(row['user']),
                float(row['system']),
                float(row['min']),
                float(row['max'])
            ))

    conn.commit()
    conn.close()

    print(f"✅ Done! SQLite saved to: {DB_FILE}")

if __name__ == "__main__":
    main()
