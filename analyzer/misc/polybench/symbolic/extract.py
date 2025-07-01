import csv
import sqlite3
import argparse

def main():
    parser = argparse.ArgumentParser(description="Load new CSV format into SQLite with ON CONFLICT UPDATE.")
    parser.add_argument('--csv', required=True, help='Path to the input CSV file')
    parser.add_argument('--db', required=True, help='Path to the output SQLite DB file')
    args = parser.parse_args()

    CSV_FILE = args.csv
    DB_FILE = args.db

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # 所有字段 NOT NULL
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
        program TEXT NOT NULL,
        script_dir TEXT NOT NULL,
        PRIMARY KEY (testname, approxmethod)
    )
    ''')

    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 从参数字段直接取
            testname = row['parameter_testname']
            approxmethod = row['parameter_approxmethod'].strip('=') or 'none'
            command = row['command']
            program = row['parameter_PROGRAM']
            script_dir = row['parameter_SCRIPT_DIR']

            cur.execute('''
                INSERT INTO results
                (testname, approxmethod, command, mean, stddev, median, user, system, min, max, program, script_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(testname, approxmethod) DO UPDATE SET
                    command=excluded.command,
                    mean=excluded.mean,
                    stddev=excluded.stddev,
                    median=excluded.median,
                    user=excluded.user,
                    system=excluded.system,
                    min=excluded.min,
                    max=excluded.max,
                    program=excluded.program,
                    script_dir=excluded.script_dir
            ''', (
                testname,
                approxmethod,
                command,
                float(row['mean']),
                float(row['stddev']),
                float(row['median']),
                float(row['user']),
                float(row['system']),
                float(row['min']),
                float(row['max']),
                program,
                script_dir
            ))

    conn.commit()
    conn.close()

    print(f"✅ Done! SQLite DB saved to: {DB_FILE}")

if __name__ == "__main__":
    main()
