
import sqlite3
from pathlib import Path

database_path = Path("data/raw/metadata_extracted/metadata.db")

if not database_path.exists():
    print("Database not found at:", database_path)
    print("Check the extracted filenames and folder.")
    raise SystemExit(1)

connection = sqlite3.connect(str(database_path))

try:
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    print("Tables:")
    for (table_name,) in tables:
        print("-", table_name)

    for (table_name,) in tables:
        print(f"\nColumns in {table_name}:")
        columns = connection.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()

        for column in columns:
            print(column[1], "|", column[2])

        print(f"\nFirst 2 records from {table_name}:")
        rows = connection.execute(
            f'SELECT * FROM "{table_name}" LIMIT 2'
        ).fetchall()

        for row in rows:
            print(row)

finally:
    connection.close()
