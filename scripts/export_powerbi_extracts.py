"""
Export the vintage curve and roll rate query results to standalone CSVs
for Power BI import.

Reads sql/01_vintage_analysis.sql and sql/02_roll_rate.sql from disk
verbatim (not retyped inline) and runs them via DuckDB, so the Power BI
extracts can never silently drift from the committed, validated SQL.

Both queries reference data/processed/loan_monthly_performance.csv by a
relative path (as written), so DuckDB needs the process's current
working directory to be the repo root to resolve it - this script
changes into the repo root itself before running them, so it works
regardless of where it's invoked from.
"""

import os
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
VINTAGE_SQL_PATH = REPO_ROOT / "sql" / "01_vintage_analysis.sql"
ROLL_RATE_SQL_PATH = REPO_ROOT / "sql" / "02_roll_rate.sql"
VINTAGE_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "vintage_curve.csv"
ROLL_RATE_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "roll_rate_matrix.csv"


def main() -> None:
    os.chdir(REPO_ROOT)
    con = duckdb.connect()

    vintage_query = VINTAGE_SQL_PATH.read_text()
    vintage_df = con.sql(vintage_query).df()
    vintage_df.to_csv(VINTAGE_OUTPUT_PATH, index=False)
    print(f"vintage_curve.csv: {len(vintage_df):,} rows -> {VINTAGE_OUTPUT_PATH}")

    roll_rate_query = ROLL_RATE_SQL_PATH.read_text()
    roll_rate_df = con.sql(roll_rate_query).df()
    roll_rate_df.to_csv(ROLL_RATE_OUTPUT_PATH, index=False)
    print(f"roll_rate_matrix.csv: {len(roll_rate_df):,} rows -> {ROLL_RATE_OUTPUT_PATH}")

    con.close()


if __name__ == "__main__":
    main()
