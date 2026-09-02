"""
Build a grade-balanced sample of the Lending Club dataset.

Reads accepted_2007_to_2018Q4.csv (only the columns needed), drops rows
missing issue_d or grade (same rule as explore_raw_data.py), then draws
exactly 500 loans per grade (A-G) with a fixed random seed for a
reproducible 3,500-row sample. Saves to data/raw/lending_club_sample.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "lending_club_sample.csv"

COLUMNS = [
    "id",
    "issue_d",
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "annual_inc",
    "dti",
    "fico_range_low",
    "purpose",
    "home_ownership",
    "emp_length",
]

PER_GRADE_SAMPLE_SIZE = 500
RANDOM_STATE = 42  # named to match pandas' random_state= parameter; see RANDOM_SEED in generate_loan_performance.py
CHUNK_SIZE = 250_000


def build_sample() -> pd.DataFrame:
    """Read the raw CSV in chunks and return a grade-balanced sample."""
    chunks = pd.read_csv(
        RAW_PATH,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    df = pd.concat(
        (chunk.dropna(subset=["issue_d", "grade"]) for chunk in chunks),
        ignore_index=True,
    )

    sample = (
        df.groupby("grade", group_keys=False)
        .sample(n=PER_GRADE_SAMPLE_SIZE, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return sample.rename(columns={"id": "loan_id"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="build and validate the sample without writing the output CSV",
    )
    args = parser.parse_args()

    sample = build_sample()

    print("=== Sample shape ===")
    print(sample.shape)

    print("\n=== Per-grade value_counts() of final sample ===")
    print(sample["grade"].value_counts().sort_index())

    print("\n=== loan_id uniqueness check ===")
    print("df['loan_id'].is_unique:", sample["loan_id"].is_unique)

    if args.no_save:
        print("\n--no-save passed: CSV not written.")
        return

    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(sample):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
