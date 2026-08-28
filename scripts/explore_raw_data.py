"""
Initial inspection of the raw LendingClub accepted-loans dataset.

The script loads the first 200,000 rows because the full dataset is
large. It examines:
- dataset dimensions and column data types;
- missing-value counts and percentages;
- summary statistics for key credit-risk variables;
- loan grade and status distributions;
- the format and parsing quality of the issue_d field.

Note: The loaded rows are an exploratory subset and may not represent
the distribution of the complete dataset.
"""

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

RAW_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "accepted_2007_to_2018Q4.csv"
)
NROWS = 200_000

# Selected using LendingClub's official data dictionary (which defines
# what each column in the dataset means), cross-referenced against the
# business question in docs/charter.md. Data dictionary source:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club (Data tab)
# or, if not available there:
# https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary
KEY_NUMERIC_COLUMNS = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
]


def main() -> None:
    """Load and inspect an exploratory subset of the raw dataset."""
    print(f"Loading the first {NROWS:,} rows from:")
    print(RAW_PATH)

    df = pd.read_csv(
        RAW_PATH,
        nrows=NROWS,
        low_memory=False,
    )

    print("\n=== Dataset shape ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\n=== Dataset structure ===")
    df.info(verbose=True, show_counts=True)

    print("\n=== Missing values: top 20 columns ===")
    missing_summary = (
        df.isna()
        .sum()
        .to_frame("missing_count")
        .assign(
            missing_pct=lambda x: (x["missing_count"] / len(df) * 100).round(2)
        )
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
        .head(20)
    )
    print(missing_summary)

    print("\n=== Key numeric variable summary ===")
    print(df[KEY_NUMERIC_COLUMNS].describe().round(2))

    print("\n=== Grade distribution ===")
    grade_distribution = pd.concat(
        [
            df["grade"].value_counts(dropna=False).sort_index().rename("count"),
            (
                df["grade"]
                .value_counts(normalize=True, dropna=False)
                .sort_index()
                .mul(100)
                .round(2)
                .rename("percent")
            ),
        ],
        axis=1,
    )
    print(grade_distribution)

    print("\n=== Loan-status distribution ===")
    loan_status_distribution = pd.concat(
        [
            df["loan_status"]
            .value_counts(dropna=False)
            .rename("count"),
            (
                df["loan_status"]
                .value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(2)
                .rename("percent")
            ),
        ],
        axis=1,
    )
    print(loan_status_distribution)

    print("\n=== issue_d date-format check ===")
    print("Raw examples:")
    print(df["issue_d"].dropna().head(10).tolist())

    parsed_issue_date = pd.to_datetime(
        df["issue_d"],
        format="%b-%Y",
        errors="coerce",
    )
    print(f"Original dtype: {df['issue_d'].dtype}")
    print(f"Parsed dtype: {parsed_issue_date.dtype}")
    print(
        "Unparseable non-null values:",
        (df["issue_d"].notna() & parsed_issue_date.isna()).sum(),
    )
    print(
        "Parsed date range:",
        parsed_issue_date.min(),
        "to",
        parsed_issue_date.max(),
    )


if __name__ == "__main__":
    main()
