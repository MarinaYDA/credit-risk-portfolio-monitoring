"""
Build the loan-level target variable for the PD model: whether a loan
reached "default" within its first 12 months on book, joined with the
loan-level attributes used for modeling (see docs/charter.md's Core
Methods section).
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "data" / "raw" / "lending_club_sample.csv"
MONTHLY_PATH = REPO_ROOT / "data" / "processed" / "loan_monthly_performance.csv"
TARGET_PATH = REPO_ROOT / "data" / "processed" / "loan_level_target.csv"

DEFAULT_WINDOW_MONTHS = 12

FEATURE_COLUMNS = [
    "grade",
    "purpose",
    "home_ownership",
    "emp_length",
    "annual_inc",
    "dti",
    "fico_range_low",
]


def build_target(monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per loan_id: 1 if the loan reached "default" within its
    first DEFAULT_WINDOW_MONTHS months on book, else 0."""
    early = monthly[monthly["months_on_book"] <= DEFAULT_WINDOW_MONTHS]
    return (
        early.groupby("loan_id")["delinquency_state"]
        .apply(lambda s: (s == "default").any())
        .astype(int)
        .rename("defaulted_12m")
        .reset_index()
    )


if __name__ == "__main__":
    sample = pd.read_csv(SAMPLE_PATH)
    monthly = pd.read_csv(MONTHLY_PATH)

    target = build_target(monthly)

    result = target.merge(
        sample[["loan_id"] + FEATURE_COLUMNS],
        on="loan_id",
        how="left",
        validate="one_to_one",
    )

    print("=== loan_level_target shape ===")
    print(result.shape)

    result.to_csv(TARGET_PATH, index=False)
    print(f"\nSaved {len(result):,} rows to {TARGET_PATH}")

    # --- Verification 1: exactly 3,500 rows, no duplicates from the join ---
    print("\n=== Verification 1: row count ===")
    print(f"rows: {len(result)}")
    print(f"distinct loan_id: {result['loan_id'].nunique()}")
    assert len(result) == 3500, f"Expected exactly 3,500 rows, got {len(result)}"
    assert result["loan_id"].is_unique, "Duplicate loan_id found in target"
    print("PASS")

    # --- Verification 2: 12m default rate meaningfully lower than
    # full-term default rate, for every grade (not just overall) ---
    print("\n=== Verification 2: 12m default rate vs full-term default rate, by grade ===")
    final_state = monthly.groupby("loan_id")["delinquency_state"].last()
    full_term = final_state.reset_index().rename(
        columns={"delinquency_state": "final_state"}
    )
    full_term = full_term.merge(sample[["loan_id", "grade"]], on="loan_id")
    full_term["defaulted_full_term"] = (full_term["final_state"] == "default").astype(int)
    rate_full_term_by_grade = full_term.groupby("grade")["defaulted_full_term"].mean()

    rate_12m_by_grade = result.groupby("grade")["defaulted_12m"].mean()

    comparison = pd.DataFrame(
        {
            "rate_12m": rate_12m_by_grade,
            "rate_full_term": rate_full_term_by_grade,
        }
    )
    comparison["12m_lower"] = comparison["rate_12m"] < comparison["rate_full_term"]
    print(comparison)

    assert comparison["12m_lower"].all(), (
        "12-month default rate is not lower than the full-term rate for "
        "every grade - investigate before proceeding."
    )
    print("PASS: 12-month default rate is lower than the full-term rate for every grade.")

    print(f"\nOverall 12m default rate: {result['defaulted_12m'].mean():.2%}")

    # --- Verification 3: clean 1:1 join, no missing/extra loan_id ---
    print("\n=== Verification 3: loan_id set match between target and sample ===")
    target_ids = set(result["loan_id"])
    sample_ids = set(sample["loan_id"])
    missing_from_sample = target_ids - sample_ids
    missing_from_target = sample_ids - target_ids
    print(f"in target but not in sample: {len(missing_from_sample)}")
    print(f"in sample but not in target: {len(missing_from_target)}")
    assert not missing_from_sample and not missing_from_target, (
        "loan_id sets differ between target and sample"
    )
    print("PASS: loan_id sets match exactly.")
