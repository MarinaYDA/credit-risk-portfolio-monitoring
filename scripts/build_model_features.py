"""
Engineer model-ready features from the loan-level target table: convert
emp_length to a numeric year count, validate that fico_range_low needs
no conversion, and report missing values across every feature column.

Reads data/processed/loan_level_target.csv and does not modify it.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "data" / "processed" / "loan_level_target.csv"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "model_features.csv"

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

FEATURE_COLUMNS = [
    "grade",
    "purpose",
    "home_ownership",
    "emp_length_years",
    "annual_inc",
    "dti",
    "fico_range_low",
    "loan_amnt",
    "int_rate",
]

# TODO (before training in pd_model.py): dti and annual_inc both have a
# handful of extreme outliers that StandardScaler alone won't fully tame
# for logistic regression - real values, not typos (dti up to 240.87;
# annual_inc up to 4,160,000). Needs a capping or transform decision
# before training.


if __name__ == "__main__":
    target = pd.read_csv(TARGET_PATH)

    original_emp_missing = target["emp_length"].isna().sum()

    emp_length_years = target["emp_length"].map(EMP_LENGTH_MAP)
    # .map() returns NaN both for missing input and for any unmapped
    # string - confirm every non-null emp_length value was recognized,
    # so an unexpected category doesn't silently become a missing value.
    unmapped = target.loc[
        target["emp_length"].notna() & emp_length_years.isna(), "emp_length"
    ].unique()
    assert len(unmapped) == 0, f"Unrecognized emp_length values: {unmapped}"

    # fico_range_low is already numeric with no missing values in this
    # dataset - validated here, not transformed.
    assert pd.api.types.is_numeric_dtype(target["fico_range_low"]), (
        "fico_range_low is not numeric - a conversion step is needed after all"
    )
    assert target["fico_range_low"].isna().sum() == 0, (
        "fico_range_low has missing values - handle before modeling"
    )

    features = target.drop(columns=["emp_length"])
    features["emp_length_years"] = emp_length_years

    print("=== missing values across all feature columns ===")
    print(features[["loan_id", "defaulted_12m"] + FEATURE_COLUMNS].isna().sum())

    # dti is null in exactly 4 rows, all with annual_inc == 0 - a
    # division-by-zero in the upstream DTI calculation, not a reporting
    # gap. Confirmed: loan_ids 111051579, 119336653, 127024211, 130789653
    # all have annual_inc == 0, and no other row has annual_inc == 0.
    print("\n=== dti missing rows (expected: 4, all annual_inc == 0) ===")
    print(
        features.loc[features["dti"].isna(), ["loan_id", "annual_inc", "dti"]]
        .to_string(index=False)
    )

    # --- Verification (before writing anything to disk) ---
    print("\n=== Verification ===")
    assert len(features) == 3500, f"Expected 3,500 rows, got {len(features)}"
    print(f"row count: {len(features)} - PASS")

    new_emp_missing = features["emp_length_years"].isna().sum()
    assert new_emp_missing == original_emp_missing, (
        f"emp_length_years missing ({new_emp_missing}) does not match "
        f"original emp_length missing ({original_emp_missing})"
    )
    print(f"emp_length_years missing count matches original emp_length ({new_emp_missing}) - PASS")

    assert features["emp_length_years"].dropna().between(0, 10).all(), (
        "emp_length_years has values outside [0, 10]"
    )
    print("emp_length_years values within [0, 10] - PASS")

    assert features["fico_range_low"].isna().sum() == 0
    print("fico_range_low fully non-null - PASS")

    features.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved {len(features):,} rows to {FEATURES_PATH}")
