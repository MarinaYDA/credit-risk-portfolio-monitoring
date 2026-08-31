"""
Simulate monthly delinquency paths for loans, since the raw Lending Club
dataset has no monthly delinquency panel (see docs/charter.md for the
disclosed limitation this addresses).

Monthly transitions follow a grade-calibrated Markov chain: the hazard
rate for current -> dpd30 increases with risk grade; roll-forward/cure
rates between delinquency buckets are held constant across grades.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "data" / "raw" / "lending_club_sample.csv"
MONTHLY_PERFORMANCE_PATH = (
    REPO_ROOT / "data" / "processed" / "loan_monthly_performance.csv"
)
RANDOM_SEED = 42

TERMINAL_STATES = {"default", "paid_off"}

# Monthly hazard of current -> dpd30, by grade. Illustrative parameters,
# not fitted to real hazard-rate data (none exists for this dataset).
# These specific values were provided directly by the project owner,
# not independently derived.
GRADE_HAZARD = {
    "A": 0.006,
    "B": 0.010,
    "C": 0.016,
    "D": 0.024,
    "E": 0.034,
    "F": 0.048,
    "G": 0.065,
}

PREPAY_PROB = 0.02  # current -> paid_off (early payoff), flat across grades

# Roll-forward / cure probabilities between delinquency buckets, held
# constant across grades (only the initial hazard is grade-calibrated).
ROLL_30_TO_60 = 0.30
CURE_30_TO_CURRENT = 0.40

ROLL_60_TO_90 = 0.40
CURE_60_TO_30 = 0.20

ROLL_90_TO_DEFAULT = 0.50
CURE_90_TO_60 = 0.15


def simulate_path(
    grade: str,
    term_months: int,
    rng: np.random.Generator,
) -> list[str]:
    """Simulate one loan's monthly delinquency state path.

    Starts at "current" and simulates month by month until the loan
    reaches a terminal state (default or paid_off) or term_months is
    exhausted, whichever comes first.

    Unlike a simpler grade-calibrated model that forces a transition
    every month, this version allows an account to remain in its current
    delinquency bucket, which better reflects observed roll-rate
    behavior.
    """
    hazard = GRADE_HAZARD[grade]
    state = "current"
    path = []

    for _ in range(term_months):
        if state == "current":
            u = rng.random()
            if u < hazard:
                state = "dpd30"
            elif u < hazard + PREPAY_PROB:
                state = "paid_off"
        elif state == "dpd30":
            u = rng.random()
            if u < ROLL_30_TO_60:
                state = "dpd60"
            elif u < ROLL_30_TO_60 + CURE_30_TO_CURRENT:
                state = "current"
        elif state == "dpd60":
            u = rng.random()
            if u < ROLL_60_TO_90:
                state = "dpd90"
            elif u < ROLL_60_TO_90 + CURE_60_TO_30:
                state = "dpd30"
        elif state == "dpd90":
            u = rng.random()
            if u < ROLL_90_TO_DEFAULT:
                state = "default"
            elif u < ROLL_90_TO_DEFAULT + CURE_90_TO_60:
                state = "dpd60"

        path.append(state)

        if state in TERMINAL_STATES:
            break

    return path


def parse_term_months(term: str) -> int:
    """Parse a term string like " 36 months" into an integer month count."""
    return int(term.strip().split()[0])


def build_monthly_performance(
    sample: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate a monthly delinquency panel for every loan in sample.

    Draws from a single shared rng across all loans, consumed in the
    same row order as sample, so the full run is only reproducible as a
    whole (with a fixed rng seed and a fixed row order), not loan by
    loan in isolation.
    """
    vintage_months = pd.to_datetime(
        sample["issue_d"], format="%b-%Y"
    ).dt.to_period("M")

    records = []
    for loan_id, grade, term, vintage_month in zip(
        sample["id"], sample["grade"], sample["term"], vintage_months
    ):
        term_months = parse_term_months(term)
        path = simulate_path(grade, term_months, rng)
        for months_on_book, state in enumerate(path, start=1):
            records.append(
                {
                    "loan_id": loan_id,
                    "vintage_month": vintage_month,
                    "months_on_book": months_on_book,
                    "snapshot_month": vintage_month + (months_on_book - 1),
                    "delinquency_state": state,
                }
            )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    sample = pd.read_csv(SAMPLE_PATH)
    rng = np.random.default_rng(RANDOM_SEED)

    monthly = build_monthly_performance(sample, rng)

    print("=== monthly performance shape ===")
    print(monthly.shape)

    monthly.to_csv(MONTHLY_PERFORMANCE_PATH, index=False)
    print(f"\nSaved {len(monthly):,} rows to {MONTHLY_PERFORMANCE_PATH}")

    # Verify grade A's default share is meaningfully lower than grade G's,
    # rather than assuming the grade-calibrated hazard produced the
    # expected risk ordering.
    final_state = monthly.groupby("loan_id")["delinquency_state"].last()
    outcomes = final_state.to_frame("final_state").merge(
        sample[["id", "grade"]].rename(columns={"id": "loan_id"}),
        on="loan_id",
    )
    outcomes["reached_default"] = outcomes["final_state"] == "default"
    default_rate_by_grade = (
        outcomes.groupby("grade")["reached_default"].mean().sort_index()
    )

    print("\n=== default rate by grade ===")
    print(default_rate_by_grade)

    rate_a = default_rate_by_grade["A"]
    rate_g = default_rate_by_grade["G"]
    print(f"\nGrade A default rate: {rate_a:.2%}")
    print(f"Grade G default rate: {rate_g:.2%}")

    assert rate_a < rate_g, (
        "Grade A default rate is not lower than grade G — "
        "investigate before proceeding."
    )
    print("Check passed: grade A default rate is lower than grade G.")
