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


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_PATH)
    loan = df.iloc[0]

    term_months = int(loan["term"].strip().split()[0])
    rng = np.random.default_rng(42)
    path = simulate_path(loan["grade"], term_months, rng)

    print(f"Loan id: {loan['id']}")
    print(f"Grade: {loan['grade']}")
    print(f"Term: {term_months} months")
    print()
    print("Simulated monthly path:")
    for month, state in enumerate(path, start=1):
        print(f"  month {month:>2}: {state}")
