# Credit Risk Portfolio Monitoring

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the `accepted_2007_to_2018Q4.csv` Lending Club dataset from
   Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
3. Place the downloaded file at `data/raw/accepted_2007_to_2018Q4.csv`.

## Data & Methodology

**Data source:** Loan-level attributes (amount, rate, grade, purpose,
income, DTI, FICO range, issue date) come from the real, publicly issued
LendingClub dataset (see [Setup](#setup) for the download source).
Monthly delinquency status is not present in the raw data — it is
simulated via a grade-calibrated Markov chain, disclosed as a deliberate
design choice in [docs/charter.md](docs/charter.md) and implemented in
[scripts/generate_loan_performance.py](scripts/generate_loan_performance.py).

**Sampling design:** Analysis runs on a reproducible, grade-balanced
sample of 3,500 loans (500 per grade, A through G, fixed random seed 42),
built from the full ~2.26M-row raw dataset by
[scripts/build_lending_club_sample.py](scripts/build_lending_club_sample.py).
Note: `installment` and `fico_range_high` were both examined during
initial exploration in
[scripts/explore_raw_data.py](scripts/explore_raw_data.py) but are not
present in the sample's final column set — no rationale for that
exclusion is on record.

**Vintage curve analysis:** Tracks cumulative default rate by loan age
for each origination cohort, using a fixed cohort-size denominator so the
rate is monotonically non-decreasing with age. See
[sql/01_vintage_analysis.sql](sql/01_vintage_analysis.sql).

**Roll rate analysis:** Tracks month-over-month transition rates between
delinquency states (e.g. 30 DPD to 60 DPD), normalized within each
starting state. See [sql/02_roll_rate.sql](sql/02_roll_rate.sql).

## AI-Augmented Development

<!-- TODO: move this section to the end of the file, after the
     Business Problem / Data & Methodology / Key Findings sections are
     written, so it's not the first thing after the title. -->

This project was built independently, using AI coding agents as a
productivity and quality-assurance tool. I made every substantive
decision: the data source, the variable selection (validated against
LendingClub's official data dictionary), the grade-balanced sampling
design, the decision to simulate monthly delinquency via a
grade-calibrated Markov chain (disclosed in `docs/charter.md`), and the
analytical methods used — vintage analysis, roll-rate analysis, and a
logistic regression PD model.

AI agents also served as a review layer: working independently, without
a manager or code reviewer, I used them to catch things I might
otherwise miss — for example, a step that looked complete but wasn't, or
a `.gitignore` rule silently excluding a file I needed tracked. Every
change was reviewed and approved by me before being committed; these
working principles are documented in `CLAUDE.md`.
