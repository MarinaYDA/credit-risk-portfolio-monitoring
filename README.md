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

## Real vs simulated data

**Real:** loan-level attributes from LendingClub's public dataset —
grade, term, interest rate, loan amount, purpose, home ownership, annual
income, DTI, FICO range, employment length, issue date — for a
grade-balanced sample of 3,500 loans drawn from about 2.26M.

**Simulated:** everything about how loans perform after issue. The raw
data has no monthly delinquency history, so
[scripts/generate_loan_performance.py](scripts/generate_loan_performance.py)
simulates each loan's monthly state (current, 30/60/90 days past due,
default, paid off) with a Markov chain. Everything built on that panel is
simulated too: the vintage curves, the roll rates, the 12-month default
target (`defaulted_12m`), the model's results, and the Power BI default
measures.

**What the simulation depends on:** only the loan's grade and its term.
Grade sets the monthly probability of moving from current to 30 days past
due (0.6% for grade A, rising to 6.5% for G — illustrative values, not
fitted to data); the term sets how many months the loan can run. Every
other transition probability (early payoff, and the roll-forward and cure
rates between delinquency stages) is the same for every loan. Issue date
only places a loan on the calendar. FICO range, DTI, income, purpose,
home ownership, employment length, loan amount and interest rate do not
enter the simulation.

**What follows from that:**
- The rise in defaults from grade A (5.0% of loans reach default over
  their term) to grade G (41.4%) is built into the simulation, not
  observed.
- In the PD model, grade dominates and the other attributes add nothing
  significant because the target depends on grade only; they can show up
  only through their correlation with grade. Interest rate is correlated
  0.96 with grade in the real data, which is why it adds nothing
  alongside it. The model's AUC mostly reflects how well grade, and
  attributes correlated with it, predict a simulated outcome; it is not
  evidence about real borrower behavior.
- **The Risk score is uncalibrated.** The model is trained with balanced
  class weights, and its mean score across the 3,496 scored loans is
  0.4252, against an observed 12-month default rate of 4.63%. It ranks
  risk; it is not a probability of default. Calibration is a next step.
- **Year-over-year comparisons are not a real trend.** The Power BI
  year-over-year measures compare defaults in one calendar year with the
  previous year, using a fixed sample of 3,500 loans that ages through
  time (loans originated 2007–2018 are observed until they mature, pay
  off or default) and a simulated outcome. The decline in defaults after
  2017 reflects the sample aging, not falling risk.

## Model

**Logistic regression PD model:** Trained on the 12-month default target
using loan-level attributes as features (see
[scripts/pd_model.py](scripts/pd_model.py)). On a held-out 25% test set,
the model achieves **ROC-AUC 0.7120** and a **KS statistic of 0.3984**,
at threshold 0.6018 — the KS-maximizing point, chosen because the model
is trained with `class_weight="balanced"`, which makes 0.5 not a
meaningful default threshold here. Confusion matrix at that threshold:
TN 645, FP 189, FN 15, TP 25. ROC curve:
[docs/img/roc_curve.png](docs/img/roc_curve.png).

**Coefficient interpretation:** In this model, credit grade is the
dominant, statistically significant predictor; none of the other
attributes (income, DTI, FICO range, loan amount, employment length) add
significant independent signal once grade is included, and interest rate
adds nothing alongside grade. **This reflects the simulation design, not
real borrower behavior:** the simulated delinquency depends only on grade
(plus term as the horizon), so grade is expected to dominate, and the
other attributes can enter only through their correlation with grade —
interest rate is correlated 0.96 with grade in the real data. See
[Real vs simulated data](#real-vs-simulated-data) and the full
coefficient discussion in [docs/charter.md](docs/charter.md).

**Per-loan scores:**
[data/processed/loan_level_scored.csv](data/processed/loan_level_scored.csv)
holds each loan's model score (3,496 loans; the 4 loans with missing DTI
are not scored).

**Power BI status:** the dashboard file is
[powerbi/credit_risk_dashboard.pbix](powerbi/credit_risk_dashboard.pbix)
(open it with Power BI Desktop). It contains the data model (loans,
monthly performance, a date table, and the vintage-curve and roll-rate
extracts), nine measures and two calculated columns, and the Headline
page: summary cards, default rate and uncalibrated risk score by grade,
and grade and vintage-year slicers. The model and the Headline page were
checked against DuckDB reference values. Still to come: the Vintage and
roll rate page, screenshots, and a Results section in this README.

The data is imported into the file, so it opens and works without any
setup. To refresh the data from your clone, set the `DataFolder`
parameter to your clone's `data/processed/` folder, then refresh: in
Power BI Desktop choose Home → Transform data → Edit parameters, enter
the full path ending in a backslash (for example
`C:\Users\you\credit-risk-portfolio-monitoring\data\processed\`), click
OK and then Apply changes. The file ships with the placeholder
`C:\path\to\credit-risk-portfolio-monitoring\data\processed\`, so a
refresh fails until you change it. All five CSV queries read from
`DataFolder`; none contains a machine-specific path.

This is a methodology demonstration on a simulated 12-month default
target, not a production credit model — see
[Real vs simulated data](#real-vs-simulated-data) and the disclosed
limitation in [docs/charter.md](docs/charter.md).

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
