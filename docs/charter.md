# Project Charter — Credit Risk Portfolio Monitoring

## Business Question
Is the quality of the loan portfolio deteriorating over time, and if so,
in which segments (loan grade, purpose, and other borrower/loan
attributes)?

## Data Source
Lending Club public dataset (Kaggle) — loan-level records for unsecured
consumer/peer-to-peer installment loans, including amount, interest rate,
grade, purpose, income, DTI, FICO range, and issue date. This is real,
publicly issued loan data, not simulated.

**Disclosed limitation:** the raw data contains no monthly delinquency
panel — only loan-level outcomes, not a month-by-month payment history.
To support vintage and roll-rate analysis, monthly delinquency status is
simulated using a Markov chain, with transition (hazard) probabilities
calibrated by loan grade — lower grades carry a higher simulated monthly
probability of delinquency. This is a deliberate, disclosed design choice
to make the monitoring methodology demonstrable in the absence of a public
monthly panel for consumer loans — not an attempt to pass off simulated
data as observed.

## Dashboard Audience
Risk management — portfolio monitoring for credit risk decisions
(underwriting, reserving, collections focus). Not retail sales or
marketing; no model-internals audience, just decision-ready portfolio
signals.

## Core Methods
1. **Vintage analysis** — cumulative default/delinquency rate by loan age,
   compared across origination cohorts.
2. **Roll rate analysis** — month-over-month transition rates between
   delinquency buckets, using the simulated delinquency panel.
3. **Logistic regression PD model** — probability of default estimated
   from loan-level attributes (grade, purpose, income, DTI, FICO range,
   home ownership, employment length), used to validate that simulated
   delinquency behavior tracks expected risk ordering.

**Note on modeling approach:** A full bank-grade PD scorecard would
typically use coarse classing and WOE (Weight of Evidence) transformation
of features instead of raw values with StandardScaler — this gives
outlier robustness and per-bin explainability. We are not doing that
here: at 3,500 loans, and an even narrower set of 12-month defaults, WOE
bins would have too few default events per bin to be statistically
stable — WOE is designed for the much larger volumes real banks have.
Capping extreme values plus standard logistic regression is the honest
choice for this sample size.

**Coefficient interpretation and business conclusion:** The single
dominant driver of 12-month default risk in this model is LendingClub's
own credit grade — risk rises sharply and statistically significantly
from grade C through G, while none of the raw financial attributes we
collected (income, debt-to-income ratio, FICO score, loan amount, or
employment length) add statistically significant independent risk
information once grade is accounted for at this sample size; the one
secondary significant effect is that home-improvement loans carry
meaningfully lower risk than other loan purposes, holding grade
constant. Critically, interest rate is not an independent risk driver
alongside grade: on its own it is a strong, highly significant predictor
of default, but its effect collapses to a small, statistically
insignificant residual once grade is added to the model. This is
expected, not a modeling flaw — LendingClub sets interest rate largely
as a direct function of the grade it assigns, so the two variables
capture overlapping information rather than two independent risk
signals. For portfolio monitoring purposes, this means grade should be
treated as the primary, decision-driving risk lever, while interest rate
is better understood as a pricing consequence of that same underlying
assessment than as an additional early-warning signal.

## Success Metrics (3)
1. **Default/delinquency rate trend** — monthly (simulated) 30/60/90+ DPD
   rate, segmented by grade and purpose.
2. **Vintage curve deterioration** — cumulative default rate by loan age
   per origination cohort, flagging cohorts trending worse than prior
   ones.
3. **Roll rate stability** — early-stage transition rates (current → 30
   DPD → 60 DPD) tracked over time as a leading indicator of portfolio
   deterioration.
