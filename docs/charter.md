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
   from loan-level attributes (grade, purpose, income, DTI, FICO range),
   used to validate that simulated delinquency behavior tracks expected
   risk ordering.

## Success Metrics (3)
1. **Default/delinquency rate trend** — monthly (simulated) 30/60/90+ DPD
   rate, segmented by grade and purpose.
2. **Vintage curve deterioration** — cumulative default rate by loan age
   per origination cohort, flagging cohorts trending worse than prior
   ones.
3. **Roll rate stability** — early-stage transition rates (current → 30
   DPD → 60 DPD) tracked over time as a leading indicator of portfolio
   deterioration.
