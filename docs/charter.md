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

**Coefficient interpretation and business conclusion:** In this model,
credit grade is the dominant predictor of the 12-month default target:
risk rises sharply and statistically significantly from grade C through
G, and none of the other attributes collected (income, debt-to-income
ratio, FICO score, loan amount, employment length) add statistically
significant independent information once grade is included. Interest
rate behaves the same way: on its own it is a strong, highly significant
predictor, but its effect collapses to a small, statistically
insignificant residual once grade is added.

**Caveat — this reflects the simulation design, not real borrower
behavior.** The target comes from the simulated delinquency panel, in
which the only loan attributes that matter are grade (the monthly hazard
of moving from current to 30 days past due) and term (the horizon); all
other transition probabilities are the same for every loan. Grade is
therefore expected to dominate, and the other attributes can enter the
model only through their correlation with grade. The interest-rate
result is the same effect: interest rate is correlated 0.96 with grade in
the real LendingClub data, so it carries no signal beyond grade in an
outcome that depends on grade alone. The one significant purpose
coefficient (home improvement, p = 0.042) is not a causal finding
either: purpose plays no role in the simulation, so a significant result
for it can come only from correlation with grade or from chance, and
about one significant result is expected by chance among the roughly 20
coefficients tested at the 5% level. What this exercise demonstrates is
the modeling workflow — a coefficient table with significance tests and a
redundancy check between correlated variables — not a statement about
what drives defaults in LendingClub's actual portfolio. Conclusions about
real portfolio risk would require real monthly performance data.

## Success Metrics (3)
1. **Default/delinquency rate trend** — monthly (simulated) 30/60/90+ DPD
   rate, segmented by grade and purpose.
2. **Vintage curve deterioration** — cumulative default rate by loan age
   per origination cohort, flagging cohorts trending worse than prior
   ones.
3. **Roll rate stability** — early-stage transition rates (current → 30
   DPD → 60 DPD) tracked over time as a leading indicator of portfolio
   deterioration.
