-- Month-over-month roll rate between delinquency states.
--
-- For loans in dpd30/dpd60/dpd90, computes the distribution of next-
-- month states (roll forward, cure, stay, or default), normalized to
-- 100% within each from_state. A NULL to_state means the loan's
-- simulated observation ended right after that state (its term ended
-- while still delinquent, without curing or defaulting) - a real,
-- right-censored outcome, not missing data.
--
-- Run against data/processed/loan_monthly_performance.csv via DuckDB.

WITH ordered AS (
    SELECT loan_id, snapshot_month, delinquency_state,
           LEAD(delinquency_state) OVER (
               PARTITION BY loan_id ORDER BY snapshot_month
           ) AS next_state
    FROM 'data/processed/loan_monthly_performance.csv'
)
SELECT delinquency_state AS from_state, next_state AS to_state,
       COUNT(*) AS n_accounts,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY delinquency_state), 2)
           AS roll_rate_pct
FROM ordered
WHERE delinquency_state IN ('dpd30', 'dpd60', 'dpd90')
GROUP BY delinquency_state, next_state
ORDER BY delinquency_state, next_state;
