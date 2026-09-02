-- Cumulative default rate by cohort age (vintage curve).
--
-- For each origination cohort (vintage_month), tracks what percentage
-- of that cohort's original loan count has ever reached "default" by
-- each loan age (months_on_book). Uses a fixed cohort-size denominator
-- (loan count at age 1) and a running cumulative sum of defaults by
-- age, rather than a shrinking survivor-only denominator, so the rate
-- is monotonically non-decreasing with age, as a cumulative default
-- rate should be.
--
-- Run against data/processed/loan_monthly_performance.csv via DuckDB.

WITH cohort_size AS (
    SELECT vintage_month, COUNT(DISTINCT loan_id) AS cohort_loans
    FROM 'data/processed/loan_monthly_performance.csv'
    WHERE months_on_book = 1
    GROUP BY vintage_month
),
marginal_defaults AS (
    SELECT vintage_month, months_on_book,
           COUNT(DISTINCT loan_id) AS defaults_at_age
    FROM 'data/processed/loan_monthly_performance.csv'
    WHERE delinquency_state = 'default'
    GROUP BY vintage_month, months_on_book
),
max_age_by_vintage AS (
    SELECT vintage_month, MAX(months_on_book) AS max_age
    FROM 'data/processed/loan_monthly_performance.csv'
    GROUP BY vintage_month
),
age_grid AS (
    SELECT m.vintage_month, gs.age AS months_on_book
    FROM max_age_by_vintage m, generate_series(1, m.max_age) AS gs(age)
),
filled AS (
    SELECT
        g.vintage_month,
        g.months_on_book,
        COALESCE(d.defaults_at_age, 0) AS defaults_at_age
    FROM age_grid g
    LEFT JOIN marginal_defaults d
        ON g.vintage_month = d.vintage_month
       AND g.months_on_book = d.months_on_book
)
SELECT
    f.vintage_month,
    f.months_on_book,
    c.cohort_loans,
    SUM(f.defaults_at_age) OVER (
        PARTITION BY f.vintage_month ORDER BY f.months_on_book
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_defaults,
    ROUND(
        100.0 * SUM(f.defaults_at_age) OVER (
            PARTITION BY f.vintage_month ORDER BY f.months_on_book
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / NULLIF(c.cohort_loans, 0),
    2) AS cum_default_rate_pct
FROM filled f
JOIN cohort_size c ON f.vintage_month = c.vintage_month
ORDER BY f.vintage_month, f.months_on_book;
