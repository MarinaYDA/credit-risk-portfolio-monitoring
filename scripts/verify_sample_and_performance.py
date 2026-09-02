"""
Cross-check the grade-balanced sample against the simulated monthly
delinquency panel, using DuckDB to query both CSVs directly (no server,
no load step).

Verifies: the sample still has exactly 500 loans per grade; no loan was
dropped or duplicated while generating the monthly panel; and the row
counts per grade in the monthly panel make sense once normalized for
term length (36 vs 60 months), rather than assuming the raw counts are
fine without checking.
"""

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = (REPO_ROOT / "data" / "raw" / "lending_club_sample.csv").as_posix()
MONTHLY_PATH = (
    REPO_ROOT / "data" / "processed" / "loan_monthly_performance.csv"
).as_posix()


def main() -> None:
    con = duckdb.connect()

    print("=== Check 1a: loan count by grade (lending_club_sample.csv) ===")
    print(
        con.sql(
            f"""
            SELECT grade, COUNT(*) AS loan_count
            FROM '{SAMPLE_PATH}'
            GROUP BY grade
            ORDER BY grade
            """
        )
        .df()
        .to_string(index=False)
    )

    print("\n=== Check 1b: loan count by issue month (lending_club_sample.csv) ===")
    issue_month_counts = con.sql(
        f"""
        SELECT strptime(issue_d, '%b-%Y') AS issue_month, COUNT(*) AS loan_count
        FROM '{SAMPLE_PATH}'
        GROUP BY issue_month
        ORDER BY issue_month
        """
    ).df()
    print(f"Distinct issue months: {len(issue_month_counts)}")
    print(
        "Range:",
        issue_month_counts["issue_month"].min(),
        "to",
        issue_month_counts["issue_month"].max(),
    )
    print(
        "Min/max loans in a single month:",
        issue_month_counts["loan_count"].min(),
        "/",
        issue_month_counts["loan_count"].max(),
    )

    print("\n=== Check 2: distinct loan_id count (loan_monthly_performance.csv) ===")
    print(
        con.sql(
            f"""
            SELECT COUNT(DISTINCT loan_id) AS distinct_loans
            FROM '{MONTHLY_PATH}'
            """
        )
        .df()
        .to_string(index=False)
    )

    print("\n=== Check 3: row count per grade in loan_monthly_performance.csv ===")
    print(
        con.sql(
            f"""
            SELECT
                s.grade,
                COUNT(*) AS monthly_row_count,
                COUNT(DISTINCT m.loan_id) AS loan_count,
                ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT m.loan_id), 2) AS avg_months_per_loan
            FROM '{MONTHLY_PATH}' m
            JOIN '{SAMPLE_PATH}' s ON m.loan_id = s.loan_id
            GROUP BY s.grade
            ORDER BY s.grade
            """
        )
        .df()
        .to_string(index=False)
    )

    print("\n=== Supporting context for check 3: term (36 vs 60 months) mix by grade ===")
    print(
        con.sql(
            f"""
            SELECT grade, term, COUNT(*) AS loan_count
            FROM '{SAMPLE_PATH}'
            GROUP BY grade, term
            ORDER BY grade, term
            """
        )
        .df()
        .to_string(index=False)
    )

    print("\n=== Check 3 (normalized): pct of available term consumed, by grade ===")
    print(
        con.sql(
            f"""
            -- Term parsing here must be kept in sync with
            -- parse_term_months() in generate_loan_performance.py.
            WITH term_parsed AS (
                SELECT loan_id, grade,
                       CAST(regexp_extract(term, '(\\d+)', 1) AS INTEGER) AS term_months
                FROM '{SAMPLE_PATH}'
            ),
            months_used AS (
                SELECT loan_id, MAX(months_on_book) AS months_on_book
                FROM '{MONTHLY_PATH}'
                GROUP BY loan_id
            )
            SELECT
                t.grade,
                ROUND(AVG(t.term_months), 2) AS avg_term_months,
                ROUND(AVG(u.months_on_book), 2) AS avg_months_used,
                ROUND(AVG(u.months_on_book) * 100.0 / AVG(t.term_months), 1) AS pct_term_consumed
            FROM term_parsed t
            JOIN months_used u ON t.loan_id = u.loan_id
            GROUP BY t.grade
            ORDER BY t.grade
            """
        )
        .df()
        .to_string(index=False)
    )

    con.close()


if __name__ == "__main__":
    main()
