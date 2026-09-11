"""
Seeds a demonstration loan book so the platform opens with a live portfolio
rather than an empty table.

Loans are deliberately spread across the whole score range, including
applicants below the approval cut-off. That is not an oversight -- it is how
a lender escapes the *reject inference* problem. If you only ever fund people
you approved, you only ever observe outcomes for the approved, and you can
never prove the cut-off was in the right place. Real alternative-data lenders
run exactly this kind of small test-and-learn book across the score range to
find out what the population below their cut-off actually does.

That spread is also what makes the portfolio backtest meaningful: it produces
outcomes at every predicted-probability level, so the chart on the portfolio
page can show whether observed repayment really does track the prediction.

Usage: python seed_portfolio.py [n_loans] [months]

# Updated on 2026-02-18
"""
import random
import sqlite3
import sys

import pandas as pd

import config
import lifecycle

DEFAULT_LOANS = 140
DEFAULT_MONTHS = 9


def clear_book():
    conn = sqlite3.connect(config.DB_NAME)
    try:
        conn.execute("DELETE FROM repayments")
        conn.execute("DELETE FROM loans")
        conn.commit()
    finally:
        conn.close()


def seed(n_loans: int = DEFAULT_LOANS, months: int = DEFAULT_MONTHS):
    rng = random.Random(config.RANDOM_SEED)
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query("""
            SELECT a.applicant_id, a.requested_loan_amount, a.requested_tenure_months,
                   s.credit_score, s.probability_good, s.guardrail_flags
            FROM applicants a JOIN credit_scores s ON s.applicant_id = a.applicant_id
        """, conn)
    finally:
        conn.close()

    # Never fund through a binding affordability guardrail, even in a pilot.
    df = df[~df["guardrail_flags"].str.contains('"critical"', na=False)]

    # Stratify across predicted-probability bands so the book covers the range.
    bands = [(0.0, 0.45), (0.45, 0.60), (0.60, 0.75), (0.75, 1.01)]
    weights = [0.15, 0.25, 0.35, 0.25]   # weighted toward, but not only, good risk
    picked = []
    for (lo, hi), w in zip(bands, weights):
        pool = df[(df["probability_good"] >= lo) & (df["probability_good"] < hi)]
        take = min(int(n_loans * w), len(pool))
        if take:
            picked.append(pool.sample(take, random_state=config.RANDOM_SEED))
    if not picked:
        print("No eligible applicants -- run scoring.py first.")
        return
    book = pd.concat(picked)

    print(f"Clearing any existing book, then disbursing {len(book)} pilot loans...")
    clear_book()
    for _, r in book.iterrows():
        # Vary ticket size a little so the book doesn't look machine-stamped.
        principal = float(r.requested_loan_amount) * rng.uniform(0.6, 1.0)
        lifecycle.disburse(r.applicant_id, round(principal, -3), int(r.requested_tenure_months),
                           int(r.credit_score), float(r.probability_good))

    print(f"Advancing the book {months} months...")
    summary = lifecycle.advance_all(months)
    print(f"  {summary['instalments']} instalments due: {summary['paid']} on time, "
          f"{summary['late']} late, {summary['missed']} missed")
    print(f"  {summary['closed']} loans fully repaid, {summary['defaulted']} written off")

    stats = lifecycle.portfolio_stats()
    print(f"\nBook: Rs {stats['disbursed']:,.0f} disbursed, Rs {stats['collected']:,.0f} collected, "
          f"on-time rate {stats['on_time_rate']:.1%}, default rate {stats['default_rate']:.1%}")
    print("\nPredicted vs actual by probability band:")
    for b in lifecycle.backtest_buckets():
        print(f"  {b['label']:>10}  n={b['n_loans']:<4} predicted {b['predicted']:.0%}  "
              f"actual on-time {b['actual_on_time']:.0%}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOANS
    m = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MONTHS
    seed(n, m)
# Updated on 2026-02-18
