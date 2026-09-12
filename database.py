# database.py
import sqlite3
import config


def create_database():
    """Initializes the database and creates/updates tables."""
    conn = None
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()
        print(f"Successfully connected to database: {config.DB_NAME}")

        cursor.execute("DROP TABLE IF EXISTS credit_scores")
        print("Dropped old 'credit_scores' table (if it existed).")

        # 'applicants' is created/replaced by data_ingestion.py (pandas to_sql),
        # this just guarantees the users table (auth) and scores table exist.
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE credit_scores (
            applicant_id TEXT PRIMARY KEY,
            credit_score REAL,
            probability_good REAL,
            confidence_label TEXT,
            band_low REAL,
            band_high REAL,
            data_completeness REAL,
            reason_codes TEXT,
            guardrail_flags TEXT,
            improvement_path TEXT,
            shap_factors TEXT,
            scored_at TEXT
        )
        """)
        print("Table 'credit_scores' created.")

        # --- Loan lifecycle ---------------------------------------------
        # What happens after a score becomes a decision: disbursal, an EMI
        # schedule, and the repayment behaviour that flows back into the
        # borrower's next assessment.
        cursor.execute("DROP TABLE IF EXISTS loans")
        cursor.execute("DROP TABLE IF EXISTS repayments")

        cursor.execute("""
        CREATE TABLE loans (
            loan_id TEXT PRIMARY KEY,
            applicant_id TEXT NOT NULL,
            principal REAL,
            tenure_months INTEGER,
            annual_rate REAL,
            monthly_emi REAL,
            status TEXT,
            months_elapsed INTEGER DEFAULT 0,
            -- the assessment as it stood at funding time, frozen so the
            -- portfolio backtest can compare prediction against outcome
            score_at_funding INTEGER,
            prob_good_at_funding REAL,
            disbursed_at TEXT,
            closed_at TEXT
        )
        """)
        print("Table 'loans' created.")

        cursor.execute("""
        CREATE TABLE repayments (
            loan_id TEXT NOT NULL,
            instalment_no INTEGER NOT NULL,
            amount_due REAL,
            status TEXT,
            due_month INTEGER,
            paid_at TEXT,
            PRIMARY KEY (loan_id, instalment_no)
        )
        """)
        print("Table 'repayments' created.")

        # --- Credit Marketplace ------------------------------------------------
        # Lets lenders browse and co-fund vetted, scored applicants.
        # Solving PS #4's core access-to-credit gap by creating competition
        # among lenders for verified creditworthy borrowers.
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS marketplace_listings (
            listing_id       TEXT PRIMARY KEY,
            applicant_id     TEXT NOT NULL,
            listed_by        TEXT NOT NULL,
            listed_at        TEXT NOT NULL,
            amount_requested REAL NOT NULL,
            tenure_months    INTEGER NOT NULL,
            credit_score     INTEGER NOT NULL,
            tier_label       TEXT,
            entity_type      TEXT,
            business_type    TEXT,
            geography_tier   TEXT,
            purpose          TEXT,
            status           TEXT DEFAULT 'open',
            total_committed  REAL DEFAULT 0.0,
            fully_funded_at  TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lender_interests (
            interest_id       TEXT PRIMARY KEY,
            listing_id        TEXT NOT NULL,
            lender_username   TEXT NOT NULL,
            committed_amount  REAL NOT NULL,
            status            TEXT DEFAULT 'active',
            created_at        TEXT NOT NULL
        )
        """)
        print("Tables 'marketplace_listings' and 'lender_interests' created.")

        conn.commit()
        print("Database setup complete.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    create_database()
# Updated on 2026-02-18
