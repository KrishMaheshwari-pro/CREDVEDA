# database_setup.py
import sqlite3
import config

def create_database():
    """Initializes the database and creates/updates tables."""
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()
        print(f"Successfully connected to database: {config.DB_NAME}")

        # --- Drop existing tables to ensure schema is updated ---
        # In a real production environment, you would use a migration tool.
        # For this hackathon, dropping and recreating is the simplest way.
        cursor.execute("DROP TABLE IF EXISTS historical_features")
        cursor.execute("DROP TABLE IF EXISTS credit_scores")
        print("Dropped old tables (if they existed).")


        # --- Create historical_features table with new NLP columns ---
        event_columns = " ".join([f"{key} INTEGER," for key in config.EVENT_KEYWORDS.keys()])
        create_features_table_sql = f"""
        CREATE TABLE historical_features (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL,
            Volume REAL, Return REAL, Volatility REAL, SMA_Short REAL, SMA_Long REAL,
            TrendRatio REAL, RSI REAL, MACD REAL, Liquidity REAL, PE REAL, PB REAL,
            PEG REAL, DebtToEquity REAL, MarketCap REAL, Beta REAL, ProfitMargin REAL,
            MacroIndicator REAL, NewsSentiment REAL, TranscriptSentiment REAL,
            {event_columns}
            PRIMARY KEY (Date, Ticker)
        )
        """
        cursor.execute(create_features_table_sql)
        print("Table 'historical_features' created with new NLP schema.")

        # --- Create credit_scores table with a column for trigger events ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS credit_scores (
            Date TEXT, Ticker TEXT, CreditScore REAL, TopReasons TEXT,
            TriggerEvent TEXT,  -- This new column will store the event snippet
            PRIMARY KEY (Date, Ticker)
        )
        """)
        print("Table 'credit_scores' created with TriggerEvent column.")

        conn.commit()
        print("Database setup complete.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_database()
