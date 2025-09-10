
# scoring.py
import sqlite3
import pandas as pd
import numpy as np
import joblib
import shap
import config
from datetime import datetime
import requests
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys # --- NEW: To read command-line arguments

# --- Helper functions (get_latest_trigger_event, etc.) remain the same ---
def analyze_text_for_events(text: str, analyzer: SentimentIntensityAnalyzer):
    sentences = re.split(r'(?<!\w\w.)(?<![A-Z][a-z].)(?<=\.|\?)\s', text)
    trigger_snippet = ""
    text_lower = text.lower()
    for event_type, keywords in config.EVENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                try:
                    snippet_sentence = next(s for s in sentences if keyword in s.lower())
                    trigger_snippet = f"[{event_type.replace('_', ' ')}]: ...{snippet_sentence.strip()}..."
                    if trigger_snippet: break
                except StopIteration: pass
        if trigger_snippet: break
    return trigger_snippet

def get_latest_trigger_event(ticker: str):
    if ticker.endswith(".NS"): return "N/A (Indian Market Stock)"
    if not hasattr(config, 'API_NINJAS_KEY') or not config.API_NINJAS_KEY or config.API_NINJAS_KEY == "YOUR_API_NINJAS_KEY_HERE": return "N/A (API key not set)"
    try:
        api_url = f"https://api.api-ninjas.com/v1/earningstranscript?ticker={ticker.split('.')[0]}"
        response = requests.get(api_url, headers={'X-Api-Key': config.API_NINJAS_KEY})
        response.raise_for_status()
        data = response.json()
        if not data or 'transcript' not in data: return "No recent transcript found."
        transcript_text = data.get('transcript', '')
        analyzer = SentimentIntensityAnalyzer()
        snippet = analyze_text_for_events(transcript_text, analyzer)
        return snippet if snippet else "No specific events detected."
    except Exception:
        return "Error fetching transcript."

def generate_scores():
    """Loads the latest data and a specified trained model to generate and save credit scores."""
    # --- NEW: Choose model based on command-line argument ---
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ['rf', 'xgb']:
        print("Usage: python scoring.py [rf|xgb]")
        print("  rf: RandomForest")
        print("  xgb: XGBoost")
        return
    
    model_choice = sys.argv[1].lower()
    print(f"--- Starting scoring process with {model_choice.upper()} model ---")

    # --- NEW: Load model files with specific names ---
    model_filename = f'model_{model_choice}.joblib'
    scaler_filename = f'scaler_{model_choice}.joblib'
    features_filename = f'feature_cols_{model_choice}.joblib'

    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        feature_cols = joblib.load(features_filename)
        print("Model, scaler, and feature list loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Model files for '{model_choice}' not found. Please run 'python model_training.py {model_choice}' first.")
        return

    # --- The rest of the script is the same ---
    try:
        conn = sqlite3.connect(config.DB_NAME)
        query = "SELECT * FROM historical_features WHERE (Ticker, Date) IN (SELECT Ticker, MAX(Date) FROM historical_features GROUP BY Ticker)"
        latest_df = pd.read_sql_query(query, conn, index_col=['Date', 'Ticker'])
        print(f"Loaded latest feature data for {len(latest_df)} tickers.")
    except Exception as e:
        print(f"Could not load data from database: {e}"); return
    finally:
        if conn: conn.close()

    if latest_df.empty:
        print("[ERROR] No data found in the database. Please run data_ingestion.py."); return

    X = latest_df[feature_cols].copy()
    X.fillna(X.median(), inplace=True); X.fillna(0, inplace=True)
    X_scaled = scaler.transform(X)

    prob_good = model.predict_proba(X_scaled)[:, 1]
    credit_scores = (prob_good * 100).round(2)

    print("Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values_output = explainer.shap_values(X_scaled)

    # This logic correctly handles the output from the explainer for both model types.
    if isinstance(shap_values_output, list):
        # RandomForest returns a list of two arrays [class_0, class_1]
        shap_for_good = shap_values_output[1]
    else:
        # XGBoost returns a single array for the positive class
        shap_for_good = shap_values_output

    reasons = []
    for i in range(X.shape[0]):
        row_contrib = shap_for_good[i]
        pairs = sorted(list(zip(feature_cols, row_contrib)), key=lambda p: abs(p[1]), reverse=True)
        reason_parts = [f"{f} ({'+' if v >= 0 else ''}{v:.2f})" for f, v in pairs[:3]]
        reasons.append(" | ".join(reason_parts))

    print("Fetching trigger events for scored tickers...")
    tickers_list = X.index.get_level_values('Ticker')
    trigger_events = [get_latest_trigger_event(ticker) for ticker in tickers_list]

    scores_df = pd.DataFrame({
        'Ticker': tickers_list,
        'CreditScore': credit_scores,
        'TopReasons': reasons,
        'TriggerEvent': trigger_events,
        'Date': datetime.now().strftime('%Y-%m-%d')
    }).set_index(['Date', 'Ticker'])
    
    print("\nGenerated Scores Preview:")
    print(scores_df.head())

    try:
        conn = sqlite3.connect(config.DB_NAME)
        scores_df.to_sql('credit_scores', conn, if_exists='replace', index=True)
        print("\nScores saved successfully to the database.")
    except sqlite3.Error as e:
        print(f"Database error during save: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    generate_scores()
