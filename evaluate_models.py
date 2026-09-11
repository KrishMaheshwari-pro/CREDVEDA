"""
Standalone evaluation report for the trained scoring model: cross-validated
ROC-AUC/accuracy on the applicants table, plus a fairness summary. Run this
after model_training.py and scoring.py to get a single "model card" printout.

Usage: python evaluate_models.py

# Updated on 2026-02-18
"""
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
import fairness


def load_data():
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM applicants", conn)
    finally:
        conn.close()
    return df


def evaluate_model():
    print("=== Model Evaluation ===")
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    medians = joblib.load("impute_medians.joblib")

    df = load_data()
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].fillna(medians.get(col, 0.0))
    y = df["repayment_outcome"]
    X_scaled = scaler.transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"5-fold ROC-AUC: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")
    print(f"5-fold Accuracy: {acc_scores.mean():.3f} (+/- {acc_scores.std():.3f})")

    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    print(f"\nFull-population ROC-AUC: {roc_auc_score(y, y_pred_proba):.3f}")
    print(classification_report(y, y_pred, digits=3))

    print("\nTop feature importances:")
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda p: p[1], reverse=True)
    for feat, imp in importances[:8]:
        print(f"  {feat:<28} {imp:.3f}")


def evaluate_fairness():
    print("\n=== Fairness Summary ===")
    try:
        report = fairness.run_fairness_audit()
    except Exception as e:
        print(f"Could not run fairness audit (has scoring.py been run yet?): {e}")
        return
    if report["flagged_disparities"]:
        print(f"{len(report['flagged_disparities'])} disparit(y/ies) flagged under the four-fifths rule:")
        for line in report["flagged_disparities"]:
            print(f"  - {line}")
    else:
        print("No group failed the four-fifths adverse-impact rule.")


if __name__ == "__main__":
    evaluate_model()
    evaluate_fairness()
# Updated on 2026-02-18
