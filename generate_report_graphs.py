"""
Generates a handful of PNG report graphics into output/ -- useful for a
pitch deck or README screenshots: confusion matrix, score distribution,
feature importances, and a fairness (approval rate by group) chart.

Usage: python generate_report_graphs.py

# Updated on 2026-02-18
"""
import os
import sqlite3

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import config
import fairness

OUTPUT_DIR = "output"
sns.set_theme(style="darkgrid")


def load_applicants_and_model():
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM applicants", conn)
    finally:
        conn.close()
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    medians = joblib.load("impute_medians.joblib")
    return df, model, scaler, feature_cols, medians


def plot_confusion_matrix(df, model, scaler, feature_cols, medians):
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].fillna(medians.get(col, 0.0))
    y = df["repayment_outcome"]
    y_pred = model.predict(scaler.transform(X))

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Poor repayment (0)", "Good repayment (1)"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (full population)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_score_distribution():
    conn = sqlite3.connect(config.DB_NAME)
    try:
        scores = pd.read_sql_query("SELECT credit_score, confidence_label FROM credit_scores", conn)
    finally:
        conn.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=scores, x="credit_score", hue="confidence_label", bins=30, ax=ax, multiple="stack")
    ax.axvline(config.APPROVAL_SCORE_THRESHOLD, color="red", linestyle="--", label="Approval threshold")
    ax.set_title("Credit Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "score_distribution.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_cols):
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda p: p[1])[-12:]
    labels, values = zip(*importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, values, color="#2563eb")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_fairness():
    report = fairness.run_fairness_audit()
    fig, axes = plt.subplots(1, len(report["groups"]), figsize=(5 * len(report["groups"]), 4))
    for ax, (group_col, rows) in zip(axes, report["groups"].items()):
        colors = ["#dc3545" if r["flagged"] else "#28a745" for r in rows]
        ax.barh([r["group"] for r in rows], [r["approval_rate"] * 100 for r in rows], color=colors)
        ax.set_title(f"Approval rate by {group_col}")
        ax.set_xlabel("%")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fairness_by_group.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, model, scaler, feature_cols, medians = load_applicants_and_model()
    print("Generating confusion matrix...")
    plot_confusion_matrix(df, model, scaler, feature_cols, medians)
    print("Generating score distribution...")
    plot_score_distribution()
    print("Generating feature importance...")
    plot_feature_importance(model, feature_cols)
    print("Generating fairness chart...")
    plot_fairness()
    print(f"Done. See ./{OUTPUT_DIR}/")
# Updated on 2026-02-18
