"""
Trains and compares the candidate scoring models, then saves the production
model plus everything the app needs to explain and audit it.

Three candidates are compared on the metrics credit-risk teams actually
report -- Gini and the KS statistic, not just accuracy -- along with Brier
score, which measures whether the predicted probabilities are *calibrated*
rather than merely well-ranked. A model can rank applicants perfectly and
still be badly calibrated, and for lending the calibration is what lets you
price risk.

Production model selection is deliberately NOT "whichever scores highest".
PS #4 requires an explanation for every individual decision, and the Random
Forest is the candidate that supports an exact additive decomposition of a
single prediction (see scoring.forest_explain). A model that wins by a
fraction of a point of AUC but can only offer approximate explanations is
the wrong trade for this problem, and that choice is recorded below rather
than hidden.

Protected attributes (gender, geography, business type) are excluded from
the feature set entirely -- they exist only for the fairness audit.

# Updated on 2026-02-18
"""
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

MODEL_FILENAME = "model.joblib"
SCALER_FILENAME = "scaler.joblib"
FEATURES_FILENAME = "feature_cols.joblib"
IMPUTE_FILENAME = "impute_medians.joblib"
PERCENTILES_FILENAME = "feature_percentiles.joblib"
ANOMALY_FILENAME = "anomaly_model.joblib"
LEADERBOARD_FILENAME = "model_leaderboard.joblib"

# The production model must support exact per-prediction decomposition.
EXPLAINABLE_MODEL = "Random Forest"


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov separation: the widest gap between the cumulative
    distributions of good and bad borrowers. The standard scorecard measure
    of how cleanly a model splits the two populations."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def candidate_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=160, max_depth=10, min_samples_split=6, min_samples_leaf=3,
            random_state=config.RANDOM_SEED, n_jobs=-1, class_weight="balanced",
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=220, max_depth=6, learning_rate=0.06,
            random_state=config.RANDOM_SEED,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED,
        ),
    }


def train_model():
    print("Loading applicants from database...")
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM applicants", conn)
    finally:
        conn.close()
    print(f"Loaded {len(df)} applicant rows.")

    feature_cols = config.FEATURE_COLS
    X = df[feature_cols].copy()
    y = df["repayment_outcome"]

    impute_medians = {}
    for col in X.columns:
        med = X[col].median()
        impute_medians[col] = float(med) if not np.isnan(med) else 0.0
        X[col] = X[col].fillna(impute_medians[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )
    print(f"Training on {len(X_train)} samples ({y_train.mean():.1%} positive), "
          f"holding out {len(X_test)}.\n")

    leaderboard, trained = [], {}
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        entry = {
            "name": name,
            "roc_auc": round(float(auc), 4),
            "gini": round(float(2 * auc - 1), 4),
            "ks": round(ks_statistic(y_test, proba), 4),
            "brier": round(float(brier_score_loss(y_test, proba)), 4),
            "accuracy": round(float(model.score(X_test, y_test)), 4),
            "explainable": name == EXPLAINABLE_MODEL,
        }
        leaderboard.append(entry)
        trained[name] = model
        print(f"  {name:<22} AUC {entry['roc_auc']:.3f} | Gini {entry['gini']:.3f} | "
              f"KS {entry['ks']:.3f} | Brier {entry['brier']:.3f}")

    leaderboard.sort(key=lambda e: e["roc_auc"], reverse=True)
    best_by_auc = leaderboard[0]["name"]
    for entry in leaderboard:
        entry["best_by_auc"] = entry["name"] == best_by_auc
        entry["in_production"] = entry["name"] == EXPLAINABLE_MODEL

    model = trained[EXPLAINABLE_MODEL]
    print(f"\nHighest AUC: {best_by_auc}")
    print(f"In production: {EXPLAINABLE_MODEL} "
          f"({'also the highest' if best_by_auc == EXPLAINABLE_MODEL else 'chosen for exact explainability'})")

    proba = model.predict_proba(X_test)[:, 1]
    print("\n--- Production model on held-out data ---")
    print(classification_report(y_test, model.predict(X_test)))

    # Calibration: does a predicted 70% actually repay 70% of the time?
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
    calibration = [{"predicted": round(float(p), 4), "actual": round(float(a), 4)}
                   for p, a in zip(mean_pred, frac_pos)]

    # Unsupervised data-consistency screen, trained on the same feature space.
    print("Training data-consistency (anomaly) screen...")
    anomaly_model = IsolationForest(
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=config.RANDOM_SEED, n_estimators=150,
    ).fit(X_scaled)
    flagged = int((anomaly_model.predict(X_scaled) == -1).sum())
    print(f"  flags {flagged} of {len(X_scaled)} applications ({flagged / len(X_scaled):.1%}) for manual review")

    percentiles = {
        col: {int(p * 100): float(X[col].quantile(p)) for p in (0.10, 0.20, 0.50, 0.80, 0.90)}
        for col in feature_cols
    }

    feature_importance = {
        col: round(float(imp), 6)
        for col, imp in sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    }

    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    joblib.dump(feature_cols, FEATURES_FILENAME)
    joblib.dump(impute_medians, IMPUTE_FILENAME)
    joblib.dump(percentiles, PERCENTILES_FILENAME)
    joblib.dump(anomaly_model, ANOMALY_FILENAME)
    joblib.dump({"leaderboard": leaderboard, "calibration": calibration,
                 "production_model": EXPLAINABLE_MODEL, "n_train": len(X_train),
                 "n_test": len(X_test), "feature_importance": feature_importance},
                LEADERBOARD_FILENAME)
    print(f"\nSaved model, scaler, features, imputation, percentiles, anomaly screen and leaderboard.")


if __name__ == "__main__":
    train_model()
# Updated on 2026-02-18
