import argparse
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RF or XGB model accuracy and classification metrics on historical data.'
    )
    parser.add_argument(
        'model',
        nargs='?', 
        choices=['rf', 'xgb', 'all'],
        default='rf',
        help='Model to evaluate: rf, xgb, or all (default: rf)'
    )
    return parser.parse_args()


def load_data():
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM historical_features", conn,
            index_col=['Date', 'Ticker'], parse_dates=['Date']
        )
    finally:
        conn.close()

    df['Future_Close'] = df.groupby(level='Ticker')['Close'].shift(-config.PREDICTION_HORIZON_DAYS)
    df['Future_Return'] = (df['Future_Close'] / df['Close']) - 1.0
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    df.dropna(subset=['Target'], inplace=True)

    feature_cols = [col for col in config.FEATURE_COLS if col in df.columns]
    X = df[feature_cols].copy()
    y = df['Target'].copy()

    for col in X.columns:
        if X[col].dtype.kind in 'biufc' and X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    X.fillna(0, inplace=True)

    return X, y, feature_cols


def evaluate_model(model_choice, X, y, feature_cols):
    print(f"\n=== Evaluating {model_choice.upper()} ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_choice == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    else:
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError(
                "XGBoost is not available in this environment. "
                "Install xgboost or run the script with 'rf' only. "
                f"Original error: {e}"
            )

        neg_count = y_train.value_counts().get(0, 0)
        pos_count = y_train.value_counts().get(1, 0)
        scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1
        model = XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight_val
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    args = parse_args()

    try:
        X, y, feature_cols = load_data()
    except Exception as e:
        print(f"Error loading training data: {e}")
        raise

    if args.model == 'all':
        for model_choice in ['rf', 'xgb']:
            evaluate_model(model_choice, X, y, feature_cols)
    else:
        evaluate_model(args.model, X, y, feature_cols)
