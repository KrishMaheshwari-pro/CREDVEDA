import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV # NEW: Import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score # NEW: Import more metrics
import joblib
import config
import sys

def train_model():
    """Loads data, trains the specified model, and saves it."""
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ['rf', 'xgb']:
        print("Usage: python model_training.py [rf|xgb]")
        print("  rf: RandomForest")
        print("  xgb: XGBoost")
        return
    
    model_choice = sys.argv[1].lower()
    print(f"--- Starting training for {model_choice.upper()} model ---")

    try:
        conn = sqlite3.connect(config.DB_NAME)
        df = pd.read_sql_query("SELECT * FROM historical_features", conn, index_col=['Date', 'Ticker'], parse_dates=['Date'])
        print(f"Loaded {len(df)} rows from the database.")
    except Exception as e:
        print(f"Could not load data from database: {e}")
        return
    finally:
        if conn:
            conn.close()

    # --- Create the Predictive Target ---
    df['Future_Close'] = df.groupby(level='Ticker')['Close'].shift(-config.PREDICTION_HORIZON_DAYS)
    df['Future_Return'] = (df['Future_Close'] / df['Close']) - 1.0
    df['Target'] = (df['Future_Return'] > 0).astype(int) # Target: 1 if future return > 0, else 0
    df.dropna(subset=['Target'], inplace=True)

    # --- Feature Selection and Imputation ---
    feature_cols = [col for col in config.FEATURE_COLS if col in df.columns]
    X = df[feature_cols].copy()
    y = df['Target']

    # Impute missing numerical values with median
    for col in X.columns:
        if X[col].dtype.kind in 'biufc' and X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    # Impute any remaining NaNs (e.g., from new columns that might appear) with 0
    X.fillna(0, inplace=True)

    # --- Scale Features and Split Data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y # stratify ensures balanced splits
    )
    
    # --- Handle Class Imbalance for XGBoost ---
    # Calculate scale_pos_weight: sum(negative instances) / sum(positive instances)
    # This gives more weight to the minority class (Target=1, assuming positive returns are rarer)
    neg_count = y_train.value_counts()[0] if 0 in y_train.value_counts() else 0
    pos_count = y_train.value_counts()[1] if 1 in y_train.value_counts() else 0
    
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1
    print(f"Calculated scale_pos_weight: {scale_pos_weight_val:.2f}")

    # --- Select and train the chosen model ---
    if model_choice == 'rf':
        print(f"Training RandomForestClassifier on {len(X_train)} samples...")
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=3, 
            random_state=42, 
            n_jobs=-1, 
            class_weight='balanced' # RandomForest has a direct 'balanced' option
        )
    elif model_choice == 'xgb':
        print(f"Training XGBClassifier on {len(X_train)} samples...")
        # Refined XGBoost hyperparameters for better performance
        model = XGBClassifier(
            n_estimators=500,        # Increased estimators
            max_depth=7,             # Slightly deeper trees
            learning_rate=0.05,      # Smaller learning rate
            subsample=0.7,           # Use 70% of data for each tree
            colsample_bytree=0.7,    # Use 70% of features for each tree
            gamma=0.1,               # Minimum loss reduction for a split
            use_label_encoder=False, # Suppress warning
            eval_metric='logloss',   # Metric for evaluation during training
            random_state=42, 
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight_val # Apply class imbalance handling
        )
        # --- Optional: Hyperparameter Tuning with GridSearchCV (for more rigorous optimization) ---
        # For a hackathon, the fixed parameters above are a good start.
        # For production, uncomment and run this for optimal tuning.
        # param_grid = {
        #     'n_estimators': [300, 500, 700],
        #     'max_depth': [5, 7, 9],
        #     'learning_rate': [0.01, 0.05, 0.1],
        #     'subsample': [0.6, 0.8, 1.0],
        #     'colsample_bytree': [0.6, 0.8, 1.0],
        #     'gamma': [0, 0.1, 0.2],
        #     'scale_pos_weight': [scale_pos_weight_val] # Keep this fixed if imbalance is significant
        # }
        # grid_search = GridSearchCV(
        #     estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        #     param_grid=param_grid,
        #     scoring='roc_auc', # Use ROC-AUC for imbalanced data
        #     cv=3,              # 3-fold cross-validation
        #     verbose=2,
        #     n_jobs=-1
        # )
        # grid_search.fit(X_train, y_train)
        # model = grid_search.best_estimator_
        # print(f"Best XGBoost parameters found: {grid_search.best_params_}")
        # print(f"Best ROC-AUC score on training set: {grid_search.best_score_:.2f}")
        
    model.fit(X_train, y_train)
    
    # --- Evaluate Model Performance ---
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    accuracy = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n--- Model Evaluation for {model_choice.upper()} ---")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Save model with a specific name ---
    model_filename = f'model_{model_choice}.joblib'
    scaler_filename = f'scaler_{model_choice}.joblib'
    features_filename = f'feature_cols_{model_choice}.joblib'
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(feature_cols, features_filename)
    print(f"\nModel saved as {model_filename}, scaler as {scaler_filename}, and features as {features_filename}.")

if __name__ == "__main__":
    train_model()
