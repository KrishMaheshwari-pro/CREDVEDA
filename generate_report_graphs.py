import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sqlite3
import config  # Importing your config file
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# CONFIGURATION SECTION (EDIT THIS)
# ==========================================
# Choose 'rf' or 'xgb' based on which model you trained last
MODEL_TYPE = 'rf' 

# Change this to 'CREDIT' if you want the graphs to say "Loan Status" 
# Change to 'STOCK' if you want them to say "Future Return"
GRAPH_LABELS = 'CREDIT' 

OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set labels based on choice
if GRAPH_LABELS == 'CREDIT':
    TARGET_NAME = "Loan Default Status"
    CLASS_LABELS = ["Paid (0)", "Default (1)"]
else:
    TARGET_NAME = "Future Market Direction"
    CLASS_LABELS = ["Down/Flat (0)", "Up (1)"]

# ==========================================
# 1. RE-LOAD DATA & PREPROCESS
# (This logic copies your model_training.py to ensure data matches)
# ==========================================
print("--- Loading Data and Model ---")

try:
    # Connect to DB
    conn = sqlite3.connect(config.DB_NAME)
    # Note: Adjust table name if your credit data uses a different table than 'historical_features'
    df = pd.read_sql_query("SELECT * FROM historical_features", conn)
    conn.close()
    print(f"Data loaded: {len(df)} rows")
except Exception as e:
    print(f"Error loading database: {e}")
    print("Make sure credit_intelligence.db is in the folder.")
    exit()

# --- Re-create the Target (Logic from your code) ---
# Ensure 'Date' is datetime if it exists, otherwise skip
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date', 'Ticker'], inplace=True)
else:
    # Fallback if using a simple credit CSV without dates
    pass

# Recalculate Target (using your logic)
# Note: If you switched to a credit CSV, ensure 'Target' column exists or is created here.
# Assuming the code structure provided:
if 'Close' in df.columns:
    df['Future_Close'] = df.groupby(level='Ticker')['Close'].shift(-config.PREDICTION_HORIZON_DAYS)
    df['Future_Return'] = (df['Future_Close'] / df['Close']) - 1.0
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    df.dropna(subset=['Target'], inplace=True)
else:
    # If you are using a raw credit csv that already has a target, just ensure it's named 'Target'
    # df.rename(columns={'loan_status': 'Target'}, inplace=True) 
    pass

# Load saved artifacts
try:
    model = joblib.load(f'model_{MODEL_TYPE}.joblib')
    scaler = joblib.load(f'scaler_{MODEL_TYPE}.joblib')
    feature_cols = joblib.load(f'feature_cols_{MODEL_TYPE}.joblib')
    print(f"Loaded {MODEL_TYPE} model and features.")
except FileNotFoundError:
    print(f"ERROR: Could not find model_{MODEL_TYPE}.joblib. Run model_training.py first!")
    exit()

# Prepare X and y
X = df[feature_cols].copy()
y = df['Target']

# Impute NaNs (Same as your training script)
for col in X.columns:
    if X[col].dtype.kind in 'biufc' and X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)
X.fillna(0, inplace=True)

# Scale X
X_scaled = scaler.transform(X)

# ==========================================
# 2. GENERATE GRAPHS
# ==========================================

# --- GRAPH 1: Target Distribution (EDA) ---
print("Generating Graph 1: Distribution...")
plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette='viridis')
plt.title(f"Distribution of {TARGET_NAME}", fontsize=15)
plt.xlabel("Class (0 vs 1)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks([0, 1], CLASS_LABELS)
plt.savefig(f"{OUTPUT_DIR}/1_target_distribution.png")
plt.close()

# --- GRAPH 2: Correlation Heatmap (EDA) ---
print("Generating Graph 2: Correlation Heatmap...")
plt.figure(figsize=(12, 10))
# Select only numeric columns for correlation
numeric_df = df[feature_cols + ['Target']].select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix", fontsize=15)
plt.savefig(f"{OUTPUT_DIR}/2_correlation_heatmap.png")
plt.close()

# --- GRAPH 3: Feature Importance (Model) ---
print("Generating Graph 3: Feature Importance...")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(10) # Top 10

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='magma')
    plt.title(f"Top 10 Features Predicting {TARGET_NAME}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_feature_importance.png")
    plt.close()
else:
    print("Skipping Feature Importance (Model type doesn't support it directly).")

# --- GRAPH 4: Confusion Matrix (Evaluation) ---
print("Generating Graph 4: Confusion Matrix...")
y_pred = model.predict(X_scaled)
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
disp.plot(cmap='Blues', values_format='d')
plt.title(f"Confusion Matrix ({MODEL_TYPE.upper()} Model)", fontsize=15)
plt.savefig(f"{OUTPUT_DIR}/4_confusion_matrix.png") # Note: standard savefig might clash with disp.plot, usually disp.figure_.savefig works best
plt.close() 
# Alternative save for CM if the above clashes:
disp.figure_.savefig(f"{OUTPUT_DIR}/4_confusion_matrix.png")

print(f"\nSUCCESS! All graphs saved in the '{OUTPUT_DIR}' folder.")