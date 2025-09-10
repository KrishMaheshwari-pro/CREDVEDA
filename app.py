import sqlite3
import pandas as pd
import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from functools import wraps
import json
import hashlib # For password hashing
import joblib # To load your trained model, scaler, and feature_cols
import numpy as np
import shap # For explainability
from chatbot import get_chatbot_assets # Import chatbot assets

# --- Configuration ---
# In a real app, use a more secure secret key and manage it outside the code.
SECRET_KEY = os.urandom(24)
DB_NAME = 'credit_intelligence.db' # IMPORTANT: Ensure this matches your data_ingestion.py config!

# ML Model Configuration
MODEL_CHOICE = "xgb" # Assuming you'll train XGBoost
MODEL_FILENAME = f'model_{MODEL_CHOICE}.joblib'
SCALER_FILENAME = f'scaler_{MODEL_CHOICE}.joblib'
FEATURES_FILENAME = f'feature_cols_{MODEL_CHOICE}.joblib'

# Company Name Mapping
COMPANY_NAMES = {
    # Mega-Cap Tech & Growth
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG": "Alphabet Inc. (Class C)",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.",
    "NFLX": "Netflix, Inc.",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce, Inc.",
    "ORCL": "Oracle Corporation",
    "SAP": "SAP SE",
    "INTC": "Intel Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "AVGO": "Broadcom Inc.",
    "TXN": "Texas Instruments Incorporated",
    "MU": "Micron Technology, Inc.",

    # Financials & Payments
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "WFC": "Wells Fargo & Company",
    "GS": "The Goldman Sachs Group, Inc.",
    "MS": "Morgan Stanley",
    "C": "Citigroup Inc.",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "PYPL": "PayPal Holdings, Inc.",
    "AXP": "American Express Company",
    "BLK": "BlackRock, Inc.",
    "BRK-B": "Berkshire Hathaway Inc. (Class B)",
    "SPGI": "S&P Global Inc.",
    "SCHW": "The Charles Schwab Corporation",
    "COF": "Capital One Financial Corporation",
    "USB": "U.S. Bancorp",
    "PNC": "The PNC Financial Services Group, Inc.",

    # Healthcare
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Incorporated",
    "PFE": "Pfizer Inc.",
    "MRK": "Merck & Co., Inc.",
    "LLY": "Eli Lilly and Company",
    "ABBV": "AbbVie Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "MDT": "Medtronic plc",
    "DHR": "Danaher Corporation",
    "GILD": "Gilead Sciences, Inc.",
    "AMGN": "Amgen Inc.",
    "BMY": "Bristol-Myers Squibb Company",
    "ABT": "Abbott Laboratories",
    "CVS": "CVS Health Corporation",
    "CI": "Cigna Corporation",
    "ANTM": "Anthem, Inc. (now Elevance Health)", # Note: ANTM is now ELV
    "ISRG": "Intuitive Surgical, Inc.",
    "SYK": "Stryker Corporation",

    # Consumer Discretionary & Staples
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corporation",
    "HD": "The Home Depot, Inc.",
    "NKE": "NIKE, Inc.",
    "MCD": "McDonald's Corporation",
    "SBUX": "Starbucks Corporation",
    "TGT": "Target Corporation",
    "LOW": "Lowe's Companies, Inc.",
    "DIS": "The Walt Disney Company",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo, Inc.",
    "PG": "The Procter & Gamble Company",
    "PM": "Philip Morris International Inc.",
    "MO": "Altria Group, Inc.",
    "CL": "Colgate-Palmolive Company",
    "KMB": "Kimberly-Clark Corporation",
    "MDLZ": "Mondelez International, Inc.",

    # Industrials & Energy
    "CAT": "Caterpillar Inc.",
    "BA": "The Boeing Company",
    "DE": "Deere & Company",
    "HON": "Honeywell International Inc.",
    "GE": "General Electric Company",
    "UNP": "Union Pacific Corporation",
    "UPS": "United Parcel Service, Inc.",
    "FDX": "FedEx Corporation",
    "LMT": "Lockheed Martin Corporation",
    "RTX": "Raytheon Technologies Corporation",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "SLB": "Schlumberger Limited",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources, Inc.",

    # Communications & Other
    "VZ": "Verizon Communications Inc.",
    "T": "AT&T Inc.",
    "CMCSA": "Comcast Corporation",
    "IBM": "International Business Machines Corporation",
    "ACN": "Accenture plc",
    "NEE": "NextEra Energy, Inc.",
    "DUK": "Duke Energy Corporation",
    "SO": "The Southern Company",

    # More S&P 500 Components
    "MMM": "3M Company",
    "AOS": "A. O. Smith Corporation",
    "AES": "The AES Corporation",
    "AFL": "Aflac Incorporated",
    "A": "Agilent Technologies, Inc.",
    "APD": "Air Products and Chemicals, Inc.",
    "AKAM": "Akamai Technologies, Inc.",
    "ALB": "Albemarle Corporation",
    "ARE": "Alexandria Real Estate Equities, Inc.",
    "ALGN": "Align Technology, Inc.",
    "ALLE": "Allegion plc",
    "LNT": "Alliant Energy Corporation",
    "ALL": "Allstate Corporation",
    "AMCR": "Amcor plc",
    "AEE": "Ameren Corporation",
    "AEP": "American Electric Power Company, Inc.",
    "AIG": "American International Group, Inc.",
    "AMT": "American Tower Corporation",
    "AWK": "American Water Works Company, Inc.",
    "AMP": "Ameriprise Financial, Inc.",
    "AME": "AMETEK, Inc.",
    "APH": "Amphenol Corporation",
    "ADI": "Analog Devices, Inc.",
    "ANSS": "Ansys, Inc.",
    "AON": "Aon plc",
    "APA": "APA Corporation",
    "AMAT": "Applied Materials, Inc.",
    "APTV": "Aptiv PLC",
    "ADM": "Archer-Daniels-Midland Company",
    "ANET": "Arista Networks, Inc.",
    "AJG": "Arthur J. Gallagher & Co.",
    "AIZ": "Assurant, Inc.",
    "ATO": "Atmos Energy Corporation",
    "ADSK": "Autodesk, Inc.",
    "ADP": "Automatic Data Processing, Inc.",
    "AZO": "AutoZone, Inc.",
    "AVB": "AvalonBay Communities, Inc.",
    "AVY": "Avery Dennison Corporation",
    "BKR": "Baker Hughes Company",
    "BALL": "Ball Corporation",
    "BDX": "Becton, Dickinson and Company",
    "WRB": "W.R. Berkley Corporation",
    "BR": "Broadridge Financial Solutions, Inc.",
    "BSX": "Boston Scientific Corporation",
    "BBY": "Best Buy Co., Inc.",
    "BIO": "Bio-Rad Laboratories, Inc.",
    "TECH": "Bio-Techne Corporation",
    "BIIB": "Biogen Inc.",
    "BK": "The Bank of New York Mellon Corporation",
    "BAX": "Baxter International Inc.",
    "BBWI": "Bath & Body Works, Inc.",
    "BWA": "BorgWarner Inc.",
    "BXP": "Boston Properties, Inc.",
    "CHRW": "C.H. Robinson Worldwide, Inc.",
    "CDNS": "Cadence Design Systems, Inc.",
    "CZR": "Caesars Entertainment, Inc.",
    "CPT": "Camden Property Trust",
    "CPB": "Campbell Soup Company",
    "CAH": "Cardinal Health, Inc.",
    "KMX": "CarMax, Inc.",
    "CCL": "Carnival Corporation & plc",
    "CARR": "Carrier Global Corporation",
    "CTLT": "Catalent, Inc.",
    "CBOE": "Cboe Global Markets, Inc.",
    "CBRE": "CBRE Group, Inc.",
    "CDW": "CDW Corporation",
    "CE": "Celanese Corporation",
    "CNC": "Centene Corporation",
    "CNP": "CenterPoint Energy, Inc.",
    "CDAY": "Ceridian HCM Holding Inc.",
    "CF": "CF Industries Holdings, Inc.",
    "CRL": "Charles River Laboratories International, Inc.",
    "CHTR": "Charter Communications, Inc.",
    "CMG": "Chipotle Mexican Grill, Inc.",
    "CB": "Chubb Limited",
    "CHD": "Church & Dwight Co., Inc.",
    "CINF": "Cincinnati Financial Corporation",
    "CTAS": "Cintas Corporation",
    "CSCO": "Cisco Systems, Inc.",
    "CFG": "Citizens Financial Group, Inc.",
    "CLX": "The Clorox Company",
    "CME": "CME Group Inc.",
    "CMS": "CMS Energy Corporation",
    "CTSH": "Cognizant Technology Solutions Corporation",
    "CMA": "Comerica Incorporated",
    "CAG": "Conagra Brands, Inc.",
    "ED": "Consolidated Edison, Inc.",
    "STZ": "Constellation Brands, Inc.",
    "COO": "The Cooper Companies, Inc.",
    "CPRT": "Copart, Inc.",
    "GLW": "Corning Incorporated",
    "CTVA": "Corteva, Inc.",
    "CTRA": "Coterra Energy Inc.",
    "CCI": "Crown Castle Inc.",
    "CSX": "CSX Corporation",
    "CMI": "Cummins Inc.",
    "DHI": "D.R. Horton, Inc.",
    "DRI": "Darden Restaurants, Inc.",
    "DVA": "DaVita Inc.",
    "DAL": "Delta Air Lines, Inc.",
    "XRAY": "Dentsply Sirona Inc.",
    "DVN": "Devon Energy Corporation",
    "DXCM": "DexCom, Inc.",
    "FANG": "Diamondback Energy, Inc.",
    "DLR": "Digital Realty Trust, Inc.",
    "DFS": "Discover Financial Services",
    "DG": "Dollar General Corporation",
    "DLTR": "Dollar Tree, Inc.",
    "D": "Dominion Energy, Inc.",
    "DPZ": "Domino's Pizza, Inc.",
    "DOV": "Dover Corporation",
    "DOW": "Dow Inc.",
    "DTE": "DTE Energy Company",
    "DD": "DuPont de Nemours, Inc.",
    "DXC": "DXC Technology Company",
    "EMN": "Eastman Chemical Company",
    "ETN": "Eaton Corporation plc",
    "EBAY": "eBay Inc.",
    "ECL": "Ecolab Inc.",
    "EIX": "Edison International",
    "EW": "Edwards Lifesciences Corporation",
    "EA": "Electronic Arts Inc.",
    "EL": "The Estée Lauder Companies Inc.",
    "EMR": "Emerson Electric Co.",
    "ENPH": "Enphase Energy, Inc.",
    "ETR": "Entergy Corporation",
    "EPAM": "EPAM Systems, Inc.",
    "EFX": "Equifax Inc.",
    "EQIX": "Equinix, Inc.",
    "EQR": "Equity Residential",
    "ESS": "Essex Property Trust, Inc.",
    "ELV": "Elevance Health, Inc.",
    "ETSY": "Etsy, Inc.",
    "RE": "Everest Re Group, Ltd.",
    "EVRG": "Evergy, Inc.",
    "ES": "Eversource Energy",
    "EXC": "Exelon Corporation",
    "EXPE": "Expedia Group, Inc.",
    "EXPD": "Expeditors International of Washington, Inc.",
    "EXR": "Extra Space Storage Inc.",
    "FFIV": "F5, Inc.",
    "FDS": "FactSet Research Systems Inc.",
    "FAST": "Fastenal Company",
    "FRT": "Federal Realty Investment Trust",
    "FITB": "Fifth Third Bancorp",
    "FSLR": "First Solar, Inc.",
    "FE": "FirstEnergy Corp.",
    "FIS": "Fidelity National Information Services, Inc.",
    "FISV": "Fiserv, Inc.",
    "FLT": "FleetCor Technologies, Inc.",
    "FMC": "FMC Corporation",
    "F": "Ford Motor Company",
    "FTNT": "Fortinet, Inc.",
    "FTV": "Fortive Corporation",
    "FOXA": "Fox Corporation (Class A)",
    "FOX": "Fox Corporation (Class B)",
    "BEN": "Franklin Resources, Inc.",
    "FCX": "Freeport-McMoRan Inc.",
    "GRMN": "Garmin Ltd.",
    "IT": "Gartner, Inc.",
    "GNRC": "Generac Holdings Inc.",
    "GD": "General Dynamics Corporation",
    "GIS": "General Mills, Inc.",
    "GM": "General Motors Company",
    "GPC": "Genuine Parts Company",
    "GL": "Globe Life Inc.",
    "GPN": "Global Payments Inc.",
    "HAL": "Halliburton Company",
    "HIG": "The Hartford Financial Services Group, Inc.",
    "HAS": "Hasbro, Inc.",
    "HCA": "HCA Healthcare, Inc.",
    "PEAK": "Healthpeak Properties, Inc.",
    "HSIC": "Henry Schein, Inc.",
    "HSY": "The Hershey Company",
    "HES": "Hess Corporation",
    "HPE": "Hewlett Packard Enterprise Company",
    "HLT": "Hilton Worldwide Holdings Inc.",
    "HOLX": "Hologic, Inc.",
    "HRL": "Hormel Foods Corporation",
    "HST": "Host Hotels & Resorts, Inc.",
    "HWM": "Howmet Aerospace Inc.",
    "HPQ": "HP Inc.",
    "HUM": "Humana Inc.",
    "HBAN": "Huntington Bancshares Incorporated",
    "HII": "Huntington Ingalls Industries, Inc.",
    "IEX": "IDEX Corporation",
    "IDXX": "IDEXX Laboratories, Inc.",
    "ITW": "Illinois Tool Works Inc.",
    "ILMN": "Illumina, Inc.",
    "INCY": "Incyte Corporation",
    
    # Indian Market (NSE)
    "TCS.NS": "Tata Consultancy Services Limited",
    "INFY.NS": "Infosys Limited",
    "WIPRO.NS": "Wipro Limited",
    "HCLTECH.NS": "HCL Technologies Limited",
    "TECHM.NS": "Tech Mahindra Limited",
    "HDFCBANK.NS": "HDFC Bank Limited",
    "ICICIBANK.NS": "ICICI Bank Limited",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Limited",
    "SBIN.NS": "State Bank of India",
    "AXISBANK.NS": "Axis Bank Limited",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone Limited",
    "ADANIGREEN.NS": "Adani Green Energy Limited",
    "HINDUNILVR.NS": "Hindustan Unilever Limited",
    "ITC.NS": "ITC Limited",
    "NESTLEIND.NS": "Nestle India Limited",
    "BRITANNIA.NS": "Britannia Industries Limited",
    "MARUTI.NS": "Maruti Suzuki India Limited",
    "TATAMOTORS.NS": "Tata Motors Limited",
    "M&M.NS": "Mahindra & Mahindra Limited",
    "BAJAJ-AUTO.NS": "Bajaj Auto Limited",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited",
    "DRREDDY.NS": "Dr. Reddy's Laboratories Limited",
    "CIPLA.NS": "Cipla Limited",
    "TATASTEEL.NS": "Tata Steel Limited",
    "JSWSTEEL.NS": "JSW Steel Limited",
    "HINDALCO.NS": "Hindalco Industries Limited",
    "ULTRACEMCO.NS": "UltraTech Cement Limited",
    "GRASIM.NS": "Grasim Industries Limited",
    "LT.NS": "Larsen & Toubro Limited",
    "AMBUJACEM.NS": "Ambuja Cements Limited",
    "ASIANPAINT.NS": "Asian Paints Limited",
    "PIDILITIND.NS": "Pidilite Industries Limited",
    "BERGEPAINT.NS": "Berger Paints India Limited",
    "TITAN.NS": "Titan Company Limited",
    "DMART.NS": "Avenue Supermarts Limited",
    "HAVELLS.NS": "Havells India Limited",
    "BHARTIARTL.NS": "Bharti Airtel Limited",
    "INDIGO.NS": "InterGlobe Aviation Limited",
    "ZEEL.NS": "Zee Entertainment Enterprises Limited",
}


# --- Flask Application Setup ---
app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- Database Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def init_user_db():
    """Initializes the users table in the database."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

def hash_password(password):
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verifies a provided password against a stored hashed password."""
    return stored_password == hash_password(provided_password)

def add_user(username, password):
    """Adds a new user to the database."""
    hashed_pwd = hash_password(password)
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pwd))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False # Username already exists
        finally:
            conn.close()
    return False

def authenticate_user(username, password):
    """Authenticates a user against the database."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return verify_password(result[0], password)
    return False

# --- ML Model Loading (Load once on app startup) ---
model = None
scaler = None
feature_cols = None

def load_ml_components():
    global model, scaler, feature_cols
    try:
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        feature_cols = joblib.load(FEATURES_FILENAME)
        print("ML Model components loaded successfully.")
    except FileNotFoundError:
        print(f"WARNING: Model files not found ({MODEL_FILENAME}, {SCALER_FILENAME}, {FEATURES_FILENAME}). Dashboard features requiring ML will be disabled.")
    except Exception as e:
        print(f"ERROR: Failed to load ML model components: {e}")

# --- Data Loading (from historical_features table) ---
# This will be loaded once and passed to functions that need it.
# In a real-time system, this would be periodically refreshed.
df_raw = pd.DataFrame()

def load_all_historical_features():
    global df_raw
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM historical_features", conn, index_col=['Date', 'Ticker'], parse_dates=['Date'])
            df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0]), level='Date')
            df.index = df.index.set_levels(df.index.levels[1].astype(str), level='Ticker')
            df_raw = df
            print(f"Loaded {len(df_raw)} rows of historical features from database.")
        except Exception as e:
            print(f"ERROR: Could not load historical features from database: {e}. Ensure data_ingestion.py has been run.")
            df_raw = pd.DataFrame() # Ensure df_raw is empty DataFrame on failure
        finally:
            conn.close()
    else:
        df_raw = pd.DataFrame()

# --- ML Prediction & Explainability Functions (adapted from Streamlit) ---
def preprocess_data_for_prediction(df_input, scaler_obj, feature_cols_list):
    df_processed = df_input[feature_cols_list].copy()
    for col in df_processed.columns:
        if df_processed[col].dtype.kind in 'biufc' and df_processed[col].isnull().any():
            df_processed[col].fillna(0, inplace=True)
    df_processed.fillna(0, inplace=True)
    X_scaled = scaler_obj.transform(df_processed)
    return X_scaled

def get_credit_score_prediction(data_point_scaled, model_obj):
    if model_obj:
        proba = model_obj.predict_proba(data_point_scaled)[:, 1][0]
        credit_score = int(proba * 100)
        return credit_score
    return None

def get_shap_explanation(scaled_data_point, model_obj, feature_names, original_data_point_series):
    if model_obj is None:
        return [], [], 50 # Return empty if model not loaded

    explainer = shap.TreeExplainer(model_obj)
    shap_values = explainer.shap_values(scaled_data_point)


    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[1][0]
    else:
        shap_values_to_use = shap_values[0]

    explanation_data = []
    for i, feature in enumerate(feature_names):
        explanation_data.append({
            'Feature': feature,
            'Contribution': shap_values_to_use[i]
        })
    explanation_df = pd.DataFrame(explanation_data).sort_values(by='Contribution', ascending=False)

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]
    base_score_contribution = int((1 / (1 + np.exp(-base_value))) * 100)

    event_snippets = []
    # Check if event columns exist before accessing
    event_cols = ['Negative_Debt', 'Negative_Demand', 'Negative_Legal', 'Positive_Growth', 'Positive_Innovation']
    for col in event_cols:
        if col in original_data_point_series and original_data_point_series[col].item() == 1:
            event_snippets.append(f"🚨 **{col.replace('_', ' ')} Alert!** Event detected.")
    
    if 'NewsSentiment' in original_data_point_series:
        news_sentiment_val = original_data_point_series['NewsSentiment'].item()
        if news_sentiment_val > 0.3:
            event_snippets.append(f"📰 **Positive News Flow:** News sentiment is high ({news_sentiment_val:.2f}).")
        elif news_sentiment_val < -0.3:
            event_snippets.append(f"🗞️ **Negative News Impact:** News sentiment is low ({news_sentiment_val:.2f}).")

    if 'TranscriptSentiment' in original_data_point_series:
        transcript_sentiment_val = original_data_point_series['TranscriptSentiment'].item()
        if transcript_sentiment_val > 0.1:
            event_snippets.append(f"🗣️ **Positive Transcript Tone:** Earnings call transcript sentiment is positive ({transcript_sentiment_val:.2f}).")
        elif transcript_sentiment_val < -0.1:
            event_snippets.append(f"🎤 **Negative Transcript Tone:** Earnings call transcript sentiment is negative ({transcript_sentiment_val:.2f}).")

    if not event_snippets:
        event_snippets.append("No significant events or strong sentiment detected for this period.")

    return explanation_df.to_dict(orient='records'), event_snippets, base_score_contribution

# --- Anomaly Detection ---
def detect_anomalies(df_ticker, feature_cols):
    """
    Detects anomalies based on significant changes in key features.
    This is a simplified rule-based approach. A more advanced system
    could use statistical methods (e.g., Z-score) or unsupervised ML.
    """
    alerts = []
    if len(df_ticker) < 7: # Need at least a week of data
        return alerts

    latest_data = df_ticker.iloc[-1]
    prev_week_data = df_ticker.iloc[-7]

    # 1. Volatility Spike
    vol_change = (latest_data['Volatility'] - prev_week_data['Volatility']) / (prev_week_data['Volatility'] + 1e-6)
    if vol_change > 0.2: # 20% increase in volatility
        alerts.append({
            "time": "Recent",
            "type": "Warning",
            "message": f"Volatility has spiked by {vol_change:.1%} over the last 7 days.",
            "color": "#ffc107"
        })

    # 2. Sudden Sentiment Drop
    if 'NewsSentiment' in latest_data and latest_data['NewsSentiment'] < -0.3:
        alerts.append({
            "time": "Today",
            "type": "Critical",
            "message": f"Extremely negative news sentiment ({latest_data['NewsSentiment']:.2f}) detected.",
            "color": "#dc3545"
        })

    # 3. Significant Score Change
    if 'Credit_Score_Predicted' in df_ticker.columns and len(df_ticker) > 1:
        score_change = df_ticker['Credit_Score_Predicted'].iloc[-1] - df_ticker['Credit_Score_Predicted'].iloc[-2]
        if abs(score_change) >= 5:
            direction = "dropped" if score_change < 0 else "increased"
            alerts.append({
                "time": "1d",
                "type": "Critical",
                "message": f"Credit score has {direction} by {abs(score_change):.0f} points in the last day.",
                "color": "#dc3545" if score_change < 0 else "#28a745"
            })

    # 4. Positive Growth Event
    if 'Positive_Growth' in latest_data and latest_data['Positive_Growth'] == 1:
        alerts.append({
            "time": "Recent",
            "type": "Info",
            "message": "Positive growth event detected in recent company announcements.",
            "color": "#28a745"
        })
        
    # 5. Negative Debt Event
    if 'Negative_Debt' in latest_data and latest_data['Negative_Debt'] == 1:
        alerts.append({
            "time": "Recent",
            "type": "Critical",
            "message": "Negative event related to company's debt structure detected.",
            "color": "#dc3545"
        })

    if not alerts:
        alerts.append({
            "time": "Current",
            "type": "Info",
            "message": "No significant anomalies detected in the recent data.",
            "color": "#17a2b8"
        })

    return alerts



# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('home')) # Already logged in, redirect to home

    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            error = 'Username and password cannot be empty.'
        elif authenticate_user(username, password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home')) # Redirect to home on successful login
        else:
            error = 'Invalid Username or Password.'
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('logged_in'):
        return redirect(url_for('home')) # Already logged in, redirect to home

    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            error = 'Username and password cannot be empty.'
        elif add_user(username, password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home')) # Redirect to home on successful signup
        else:
            error = 'Username already exists. Please choose a different one.'
    return render_template('signup.html', error=error)

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('home.html', username=session.get('username'))

@app.route('/dashboard')
@login_required
def dashboard_page():
    if df_raw.empty:
        # In a real app, you might trigger data ingestion or show a loading page
        return render_template('error.html', message="Data not loaded. Please run data_ingestion.py first.")

    tickers = sorted(df_raw.index.get_level_values('Ticker').unique().tolist())
    selected_ticker = request.args.get('ticker', tickers[0] if tickers else None)
    
    # Get latest data point for the selected ticker
    df_ticker = df_raw.loc[(slice(None), selected_ticker), :].copy()
    df_ticker = df_ticker.reset_index(level='Ticker', drop=True).sort_index()
    
    # Get available years for the dropdown
    available_years = sorted(df_ticker.index.year.unique(), reverse=True) if not df_ticker.empty else []
    
    # Get selected year, default to the latest year in data or current year
    latest_year_in_data = available_years[0] if available_years else pd.Timestamp.now().year
    selected_year = request.args.get('year', default=latest_year_in_data, type=int)

    # Detect anomalies for the selected ticker
    anomaly_alerts = []
    if feature_cols and not df_ticker.empty:
        anomaly_alerts = detect_anomalies(df_ticker, feature_cols)

    # Calculate current metrics and deltas
    current_credit_score = None
    current_volatility = None
    current_macro = None
    delta_score = None
    delta_vol = None
    delta_macro = None
    
    if not df_ticker.empty:
        latest_date_data = df_ticker.iloc[-1]
        
        # Predict current credit score
        if model and scaler and feature_cols:
            try:
                current_data_point_df = latest_date_data[feature_cols].to_frame().T
                scaled_data_point = preprocess_data_for_prediction(current_data_point_df, scaler, feature_cols)
                current_credit_score = get_credit_score_prediction(scaled_data_point, model)
            except Exception as e:
                print(f"Prediction error for current score: {e}")
                current_credit_score = None

        # Calculate deltas (ensure enough history)
        if 'Credit_Score_Predicted' not in df_ticker.columns and model and scaler and feature_cols:
            score_history = []
            for date_idx in range(len(df_ticker)):
                hist_data_point = df_ticker.iloc[[date_idx]][feature_cols].copy()
                hist_scaled = preprocess_data_for_prediction(hist_data_point, scaler, feature_cols)
                score_history.append(get_credit_score_prediction(hist_scaled, model))
            df_ticker['Credit_Score_Predicted'] = score_history

        if 'Credit_Score_Predicted' in df_ticker.columns and len(df_ticker['Credit_Score_Predicted']) >= 2 and pd.notna(df_ticker['Credit_Score_Predicted'].iloc[-2]):
            delta_score = df_ticker['Credit_Score_Predicted'].iloc[-1] - df_ticker['Credit_Score_Predicted'].iloc[-2]

        current_volatility = latest_date_data['Volatility'].item() * 100 if 'Volatility' in latest_date_data else None
        if 'Volatility' in df_ticker.columns and len(df_ticker['Volatility']) >= 7 and pd.notna(df_ticker['Volatility'].iloc[-7]):
            delta_vol = (current_volatility - (df_ticker['Volatility'].iloc[-7].item() * 100))

        current_macro = latest_date_data['MacroIndicator'].item() if 'MacroIndicator' in latest_date_data else None
        if 'MacroIndicator' in df_ticker.columns and len(df_ticker['MacroIndicator']) >= 30 and pd.notna(df_ticker['MacroIndicator'].iloc[-30]):
            delta_macro = (current_macro - df_ticker['MacroIndicator'].iloc[-30].item())

    # Prepare data for rendering
    dashboard_data = {
        'selected_ticker': selected_ticker,
        'current_credit_score': current_credit_score,
        'delta_score': f"{delta_score:+.0f} points (1d)" if delta_score is not None else "N/A",
        'current_volatility': f"{current_volatility:.2f}%" if current_volatility is not None else "N/A",
        'delta_vol': f"{delta_vol:+.2f}% (7d)" if delta_vol is not None else "N/A",
        'current_macro': f"{current_macro:.2f}" if current_macro is not None else "N/A",
        'delta_macro': f"{delta_macro:+.2f} (30d)" if delta_macro is not None else "N/A",
        'tickers': {ticker: COMPANY_NAMES.get(ticker, ticker) for ticker in tickers},
        'available_years': available_years,
        'selected_year': selected_year,
        # Pass sentiment values for unstructured data feed tab
        'current_news_sentiment': latest_date_data['NewsSentiment'].item() if 'NewsSentiment' in latest_date_data else None,
        'current_transcript_sentiment': latest_date_data['TranscriptSentiment'].item() if 'TranscriptSentiment' in latest_date_data else None,
        # Pass latest macro indicator for global snapshot tab
        'latest_macro_indicator': latest_date_data['MacroIndicator'].item() if 'MacroIndicator' in latest_date_data else None,
        # Pass market sentiment for global snapshot tab (calculated across all tickers for latest date)
        'market_sentiment': df_raw.loc[df_raw.index.get_level_values('Date').max(), 'NewsSentiment'].mean() if not df_raw.empty and 'NewsSentiment' in df_raw.columns else 0.0,
        'anomaly_alerts': anomaly_alerts
    }

    return render_template('dashboard.html', **dashboard_data)

@app.route('/ai-features')
def ai_features_page():
    return render_template('ai_features.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/contact')
@login_required
def contact_page():
    return render_template('contact.html')

@app.route('/privacy')
@login_required
def privacy_page():
    return render_template('privacy.html')

@app.route('/terms')
@login_required
def terms_page():
    return render_template('terms.html')

# --- API Endpoints for dynamic content (AJAX calls from frontend) ---
@app.route('/api/historical_data/<string:ticker>')
@login_required
def api_historical_data(ticker):
    hist_df = df_raw.loc[(slice(None), ticker), 'Close'].copy()
    if hist_df.empty:
        return jsonify({'error': 'No data found for ticker'}), 404
    
    # For Chart.js, dates and prices should be lists
    dates = hist_df.index.get_level_values('Date').strftime('%Y-%m-%d').tolist()
    prices = hist_df.values.tolist()
    
    return jsonify({'dates': dates, 'prices': prices})

@app.route('/api/credit_score_trend/<string:ticker>/<int:year>')
@login_required
def api_credit_score_trend(ticker, year):
    df_ticker = df_raw.loc[(slice(None), ticker), :].copy()
    df_ticker = df_ticker.reset_index(level='Ticker', drop=True).sort_index()
    
    # Filter by selected year
    df_ticker = df_ticker[df_ticker.index.year == year]

    if df_ticker.empty or model is None or scaler is None or feature_cols is None:
        return jsonify({'error': 'No data or model not loaded'}), 404

    score_history = []
    for date_idx in range(len(df_ticker)):
        hist_data_point = df_ticker.iloc[[date_idx]][feature_cols].copy()
        hist_scaled = preprocess_data_for_prediction(hist_data_point, scaler, feature_cols)
        score_history.append(get_credit_score_prediction(hist_scaled, model))
    
    dates = df_ticker.index.strftime('%Y-%m-%d').tolist()
    scores = score_history
    
    return jsonify({'dates': dates, 'scores': scores})

@app.route('/api/explainability/<string:ticker>/latest')
@login_required
def api_explainability_latest(ticker):
    if df_raw.empty or model is None or scaler is None or feature_cols is None:
        return jsonify({'error': 'Backend model or data not available.'}), 503

    try:
        # Get the latest data point for the selected ticker
        df_ticker = df_raw.loc[(slice(None), ticker), :].copy().sort_index()
        if df_ticker.empty:
            return jsonify({'error': f'No historical data found for {ticker}.'}), 404
        
        latest_data_point = df_ticker.iloc[[-1]]
        latest_date = latest_data_point.index.get_level_values('Date')[0]

        # Ensure all required feature columns are present
        if not all(col in latest_data_point.columns for col in feature_cols):
            missing_cols = [col for col in feature_cols if col not in latest_data_point.columns]
            return jsonify({'error': f'Data is missing required columns: {", ".join(missing_cols)}'}), 500

        # Preprocess data and get prediction
        scaled_data_point = preprocess_data_for_prediction(latest_data_point[feature_cols], scaler, feature_cols)
        predicted_score = get_credit_score_prediction(scaled_data_point, model)

        # Get SHAP explanations
        explanation_data, events_insights, base_score = get_shap_explanation(
            scaled_data_point, model, feature_cols, latest_data_point.iloc[0]
        )
        
        return jsonify({
            'date': latest_date.strftime('%Y-%m-%d'),
            'predicted_score': predicted_score,
            'explanation_data': explanation_data,
            'events_insights': events_insights,
            'base_score': base_score
        })
    except Exception as e:
        print(f"Explainability error for {ticker}: {e}")
        # It's good practice to log the full error: import traceback; traceback.print_exc();
        return jsonify({'error': f'An unexpected error occurred while generating the explanation for {ticker}.'}), 500

@app.route('/set_theme', methods=['POST'])
def set_theme():
    data = request.get_json()
    theme = data.get('theme')
    if theme in ['light', 'dark']:
        session['theme'] = theme
        return jsonify({'status': 'success', 'theme': theme})
    return jsonify({'status': 'error', 'message': 'Invalid theme'}), 400

# --- Setup and Run ---
def setup_templates():
    """Creates the templates directory and HTML files if they don't exist."""
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)

    # Base HTML structure for all pages
    chatbot_assets = get_chatbot_assets()
    BASE_TEMPLATE = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{% block title %}}CredVeda AI-Powered Credit Intelligence{{% endblock %}}</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
        {{% if session.get('logged_in') %}}
        <style>
            {chatbot_assets['css']}
        </style>
        {{% endif %}}
        <style>
            /* Base styles for dark theme */
            body {{ 
                font-family: 'Inter', sans-serif; 
                background-color: #0d1117; 
                color: #e6edf3; 
                margin: 0; 
                padding: 0; 
                font-size: 18px; /* Base font size increased for readability */
                /* Dynamic Background Image */
                background-image: url('https://images.unsplash.com/photo-1518458028782-cf817c18953e?q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1920&h=1080&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'); /* Abstract tech/finance image */
                background-size: cover;
                background-position: center;
                background-attachment: fixed; /* Makes background fixed when scrolling */
                position: relative; /* Needed for overlay */
            }}
            /* Background Overlay */
            body::before {{
                content: '';
                position: fixed; /* Use fixed to ensure it covers the whole viewport */
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.75); /* Darker overlay for better text contrast */
                z-index: -1; /* Puts it behind content */
            }}

            h1, h2, h3, h4, h5, h6 {{ 
                color: #e6edf3; 
                font-weight: bold; 
                text-shadow: 0 0 5px rgba(230, 237, 243, 0.3); 
                margin-bottom: 1em; /* Increased spacing */
                line-height: 1.4; /* Better line height */
            }}
            p, li, span, div:not(.logo):not(.fixed-header):not(.fixed-footer):not(.auth-container):not(.auth-form):not(.auth-toggle-text):not(.nav-links):not(.footer-links):not(.copyright) {{ 
                color: #c0c0c0; /* Slightly lighter gray for general text */
                font-size: 1.15em; /* Increased general text font size */
                line-height: 1.7; /* Better line spacing */
                margin-bottom: 1.2em; /* Increased paragraph spacing */
            }}
            a {{ color: #58a6ff; text-decoration: none; }}
            a:hover {{ color: #79c0ff; text-decoration: underline; }}

            /* Light theme styles */
            body.light-theme {{
                background-color: #f0f2f5;
                color: #333;
                background-image: url('https://images.unsplash.com/photo-1518458028782-cf817c18953e?q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1920&h=1080&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'); /* Same image, but light theme overlay */
            }}
            body.light-theme::before {{
                background-color: rgba(255, 255, 255, 0.85); /* Lighter, more opaque overlay for readability */
            }}
            body.light-theme h1, body.light-theme h2, body.light-theme h3, body.light-theme h4, body.light-theme h5, body.light-theme h6 {{
                color: #1a202c;
                text-shadow: none;
            }}
            body.light-theme p, body.light-theme li, body.light-theme span, body.light-theme div:not(.logo):not(.fixed-header):not(.fixed-footer):not(.auth-container):not(.auth-form):not(.auth-toggle-text):not(.nav-links):not(.footer-links):not(.copyright) {{
                color: #4a5568;
            }}
            body.light-theme .fixed-header, body.light-theme .fixed-footer, body.light-theme .bg-dark-card, body.light-theme .auth-form, body.light-theme .table-auto, body.light-theme .table-auto th, body.light-theme .table-auto tbody tr:hover, body.light-theme .auth-form input, body.light-theme .auth-form button, body.light-theme .auth-form .auth-toggle-text a, body.light-theme .tab-button, body.light-theme .tab-button.active {{
                background-color: #ffffff;
                color: #333;
                border-color: #e2e8f0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
            body.light-theme .fixed-header .logo, body.light-theme .fixed-header .nav-links a, body.light-theme .fixed-footer .logo, body.light-theme .fixed-footer .footer-links a, body.light-theme .fixed-footer .copyright {{
                color: #1a202c;
            }}
            body.light-theme .fixed-header .nav-links a:hover, body.light-theme .fixed-footer .footer-links a:hover {{
                color: #007bff;
            }}
            body.light-theme .btn-primary {{
                background-color: #007bff;
                color: white;
            }}
            body.light-theme .btn-primary:hover {{
                background-color: #0056b3;
            }}
            body.light-theme .text-blue-accent {{ color: #007bff; }}
            body.light-theme .text-green-accent {{ color: #28a745; }}
            body.light-theme .text-red-accent {{ color: #dc3545; }}
            body.light-theme .bg-gray-700 {{ background-color: #f0f2f5; }} /* For selectbox/input background */
            body.light-theme .text-gray-700 {{ color: #4a5568; }} /* For selectbox/input text */
            body.light-theme .border-gray-600 {{ border-color: #cbd5e0; }}
            body.light-theme .bg-gray-800 {{ background-color: #e2e8f0; }} /* For chart backgrounds */
            body.light-theme .bg-red-900 {{ background-color: #fed7d7; }} /* Error alert */
            body.light-theme .border-red-700 {{ border-color: #fc8181; }}
            body.light-theme .text-red-300 {{ color: #c53030; }}
            body.light-theme .bg-green-900 {{ background-color: #c6f6d5; }} /* Success alert */
            body.light-theme .border-green-700 {{ border-color: #68d391; }}
            body.light-theme .text-green-300 {{ color: #2f855a; }}
            body.light-theme .auth-form input {{
                background-color: #f0f2f5;
                color: #333;
            }}
            body.light-theme .auth-form label {{
                color: #1a202c; /* Dark color for labels in light mode */
            }}
            body.light-theme .auth-form button {{
                background-color: #007bff;
                color: white;
            }}
            body.light-theme .auth-form button:hover {{
                background-color: #0056b3;
            }}
            body.light-theme .auth-toggle-text a {{
                color: #007bff;
            }}
            body.light-theme .tab-button.active {{
                border-color: #007bff;
            }}


            /* Dropdown/Select Box Styling */
            select {{
                background-color: #21262d; /* Dark background for select */
                color: #e6edf3; /* White text for select */
                border: 1px solid #30363d;
            }}
            body.light-theme select {{
                background-color: #f0f2f5; /* Light background for select */
                color: #1a202c; /* Black text for select */
                border-color: #cbd5e0;
            }}

            /* Fixed Header Styling */
            .fixed-header {{
                position: fixed; top: 0; left: 0; width: 100%; background-color: #0d1117; padding: 10px 20px; /* Reduced horizontal padding */
                display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                z-index: 1000; border-bottom: 1px solid #30363d; height: 70px;
            }}
            .fixed-header .logo {{ display: flex; align-items: center; font-size: 1.8em; font-weight: bold; color: #e6edf3; }}
            .fixed-header .logo img {{ height: 40px; margin-right: 10px; border-radius: 5px; }}
            .fixed-header .nav-links a {{ color: #e6edf3; text-decoration: none; margin-left: 15px; font-size: 1.0em; transition: color 0.2s ease-in-out; }} /* Reduced margin and font size */
            .fixed-header .nav-links a:hover {{ color: #58a6ff; }}
            
            .main-content {{ padding-top: 80px; padding-bottom: 20px; min-height: calc(100vh - 220px); }} /* Adjusted for new footer height */

            .site-footer {{
                background-color: #0d1117;
                color: #e6edf3;
                padding: 40px 50px; /* Increased padding for more height */
                border-top: 1px solid #30363d;
                margin-top: 60px; /* More space above the footer */
            }}
            .footer-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 30px;
                margin-bottom: 30px;
            }}
            .footer-column h4 {{
                font-size: 1.1em;
                font-weight: bold;
                color: #58a6ff;
                margin-bottom: 15px;
            }}
            .footer-column p, .footer-column a {{
                font-size: 0.9em;
                color: #8b949e;
                text-decoration: none;
                display: block;
                margin-bottom: 8px;
            }}
            .footer-column a:hover {{
                color: #79c0ff;
                text-decoration: underline;
            }}
            .footer-disclaimer {{
                font-size: 0.8em;
                color: #8b949e;
                text-align: center;
                border-top: 1px solid #30363d;
                padding-top: 20px;
                margin-top: 20px;
            }}

            /* General content styling */
            .container {{ max-width: 1200px; margin-left: auto; margin-right: auto; padding-left: 1rem; padding-right: 1rem; }}
            h1, h2, h3, h4, h5, h6 {{ color: #e6edf3; font-weight: bold; text-shadow: 0 0 5px rgba(230, 237, 243, 0.3); }}
            p, li {{ color: #8b949e; }}
            .bg-dark-card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); padding: 20px; }}
            .text-blue-accent {{ color: #58a6ff; }}
            .text-green-accent {{ color: #28a745; }}
            .text-red-accent {{ color: #dc3545; }}
            .btn-primary {{ background-color: #007bff; color: white; border-radius: 8px; padding: 12px 25px; font-size: 1.1em; transition: background-color 0.2s; }}
            .btn-primary:hover {{ background-color: #0056b3; }}
            .table-auto {{ width: 100%; border-collapse: collapse; }}
            .table-auto th, .table-auto td {{ padding: 12px; text-align: left; border-bottom: 1px solid #30363d; }}
            .table-auto th {{ background-color: #21262d; color: #58a6ff; font-weight: bold; }}
            .table-auto tbody tr:hover {{ background-color: #1a1f26; }}
            .text-center {{ text-align: center; }}

            /* Auth Form Styling - Adjusted for smaller inputs and better UI */
            .auth-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 70vh;
                padding: 20px;
            }}
            .auth-form {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 30px; /* Reduced padding */
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
                width: 100%;
                max-width: 350px; /* Reduced max-width */
                text-align: center;
            }}
            .auth-form h2 {{
                color: #58a6ff;
                margin-bottom: 20px; /* Reduced margin */
                font-size: 1.8em; /* Adjusted font size */
            }}
            .auth-form p {{
                font-size: 0.9em; /* Smaller text */
                margin-bottom: 25px; /* Adjusted margin */
            }}
            .auth-form label {{
                color: #e6edf3;
                font-size: 0.9em; /* Smaller label font */
                margin-bottom: 5px;
                display: block;
                text-align: left;
            }}
            .auth-form input {{
                width: calc(100% - 16px); /* Adjusted width for padding */
                padding: 8px; /* Smaller padding */
                margin-bottom: 15px; /* Reduced margin */
                font-size: 0.9em; /* Smaller input font */
                background-color: #21262d;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 6px; /* Slightly smaller border radius */
            }}
            .auth-form button {{
                width: 100%;
                padding: 10px;
                font-size: 1em;
                border-radius: 6px;
                background-color: #2563eb; /* Blue background */
                color: white;
                transition: background-color 0.2s;
            }}
            .auth-form button:hover {{
                background-color: #1d4ed8; /* Darker blue on hover */
            }}
            .auth-toggle-text {{
                color: #8b949e;
                margin-top: 15px; /* Reduced margin */
                font-size: 0.8em; /* Smaller text */
            }}
            .auth-toggle-text a {{
                color: #58a6ff;
                cursor: pointer;
                text-decoration: none;
            }}
            .auth-toggle-text a:hover {{
                text-decoration: underline;
            }}

            .auth-video-background {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                object-fit: cover;
                z-index: -2; /* Behind the body::before overlay */
            }}

            /* Theme Toggle Button */
            .theme-toggle {{
                background-color: #21262d;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 8px 12px;
                cursor: pointer;
                transition: background-color 0.2s, color 0.2s;
                font-size: 0.9em;
            }}
            .theme-toggle:hover {{
                background-color: #30363d;
                color: #79c0ff;
            }}
            .fixed-header .header-right {{
                display: flex;
                align-items: center;
                gap: 20px; /* Space between nav links and theme toggle */
            }}

            /* --- New Styles for Animated Auth Pages --- */

            /* Animated Gradient Background for Auth Pages */
            @keyframes gradient-animation {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}

            /* New Auth Page Layout */
            .auth-page-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                align-items: center;
                min-height: calc(100vh - 70px); /* Full height minus header */
                padding: 0 50px;
                gap: 40px;
            }}

            .auth-intro-text {{
                text-align: left;
                color: #e6edf3;
            }}

            .auth-intro-text h1 {{
                font-size: 4.5em; /* Larger font size */
                font-weight: bold;
                color: #e6edf3;
                margin-bottom: 20px;
                text-shadow: 0 0 15px rgba(88, 166, 255, 0.5);
            }}

            .auth-intro-text p {{
                font-size: 1.5em;
                color: #8b949e;
                max-width: 500px;
            }}

            /* Typing Animation */
            .typing-effect {{
                display: inline-block;
                overflow: hidden;
                white-space: nowrap;
                border-right: .12em solid #58a6ff; /* The typewriter cursor */
                animation: typing-untyping 4s steps(9, end) infinite;
            }}
            
            .typing-effect.p-typing {{
                /* Adjust timing for the longer paragraph */
                animation: typing-untyping 7s steps(54, end) infinite;
            }}

            /* A more realistic typing animation with a pause and blinking caret */
            @keyframes typing-untyping {{
                0%, 100% {{ 
                    width: 0; /* Start and end with no text */
                }}
                40%, 60% {{ 
                    width: 100%; /* Stay fully typed for a pause */
                }}
                /* Blinking effect during the pause */
                45% {{ border-color: transparent; }}
                50% {{ border-color: #58a6ff; }}
                55% {{ border-color: transparent; }}
            }}

            @keyframes blink-caret {{
                from, to {{ border-color: transparent; }}
                50% {{ border-color: #58a6ff; }}
            }}
            
            /* Ensure the auth form is centered in its column */
            .auth-container {{
                justify-content: center;
                width: 100%;
            }}

            /* Base styles for dark theme */
body {{
    /* ... existing styles ... */
    font-size: 15px; /* Reduced base font size for overall readability */
}}

/* Adjusted font sizes for various text elements */
h1, h2, h3, h4, h5, h6 {{
    /* ... existing styles ... */
    font-size: 1.8em; /* Example: h1 will be 1.8 * 15px = 27px */
    line-height: 1.3;
    margin-bottom: 0.8em; /* Slightly less spacing for headers */
}}

/* General text, paragraphs, list items */
p, li, span, div:not(.logo):not(.fixed-header):not(.fixed-footer):not(.auth-container):not(.auth-form):not(.auth-toggle-text):not(.nav-links):not(.footer-links):not(.copyright) {{
    color: #c0c0c0;
    font-size: 0.95em; /* Slightly smaller than base font size (0.95 * 15px = 14.25px) */
    line-height: 1.6; /* Adjusted line spacing */
    margin-bottom: 1em; /* Adjusted paragraph spacing */
}}

/* Specific adjustments for landing page titles */
.landing-title {{
    /* ... existing styles ... */
    font-size: 3em; /* Smaller than previous 3.5em */
    margin-bottom: 15px; /* Adjust spacing */
}}
.landing-subtitle {{
    /* ... existing styles ... */
    font-size: 1.3em; /* Smaller than previous 1.5em */
    margin-bottom: 30px; /* Adjust spacing */
}}

/* Specific adjustments for auth forms */
.auth-form h2 {{
    /* ... existing styles ... */
    font-size: 1.6em; /* Smaller heading in auth form */
}}
.auth-form p {{
    /* ... existing styles ... */
    font-size: 0.85em; /* Smaller text in auth form */
}}
.auth-form label {{
    /* ... existing styles ... */
    font-size: 0.8em; /* Smaller label font in auth form */
}}
.auth-form input {{
    /* ... existing styles ... */
    font-size: 0.85em; /* Smaller input font in auth form */
}}
.auth-form button {{
    /* ... existing styles ... */
    font-size: 0.9em; /* Smaller button font in auth form */
}}
.auth-toggle-text {{
    /* ... existing styles ... */
    font-size: 0.75em; /* Smaller toggle text in auth form */
}}

/* Adjustments for dashboard metrics */
[data-testid="stMetricValue"] {{
    font-size: 2.2rem; /* Adjusted for better fit */
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.9em; /* Adjusted for better fit */
}}
[data-testid="stMetricDelta"] {{
    font-size: 1.1rem; /* Adjusted for better fit */
}}

/* Adjustments for header/footer links */
.fixed-header .nav-links a {{
    font-size: 1em; /* Relative to body font size */
}}
.fixed-footer .footer-links a {{
    font-size: 0.85em; /* Slightly smaller for footer links */
}}
.fixed-footer .copyright {{
    font-size: 0.75em; /* Slightly smaller for copyright */
}}
        </style>
        {{% block head_extra %}}{{% endblock %}}
    </head>
    <body class="{{ 'light-theme' if session.get('theme') == 'light' else '' }}">
        {{% if session.get('logged_in') %}}
            {chatbot_assets['html']}
        {{% endif %}}
        <div class="fixed-header">
            <div class="logo">
                <img src="https://placehold.co/40x40/58a6ff/0d1117?text=CV" alt="CredVeda Logo">
                CredVeda
            </div>
            <div class="header-right">
                <div class="nav-links">
                    <a href="{{{{ url_for('home') }}}}">Home</a>
                    <a href="{{{{ url_for('dashboard_page') }}}}">Dashboard</a>
                    <a href="{{{{ url_for('ai_features_page') }}}}">AI Features</a>
                    <a href="{{{{ url_for('about_page') }}}}">About</a>
                    {{% if session.get('logged_in') %}}
                        <a href="{{{{ url_for('logout') }}}}">Logout ({{{{ session.get('username') }}}})</a>
                    {{% else %}}
                        <a href="{{{{ url_for('login') }}}}">Login</a>
                    {{% endif %}}
                </div>
                <button class="theme-toggle" onclick="toggleTheme()">
                    {{% if session.get('theme') == 'light' %}}
                        🌙 Dark Mode
                    {{% else %}}
                        ☀️ Light Mode
                    {{% endif %}}
                </button>
            </div>
        </div>

        <div class="main-content container">
            {{% block content %}}{{% endblock %}}
        </div>

        {{% block footer_content %}}
        <footer class="site-footer">
            <div class="footer-grid">
                <div class="footer-column">
                    <h4>CredVeda</h4>
                    <p>AI-Powered Credit Intelligence to help you make smarter, faster, and more transparent decisions in the global credit markets.</p>
                </div>
                <div class="footer-column">
                    <h4>Quick Links</h4>
                    <a href="{{{{ url_for('home') }}}}">Home</a>
                    <a href="{{{{ url_for('dashboard_page') }}}}">Dashboard</a>
                    <a href="{{{{ url_for('ai_features_page') }}}}">AI Features</a>
                    <a href="{{{{ url_for('about_page') }}}}">About Us</a>
                </div>
                <div class="footer-column">
                    <h4>Legal</h4>
                    <a href="{{{{ url_for('privacy_page') }}}}">Privacy Policy</a>
                    <a href="{{{{ url_for('terms_page') }}}}">Terms of Service</a>
                    <a href="{{{{ url_for('contact_page') }}}}">Contact Us</a>
                </div>
                <div class="footer-column">
                    <h4>Connect</h4>
                    <a href="#">LinkedIn</a>
                    <a href="#">Twitter</a>
                    <p class="mt-4">contact@credveda.com</p>
                </div>
            </div>
            <div class="footer-disclaimer">
                © 2025 CredVeda. All Rights Reserved. The information provided by this platform is for informational purposes only and does not constitute financial advice.
            </div>
        </footer>
        {{% endblock %}}

        {{% block scripts %}}{{% endblock %}}

        <script>
            {{% if session.get('logged_in') %}}
            {chatbot_assets['js']}
            {{% endif %}}
            // Theme toggle JavaScript
            function toggleTheme() {{
                const body = document.body;
                const currentTheme = body.classList.contains('light-theme') ? 'light' : 'dark';
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                
                // Send request to Flask backend to update session
                fetch('/set_theme', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ theme: newTheme }})
                }}).then(response => {{
                    if (response.ok) {{
                        body.classList.toggle('light-theme');
                        // Update button text
                        const themeToggleButton = document.querySelector('.theme-toggle');
                        if (newTheme === 'light') {{
                            themeToggleButton.innerHTML = '🌙 Dark Mode';
                        }} else {{
                            themeToggleButton.innerHTML = '☀️ Light Mode';
                        }}
                    }}
                }}).catch(error => console.error('Error toggling theme:', error));
            }}

            // Set initial theme based on session on page load
            document.addEventListener('DOMContentLoaded', () => {{
                const savedTheme = "{{{{ session.get('theme', 'dark') }}}}"; // Default to dark
                if (savedTheme === 'light') {{
                    document.body.classList.add('light-theme');
                }}
                // Update button text on load
                const themeToggleButton = document.querySelector('.theme-toggle');
                if (themeToggleButton) {{
                    if (savedTheme === 'light') {{
                        themeToggleButton.innerHTML = '🌙 Dark Mode';
                    }} else {{
                        themeToggleButton.innerHTML = '☀️ Light Mode';
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """



    # Login Page
    with open(os.path.join(template_dir, 'login.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Login - CredVeda{% endblock %}
{% block content %}
<video autoplay muted loop class="auth-video-background">
    <source src="{{ url_for('static', filename='hero_video.mp4') }}" type="video/mp4">
</video>
<div class="auth-page-container">
    <div class="auth-intro-text">
        <h1>CredVeda</h1>
        <p>Real-Time Explainable Credit Intelligence Platform</p>
    </div>
    <div class="auth-container">
        <div class="auth-form">
            <h2 class="text-2xl font-bold text-center text-white mb-2">Login to CredVeda</h2>
            <p class="text-center text-gray-400 mb-8">Please sign in to continue</p>
            <form method="post" action="{{ url_for('login') }}">
                {% if error %}
                    <div class="bg-red-900 border border-red-700 text-red-300 px-4 py-3 rounded relative mb-4 text-sm" role="alert">
                        <span class="block sm:inline">{{ error }}</span>
                    </div>
                {% endif %}
                <div class="mb-4">
                    <label class="block text-gray-200 text-sm font-bold mb-2" for="username">
                        Username
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline bg-gray-700 text-white" id="username" name="username" type="text" placeholder="username" required>
                </div>
                <div class="mb-6">
                    <label class="block text-gray-200 text-sm font-bold mb-2" for="password">
                        Password
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline bg-gray-700 text-white" id="password" name="password" type="password" placeholder="password" required>
                </div>
                <div class="flex items-center justify-between">
                    <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full" type="submit">
                        Sign In
                    </button>
                </div>
            </form>
            <p class="auth-toggle-text">Don't have an account? <a href="{{ url_for('signup') }}">Sign Up</a></p>
        </div>
    </div>
</div>
{% endblock %}
{% block footer_content %}{% endblock %}
        """)

    # Signup Page
    with open(os.path.join(template_dir, 'signup.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Sign Up - CredVeda{% endblock %}
{% block content %}
<video autoplay muted loop class="auth-video-background">
    <source src="{{ url_for('static', filename='hero_video.mp4') }}" type="video/mp4">
</video>
<div class="auth-page-container">
    <div class="auth-intro-text">
        <h1>CredVeda</h1>
        <p>Real-Time Explainable Credit Intelligence Platform</p>
    </div>
    <div class="auth-container">
        <div class="auth-form">
            <h2 class="text-2xl font-bold text-center text-white mb-2">Sign Up for CredVeda</h2>
            <p class="text-center text-gray-400 mb-8">Create your account</p>
            <form method="post" action="{{ url_for('signup') }}">
                {% if error %}
                    <div class="bg-red-900 border border-red-700 text-red-300 px-4 py-3 rounded relative mb-4 text-sm" role="alert">
                        <span class="block sm:inline">{{ error }}</span>
                    </div>
                {% endif %}
                {% if message %}
                    <div class="bg-green-900 border border-green-700 text-green-300 px-4 py-3 rounded relative mb-4 text-sm" role="alert">
                        <span class="block sm:inline">{{ message }}</span>
                    </div>
                {% endif %}
                <div class="mb-4">a
                    <label class="block text-gray-200 text-sm font-bold mb-2" for="username">
                        Username
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline bg-gray-700 text-white" id="username" name="username" type="text" placeholder="Choose a username" required>
                </div>
                <div class="mb-6">
                    <label class="block text-gray-200 text-sm font-bold mb-2" for="password">
                        Password
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline bg-gray-700 text-white" id="password" name="password" type="password" placeholder="Choose a password" required>
                </div>
                <div class="flex items-center justify-between">
                    <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full" type="submit">
                        Sign Up
                    </button>
                </div>
            </form>
            <p class="auth-toggle-text">Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
        </div>
    </div>
</div>
{% endblock %}
{% block footer_content %}{% endblock %}
        """)

    # Home Page
    with open(os.path.join(template_dir, 'home.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Home - CredVeda{% endblock %}
{% block content %}
<div style="padding-top: 100px; text-align: center;">
    <h1 class="landing-title">AI-Powered Credit Intelligence</h1>
    <p class="landing-subtitle">Leverage advanced AI to understand credit risks, simulate scenarios, and get intelligent insights</p>
    <a href="{{ url_for('dashboard_page') }}" class="btn-primary inline-block">Go to Dashboard →</a>
</div>
<video class="block mx-auto mt-8 rounded-lg shadow-lg" style="width: 100%; max-width: 1200px;" autoplay loop muted>
    <source src="{{ url_for('static', filename='business concept.mp4') }}" type="video/mp4">
    Your browser does not support the video tag.
</video>
<div class="mt-12 text-center">
    <h2 class="text-3xl font-bold mb-4">Ready to Transform Your Credit Analysis?</h2>
    {% if not session.get('logged_in') %}
    <p class="text-gray-400 text-lg mb-8">Join the future of transparent, AI-powered credit scoring</p>
    <a href="{{ url_for('signup') }}" class="btn-primary inline-block">Start Free Trial →</a>
    {% else %}
    <p class="text-gray-400 text-lg mb-8">You're already logged in. Explore the dashboard to get started.</p>
    <a href="{{ url_for('dashboard_page') }}" class="btn-primary inline-block">Go to Dashboard →</a>
    {% endif %}
</div>
{% endblock %}
        """)

    # Dashboard Page
    with open(os.path.join(template_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Dashboard - CredVeda{% endblock %}
{% block head_extra %}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
{% endblock %}
{% block content %}
<div class="container mx-auto py-8">
    <h1 class="text-3xl font-bold mb-8 text-center">Real-time Credit Insights Dashboard</h1>

    <!-- Metrics Row -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div class="bg-dark-card text-center p-6">
            <p class="text-xl text-gray-400">Current Credit Score (<span id="metric-ticker-display">{{ selected_ticker }}</span>)</p>
            <p class="text-5xl font-bold text-blue-accent my-2" id="current-score-value">{{ current_credit_score if current_credit_score is not none else 'N/A' }}</p>
            <p class="text-lg text-gray-400" id="delta-score-value">{{ delta_score }}</p>
        </div>
        <div class="bg-dark-card text-center p-6">
            <p class="text-xl text-gray-400">Market Volatility Index</p>
            <p class="text-5xl font-bold text-blue-accent my-2" id="current-vol-value">{{ current_volatility }}</p>
            <p class="text-lg text-gray-400" id="delta-vol-value">{{ delta_vol }}</p>
        </div>
        <div class="bg-dark-card text-center p-6">
            <p class="text-xl text-gray-400">Global Macro Indicator</p>
            <p class="text-5xl font-bold text-blue-accent my-2" id="current-macro-value">{{ current_macro }}</p>
            <p class="text-lg text-gray-400" id="delta-macro-value">{{ delta_macro }}</p>
        </div>
    </div>

    <!-- Ticker and Year Selection -->
    <div class="bg-dark-card p-6 mb-8">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <label for="ticker_select" class="block text-xl font-medium text-white mb-2">Select Company</label>
                <select id="ticker_select" name="ticker" class="mt-1 block w-full pl-3 pr-10 py-1 text-base border-gray-600 focus:outline-none focus:ring-blue-500 focus:border-blue-500 rounded-md bg-gray-700 text-white" onchange="handleFilterChange()">
                    {% for ticker, name in tickers.items() %}
                        <option value="{{ ticker }}" {% if ticker == selected_ticker %}selected{% endif %}>{{ name }}</option>
                    {% endfor %}
                </select>
            </div>
            <div>
                <label for="year_select" class="block text-xl font-medium text-white mb-2">Select Year</label>
                <select id="year_select" name="year" class="mt-1 block w-full pl-3 pr-10 py-1 text-base border-gray-600 focus:outline-none focus:ring-blue-500 focus:border-blue-500 rounded-md bg-gray-700 text-white" onchange="handleFilterChange()">
                    {% for year in available_years %}
                        <option value="{{ year }}" {% if year == selected_year %}selected{% endif %}>{{ year }}</option>
                    {% endfor %}
                </select>
            </div>
        </div>
    </div>

    <!-- Tabs for Detailed Analysis -->
    <div class="bg-dark-card p-6">
        <div class="flex border-b border-gray-700 mb-4">
            <button class="tab-button px-4 py-2 text-white border-b-2 border-blue-500" onclick="openTab(event, 'score-trend')">📈 Score Trend Analysis</button>
            <button class="tab-button px-4 py-2 text-gray-400 hover:text-white" onclick="openTab(event, 'explainability')">🔍 Explainability Insights</button>
            <button class="tab-button px-4 py-2 text-gray-400 hover:text-white" onclick="openTab(event, 'unstructured-feed')">📰 Unstructured Data Feed</button>
            <button class="tab-button px-4 py-2 text-gray-400 hover:text-white" onclick="openTab(event, 'global-snapshot')">🌍 Global Market Snapshot</button>
        </div>

        <!-- Tab Content: Score Trend -->
        <div id="score-trend" class="tab-content">
            <h2 class="text-2xl font-bold mb-4">Credit Score Trend Over Time</h2>
            <div class="bg-gray-800 p-4 rounded-lg">
                <canvas id="scoreTrendChart"></canvas>
            </div>
        </div>

        <!-- Tab Content: Explainability -->
        <div id="explainability" class="tab-content hidden">
            <h2 class="text-2xl font-bold mb-4">Why This Score? (Explainability Breakdown)</h2>
            <h3 class="text-xl font-semibold text-white mb-2">Feature Contributions for <span id="explain-ticker">{{ selected_ticker }}</span> on <span id="explain-date"></span></h3>
            <p class="text-gray-400 mb-4"><b>Base Probability (Average):</b> <span id="base-score-display"></span>/100. Features adjusted the score to <b><span id="predicted-score-display"></span>/100</b>.</p>
            <div class="bg-gray-800 p-4 rounded-lg">
                <canvas id="explainChart"></canvas>
            </div>
            <h3 class="text-xl font-semibold text-white mt-6 mb-2">Latest Event & Sentiment Insights:</h3>
            <ul id="event-insights-list" class="list-disc list-inside space-y-2 text-gray-400">
                <li>No insights available.</li>
            </ul>
        </div>

        <!-- Tab Content: Unstructured Data Feed -->
        <div id="unstructured-feed" class="tab-content hidden">
            <h2 class="text-2xl font-bold mb-4">Real-time Unstructured Data Feed</h2>
            <h3 class="text-xl font-semibold text-white mb-2">Recent News Headlines & Sentiment</h3>
            <p class="text-gray-400 mb-4"><b>Overall News Sentiment:</b> <span id="news-sentiment-display">N/A</span></p>
            <div id="news-headlines-list" class="space-y-4">
                <!-- News headlines will be loaded here -->
            </div>
            <h3 class="text-xl font-semibold text-white mt-6 mb-2">Latest Earnings Call Transcript Snippets & Sentiment</h3>
            <p class="text-gray-400 mb-4"><b>Overall Transcript Sentiment:</b> <span id="transcript-sentiment-display">N/A</span></p>
            <div id="transcript-snippets-list" class="space-y-4">
                <!-- Transcript snippets will be loaded here -->
            </div>
        </div>

        <!-- Tab Content: Global Market Snapshot -->
        <div id="global-snapshot" class="tab-content hidden">
            <h2 class="text-2xl font-bold mb-4">Global Market Snapshot</h2>
            <h3 class="text-xl font-semibold text-white mb-2">Overall Market Sentiment</h3>
            <p class="text-gray-400 mb-4"><b>Global Sentiment:</b> <span id="global-sentiment-value">N/A</span> <span id="global-sentiment-emoji"></span></p>
            <div class="w-full bg-gray-700 rounded-full h-2.5 mb-4">
                <div id="global-sentiment-progress" class="bg-blue-600 h-2.5 rounded-full" style="width: 0%"></div>
            </div>

            <h3 class="text-xl font-semibold text-white mt-6 mb-2">Top Movers (Illustrative)</h3>
            <div id="top-movers-table"></div>

            <h3 class="text-xl font-semibold text-white mt-6 mb-2">Key Macroeconomic Trends</h3>
            <div id="macro-trends-table"></div>
        </div>
    </div>

    <!-- Anomaly Alerts -->
    <div class="bg-dark-card p-6 mt-8">
        <h2 class="text-2xl font-bold mb-4 text-red-accent">🚨 Quantum Anomaly Alerts</h2>
        <p class="text-gray-400 mb-4">Real-time detection of significant shifts in creditworthiness signals.</p>
        <div id="anomaly-alerts-list" class="space-y-3">
            {% for alert in anomaly_alerts %}
            <div class="flex items-start p-3 rounded-lg" style="background-color: {{ alert.color }}20; border-left: 4px solid {{ alert.color }};">
                <div class="mr-3 flex-shrink-0">
                    <span class="font-bold px-2 py-1 rounded-md text-xs" style="background-color: {{ alert.color }}; color: #ffffff;">{{ alert.type }}</span>
                </div>
                <div class="flex-1">
                    <p class="text-sm text-white">{{ alert.message }}</p>
                    <p class="text-xs text-gray-400 mt-1">{{ alert.time }}</p>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

</div>
{% endblock %}

{% block scripts %}
<script>
    let scoreTrendChartInstance; // Renamed to avoid conflict
    let explainChartInstance; // Renamed to avoid conflict

    // Function to switch tabs
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tab-content");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        tablinks = document.getElementsByClassName("tab-button");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" border-b-2 border-blue-500", "");
            tablinks[i].className = tablinks[i].className.replace(" text-white", " text-gray-400 hover:text-white");
        }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " border-b-2 border-blue-500 text-white";
        evt.currentTarget.className = evt.currentTarget.className.replace(" text-gray-400 hover:text-white", "");
    }

    // Function to handle changes in ticker or year dropdowns
    function handleFilterChange() {
        const ticker = document.getElementById('ticker_select').value;
        const year = document.getElementById('year_select').value;
        window.location.href = `/dashboard?ticker=${ticker}&year=${year}`;
    }

    // Function to fetch data and update dashboard sections
    async function fetchAndUpdateDashboard(ticker, year) {
        document.getElementById('metric-ticker-display').innerText = ticker; // Update ticker in metric card
        document.getElementById('explain-ticker').innerText = ticker;
        
        // Fetch credit score trend
        fetch(`/api/credit_score_trend/${ticker}/${year}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) { console.error("Error fetching credit score trend:", data.error); return; }
                const ctx = document.getElementById('scoreTrendChart').getContext('2d');
                if (scoreTrendChartInstance) { scoreTrendChartInstance.destroy(); }
                scoreTrendChartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.dates,
                        datasets: [{
                            label: `${ticker} Credit Score (${year})`,
                            data: data.scores,
                            borderColor: 'rgb(121, 192, 255)',
                            backgroundColor: 'rgba(121, 192, 255, 0.2)',
                            tension: 0.1, pointRadius: 2,
                            pointBackgroundColor: 'rgb(121, 192, 255)'
                        }]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: {
                            x: { title: { display: true, text: 'Date' }, ticks: { autoSkip: true, maxTicksLimit: 10 } },
                            y: { title: { display: true, text: 'Credit Score (0-100)' }, min: 0, max: 100 }
                        },
                        plugins: { legend: { display: true } }
                    }
                });
            });

        // Fetch explainability data for the latest available date
        fetch(`/api/explainability/${ticker}/latest`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error("Error fetching explainability data:", data.error);
                    const explainChartContainer = document.getElementById('explainChart').parentElement;
                    explainChartContainer.innerHTML = `<p class="text-red-400 text-center">${data.error}</p>`;
                    return;
                }
                
                document.getElementById('explain-date').innerText = data.date;
                document.getElementById('base-score-display').innerText = data.base_score;
                document.getElementById('predicted-score-display').innerText = data.predicted_score;

                const ctxExplain = document.getElementById('explainChart').getContext('2d');
                if (explainChartInstance) { explainChartInstance.destroy(); }

                const features = data.explanation_data.map(d => d.Feature);
                const contributions = data.explanation_data.map(d => d.Contribution);
                
                explainChartInstance = new Chart(ctxExplain, {
                    type: 'bar',
                    data: {
                        labels: features,
                        datasets: [{
                            label: 'Feature Contribution to Score',
                            data: contributions,
                            backgroundColor: contributions.map(c => c >= 0 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'),
                            borderColor: contributions.map(c => c >= 0 ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)'),
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: 'y',
                        scales: {
                            x: { 
                                title: { display: true, text: 'Impact on Credit Score (SHAP Value)' },
                                ticks: { color: '#e6edf3' },
                                grid: { color: '#30363d' }
                            },
                            y: { 
                                title: { display: false },
                                ticks: { color: '#e6edf3' },
                                grid: { color: '#30363d' }
                            }
                        },
                        plugins: { 
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        let label = context.dataset.label || '';
                                        if (label) {
                                            label += ': ';
                                        }
                                        if (context.parsed.x !== null) {
                                            label += `Contribution: ${context.parsed.x.toFixed(4)}`;
                                        }
                                        return label;
                                    }
                                }
                            }
                        }
                    }
                });

                const eventList = document.getElementById('event-insights-list');
                eventList.innerHTML = '';
                if (data.events_insights && data.events_insights.length > 0) {
                    data.events_insights.forEach(event => {
                        const li = document.createElement('li');
                        li.innerHTML = event; // Assuming event is already formatted HTML
                        eventList.appendChild(li);
                    });
                } else {
                    eventList.innerHTML = '<li>No specific events or strong sentiment detected for this period.</li>';
                }
            })
            .catch(err => {
                console.error('Fetch error for explainability:', err);
                const explainChartContainer = document.getElementById('explainChart').parentElement;
                explainChartContainer.innerHTML = `<p class="text-red-400 text-center">Could not load explanation data. Please try again later.</p>`;
            });

        // Update Unstructured Data Feed (Sentiment values are passed, headlines are mock)
        const currentNewsSentiment = {{ current_news_sentiment | tojson }};
        const currentTranscriptSentiment = {{ current_transcript_sentiment | tojson }};

        if (currentNewsSentiment !== null) {
            document.getElementById('news-sentiment-display').innerText = `${currentNewsSentiment.toFixed(2)} ${currentNewsSentiment > 0.2 ? '😊 Positive' : (currentNewsSentiment < -0.2 ? '😞 Negative' : '😐 Neutral')}`;
        } else {
            document.getElementById('news-sentiment-display').innerText = 'N/A';
        }
        // Mock headlines remain static as they are not fetched from DB
        
        if (currentTranscriptSentiment !== null) {
            document.getElementById('transcript-sentiment-display').innerText = `${currentTranscriptSentiment.toFixed(2)} ${currentTranscriptSentiment > 0.1 ? '😊 Positive' : (currentTranscriptSentiment < -0.1 ? '😞 Negative' : '😐 Neutral')}`;
        } else {
            document.getElementById('transcript-sentiment-display').innerText = 'N/A';
        }
        // Mock snippets remain static

        // Update Global Market Snapshot (some data is live, some mock)
        updateGlobalSnapshot();
    }

    function updateGlobalSnapshot() {
        const globalSentimentValue = {{ market_sentiment | tojson }};
        document.getElementById('global-sentiment-value').innerText = globalSentimentValue.toFixed(2);
        let emoji = '😐';
        let color = 'orange';
        if (globalSentimentValue > 0.2) { emoji = '😊 Positive'; color = 'green'; }
        else if (globalSentimentValue < -0.2) { emoji = '😞 Negative'; color = 'red'; }
        document.getElementById('global-sentiment-emoji').innerText = emoji;
        document.getElementById('global-sentiment-emoji').style.color = color;
        document.getElementById('global-sentiment-progress').style.width = `${((globalSentimentValue + 1) / 2) * 100}%`;
        document.getElementById('global-sentiment-progress').style.backgroundColor = color;

        const topMoversData = [
            {Company: 'GOOGL', 'Change (%)': 3.5, Reason: 'Strong Earnings'},
            {Company: 'MSFT', 'Change (%)': 2.8, Reason: 'New Product Launch'},
            {Company: 'RELIANCE.NS', 'Change (%)': 1.9, Reason: 'Market Optimism'},
            {Company: 'ADANIENT.NS', 'Change (%)': -2.1, Reason: 'Regulatory Scrutiny'},
            {Company: 'TSLA', 'Change (%)': -4.2, Reason: 'Supply Chain Issues'}
        ];
        let topMoversHtml = '<table class="table-auto"><thead><tr><th>Company</th><th>Change (%)</th><th>Reason</th></tr></thead><tbody>';
        topMoversData.forEach(row => {
            const changeColor = row['Change (%)'] > 0 ? 'text-green-accent' : 'text-red-accent';
            topMoversHtml += `<tr><td>${row.Company}</td><td class="${changeColor}">${row['Change (%)'].toFixed(2)}%</td><td>${row.Reason}</td></tr>`;
        });
        topMoversHtml += '</tbody></table>';
        document.getElementById('top-movers-table').innerHTML = topMoversHtml;

        const latestMacroIndicator = {{ latest_macro_indicator | tojson }};
        const macroTrendsData = [
            {Indicator: 'FRED DGS10 (10-Year Yield)', Value: latestMacroIndicator !== null ? latestMacroIndicator.toFixed(2) : 'N/A', Trend: 'N/A', Impact: 'N/A'},
            {Indicator: 'Inflation Rate', Value: '2.80%', Trend: 'Stable', Impact: 'Consumer Spending Pressure'},
            {Indicator: 'Global Trade Volume', Value: '1.05', Trend: 'Fluctuating', Impact: 'Supply Chain Risk'}
        ];
        let macroTrendsHtml = '<table class="table-auto"><thead><tr><th>Indicator</th><th>Value</th><th>Trend</th><th>Impact</th></tr></thead><tbody>';
        macroTrendsData.forEach(row => {
            macroTrendsHtml += `<tr><td>${row.Indicator}</td><td>${row.Value}</td><td>${row.Trend}</td><td>${row.Impact}</td></tr>`;
        });
            macroTrendsHtml += '</tbody></table>';
        document.getElementById('macro-trends-table').innerHTML = macroTrendsHtml;
    }

    // Initial load for dashboard
    document.addEventListener('DOMContentLoaded', () => {
        const urlParams = new URLSearchParams(window.location.search);
        const initialTicker = urlParams.get('ticker') || document.getElementById('ticker_select').value;
        const initialYear = urlParams.get('year') || document.getElementById('year_select').value;
        
        document.getElementById('ticker_select').value = initialTicker; // Set selectbox value
        document.getElementById('year_select').value = initialYear; // Set selectbox value

        fetchAndUpdateDashboard(initialTicker, initialYear);
        openTab(event, 'score-trend'); // Open first tab by default
    });
</script>
{% endblock %}
        """)

    # AI Features Page
    with open(os.path.join(template_dir, 'ai_features.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}AI Features - CredVeda{% endblock %}
{% block content %}
<div class="container mx-auto py-8">
    <h1 class="text-3xl font-bold mb-8 text-center">Our AI-Powered Features</h1>
    <div class="bg-dark-card p-6">
        <p class="text-lg text-gray-400 mb-6">
        CredVeda is built on a foundation of advanced Artificial Intelligence and Machine Learning to deliver deep, actionable credit insights. Our platform is designed to go beyond traditional metrics, providing a holistic and forward-looking view of creditworthiness.
        </p>

        <h3 class="text-2xl font-bold text-blue-accent mb-3">Predictive Credit Scoring with XGBoost</h3>
        <p class="text-gray-400 mb-4">
        The core of our platform is a powerful XGBoost (Extreme Gradient Boosting) model. This sophisticated machine learning algorithm is trained on a diverse dataset encompassing historical market data, fundamental financial indicators, and macroeconomic trends. It excels at uncovering complex, non-linear relationships within the data to produce highly accurate and dynamic credit scores, offering a significant improvement over static, traditional rating methods.
        </p>

        <h3 class="text-2xl font-bold text-blue-accent mb-3">Transparent Insights with Explainable AI (XAI)</h3>
        <p class="text-gray-400 mb-4">
        We eliminate the "black box" problem common in AI. By integrating SHAP (SHapley Additive exPlanations), our platform provides a clear, intuitive breakdown of every credit score. Users can see exactly how much each factor—such as market volatility, news sentiment, or a specific financial ratio—contributed to the final score. This transparency builds trust and empowers users to make decisions with confidence.
        </p>

        <h3 class="text-2xl font-bold text-blue-accent mb-3">Natural Language Processing (NLP) for Real-Time Sentiment Analysis</h3>
        <p class="text-gray-400 mb-4">
        Financial health isn't just about numbers. Our NLP engine analyzes unstructured data from news articles and earnings call transcripts to gauge market and company-specific sentiment. This qualitative data is quantified and fed into our model, allowing the platform to capture the impact of breaking news and management tone—factors that traditional models often miss until it's too late.
        </p>

        <h3 class="text-2xl font-bold text-blue-accent mb-3">Intelligent Anomaly Detection</h3>
        <p class="text-gray-400 mb-4">
        Our system continuously monitors key risk indicators to provide timely "Quantum Anomaly Alerts." This feature uses a rule-based engine to flag unusual events, such as:
        <ul class="list-disc list-inside text-gray-400 ml-4 mt-2">
            <li>Sudden spikes in market volatility.</li>
            <li>Significant drops in news or transcript sentiment.</li>
            <li>Detection of critical events related to debt, growth, or legal issues from text sources.</li>
            <li>Abrupt changes in the predicted credit score.</li>
        </ul>
        </p>
    </div>
</div>
{% endblock %}
        """)

    # About Page
    with open(os.path.join(template_dir, 'about.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}About Us - CredVeda{% endblock %}
{% block content %}
<div class="container mx-auto py-8">
    <h1 class="text-3xl font-bold mb-8 text-center">About Us</h1>
    <div class="bg-dark-card p-6 rounded-lg shadow-lg">
        <h2 class="text-2xl font-bold text-blue-accent mb-4">Our Mission: To Bring Clarity and Confidence to Credit Risk Analysis</h2>
        <p class="text-gray-400 mb-6">
            In today's volatile economic landscape, traditional credit assessment models are no longer sufficient. Relying on static, historical data often means missing the critical, forward-looking signals that truly define a company's financial health. Decisions are made with an incomplete picture, exposing institutions to unforeseen risks.
        </p>
        <p class="text-gray-400 mb-6">
            At CredVeda, we are changing that.
        </p>
        <p class="text-gray-400 mb-6">
            We are a team of financial analysts, data scientists, and software engineers dedicated to building the future of credit intelligence. We founded this platform on a simple but powerful belief: that the best financial decisions are made with a clear, transparent, and dynamic understanding of risk.
        </p>
        <p class="text-gray-400 mb-6">
            Our platform moves beyond outdated methods by harnessing the power of artificial intelligence to analyze a vast spectrum of data in real-time. We combine structured financial metrics with unstructured market signals—from news sentiment to macroeconomic trends—to generate a holistic, forward-looking credit score.
        </p>
        <p class="text-gray-400 mb-6">
            But we don't just provide a number. We provide understanding. Through our commitment to Explainable AI (XAI), we deliver the key drivers behind every score, empowering our clients not just to see the risk, but to understand it. Our goal is to replace uncertainty with insight, enabling you to navigate the complexities of the market with confidence.
        </p>
    </div>

    <div class="bg-dark-card p-6 rounded-lg shadow-lg mt-8">
        <h2 class="text-2xl font-bold text-blue-accent mb-4">Our AI-Powered Platform: The Technology Driving Smarter Decisions</h2>
        <p class="text-gray-400 mb-6">
            Our platform is built on a sophisticated AI engine designed to deliver unparalleled depth and accuracy in credit risk assessment. We go beyond the surface to provide a multi-dimensional view of financial health.
        </p>
        
        <h3 class="text-xl font-bold text-white mb-3">1. Hybrid Data Fusion</h3>
        <p class="text-gray-400 mb-4">Our models don't just look at balance sheets. We synthesize thousands of data points from diverse sources, including:</p>
        <ul class="list-disc list-inside text-gray-400 mb-4 ml-4">
            <li><b>Structured Financial Data:</b> In-depth analysis of company fundamentals, historical performance, and key financial ratios.</li>
            <li><b>Unstructured Market Signals:</b> Real-time processing of global news, financial reports, and sentiment analysis to capture the market's perception and identify emerging risks or opportunities before they impact the numbers.</li>
            <li><b>Macroeconomic Indicators:</b> Integration of key economic trends to understand how broader market forces will affect a company's performance.</li>
        </ul>

        <h3 class="text-xl font-bold text-white mb-3">2. Advanced Predictive Modeling</h3>
        <p class="text-gray-400 mb-4">At the core of our platform is a suite of advanced machine learning models, including Gradient Boosting, Random Forests, and custom-trained Neural Networks. Unlike traditional models that are purely retrospective, our algorithms are trained to identify complex, non-linear patterns and forecast future creditworthiness with a high degree of accuracy. The system continuously learns and adapts as new data becomes available.</p>

        <h3 class="text-xl font-bold text-white mb-3">3. Explainable AI (XAI) for Full Transparency</h3>
        <p class="text-gray-400 mb-4">A black-box credit score is a liability. Our commitment to transparency is powered by Explainable AI. For every score we generate, our platform provides a clear, concise breakdown of the top contributing factors (both positive and negative). This allows you to:</p>
        <ul class="list-disc list-inside text-gray-400 mb-4 ml-4">
            <li><b>Trust the Score:</b> Understand the "why" behind the "what."</li>
            <li><b>Conduct Deeper Due Diligence:</b> Immediately focus on the most critical variables affecting a company's risk profile.</li>
            <li><b>Communicate with Confidence:</b> Justify decisions to stakeholders with clear, data-backed evidence.</li>
        </ul>

        <h3 class="text-xl font-bold text-white mb-3">4. Real-Time Monitoring and Alerts</h3>
        <p class="text-gray-400 mb-4">Risk is not static. Our platform operates in near real-time, constantly ingesting new data and re-evaluating scores. This ensures you are always working with the most current assessment of a company's financial standing, allowing you to react swiftly to changing conditions.</p>
    </div>
</div>
{% endblock %}
        """)

    # Contact Page
    with open(os.path.join(template_dir, 'contact.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Contact - CredTech{% endblock %}
{% block content %}
<div class="container mx-auto py-8">
    <h1 class="text-3xl font-bold mb-8 text-center">Contact Us</h1>
    <div class="bg-dark-card p-6 max-w-lg mx-auto">
        <p class="text-lg text-gray-400 mb-4">
        Have questions, feedback, or want to explore collaboration opportunities with CredTech? We'd love to hear from you!
        </p>
        <div class="text-gray-400 space-y-3">
            <p><b>Email:</b> contact@credveda.com</p>
            <p><b>Phone:</b> +91-XXX-XXXXXXX</p>
            <p><b>Address:</b> IIT Kanpur, Kanpur, Uttar Pradesh, India</p>
            <p>Our team is dedicated to providing you with the best credit intelligence solutions. Reach out to us for any inquiries!</p>
        </div>
    </div>
</div>
{% endblock %}
        """)

    # Privacy Page
    with open(os.path.join(template_dir, 'privacy.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Privacy Policy - CredVeda{% endblock %}
{% block content %}
<div class="container mx-auto py-8">
    <h1 class="text-3xl font-bold mb-8 text-center">Privacy Policy</h1>
    <div class="bg-dark-card p-6 rounded-lg shadow-lg">
        <p class="text-sm text-gray-500 mb-6">Last Updated: August 21, 2025</p>

        <h2 class="text-2xl font-bold text-blue-accent mb-3">1. Introduction</h2>
        <p class="text-gray-400 mb-4">
            Welcome to CredVeda. We are committed to protecting the privacy and security of our users' data. This Privacy Policy outlines how we collect, use, disclose, and safeguard your information when you visit our website and use our services.
        </p>

        <h2 class="text-2xl font-bold text-blue-accent mb-3">2. Information We Collect</h2>
        <p class="text-gray-400 mb-4">We may collect information about you in a variety of ways. The information we may collect on the Site includes:</p>
        <ul class="list-disc list-inside text-gray-400 mb-4 ml-4">
            <li><b>Personal Data:</b> Personally identifiable information, such as your username and hashed password, that you voluntarily give to us when you register with the Site. We do not store plain-text passwords.</li>
            <li><b>Derivative Data:</b> Information our servers automatically collect when you access the Site, such as your IP address, your browser type, your operating system, your access times, and the pages you have viewed directly before and after accessing the Site.</li>
        </ul>

        <h2 class="text-2xl font-bold text-blue-accent mb-3">3. Use of Your Information</h2>
        <p class="text-gray-400 mb-4">Having accurate information about you permits us to provide you with a smooth, efficient, and customized experience. Specifically, we may use information collected about you via the Site to:</p>
        <ul class="list-disc list-inside text-gray-400 mb-4 ml-4">
            <li>Create and manage your account.</li>
            <li>Monitor and analyze usage and trends to improve your experience with the Site.</li>
            <li>Maintain the security of our Site and services.</li>
            <li>Respond to user inquiries and provide customer support.</li>
        </ul>

        <h2 class="text-2xl font-bold text-blue-accent mb-3">4. Disclosure of Your Information</h2>
        <p class="text-gray-400 mb-4">We do not share, sell, rent, or trade your personal information with third parties for their commercial purposes. We may share information we have collected about you in certain situations:</p>
        <ul class="list-disc list-inside text-gray-400 mb-4 ml-4">
            <li><b>By Law or to Protect Rights:</b> If we believe the release of information about you is necessary to respond to legal process, to investigate or remedy potential violations of our policies, or to protect the rights, property, and safety of others, we may share your information as permitted or required by any applicable law, rule, or regulation.</li>
            <li><b>Third-Party Service Providers:</b> We may share your information with third parties that perform services for us or on our behalf, including data analysis, hosting services, and customer service.</li>
        </ul>

        <h2 class="text-2xl font-bold text-blue-accent mb-3">5. Security of Your Information</h2>
        <p class="text-gray-400 mb-4">
            We use administrative, technical, and physical security measures to help protect your personal information. We store user passwords in a hashed format and take all reasonable precautions to secure your data. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable, and no method of data transmission can be guaranteed against any interception or other type of misuse.
        </p>

        <h2 class="text-2xl font-bold text-blue-accent mb-3">6. Your Rights</h2>
        <p class="text-gray-400 mb-4">
            You have the right to access, correct, or delete your personal information at any time. You may review or change the information in your account or terminate your account by logging into your account settings and updating your account.
        </p>

        <h2 class="text-2xl font-bold text-blue-accent mb-3">7. Contact Us</h2>
        <p class="text-gray-400 mb-4">
            If you have questions or comments about this Privacy Policy, please contact us at: <a href="mailto:credVeda@gmail.com" class="text-blue-accent">credVeda@gmail.com</a>
        </p>
    </div>
</div>
{% endblock %}
        """)

    # Terms Page
    with open(os.path.join(template_dir, 'terms.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Terms of Service - CredVeda{% endblock %}
{% block content %}
<div class="container mx-auto py-8">
    <h1 class="text-3xl font-bold mb-8 text-center">Terms of Service</h1>
    <div class="bg-dark-card p-6">
        <p class="text-lg text-gray-400 mb-4">
        Welcome to CredVeda. By accessing or using our AI-Powered Credit Intelligence Platform, you agree to comply with and be bound by these Terms of Service. Please review them carefully.
        </p>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">1. Acceptance of Terms</h3>
        <p class="text-gray-400 mb-4">
        By using the CredVeda platform, you signify your acceptance of these Terms of Service. If you do not agree to these terms, please do not use our platform.
        </p>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">2. Service Description</h3>
        <p class="text-gray-400 mb-4">
        CredVeda provides an AI-powered platform for credit risk assessment and financial intelligence. The information provided is for informational and analytical purposes only and does not constitute financial, investment, legal, or professional advice.
        </p>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">3. User Responsibilities</h3>
        <ul class="list-disc list-inside text-gray-400 mb-4 ml-4">
            <li>You are responsible for maintaining the confidentiality of your account credentials.</li>
            <li>You agree to use the platform only for lawful purposes and in accordance with these terms.</li>
            <li>You acknowledge that all investment decisions are solely your responsibility.</li>
        </ul>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">4. Intellectual Property</h3>
        <p class="text-gray-400 mb-4">
        All content, features, and functionality on the CredVeda platform, including text, graphics, logos, and software, are the exclusive property of CredVeda and are protected by international copyright, trademark, patent, trade secret, and other intellectual property or proprietary rights laws.
        </p>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">5. Disclaimer of Warranties</h3>
        <p class="text-gray-400 mb-4">
        The platform is provided on an "as is" and "as available" basis. CredVeda makes no warranties, expressed or implied, regarding the accuracy, completeness, reliability, or availability of the platform or its content.
        </p>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">6. Limitation of Liability</h3>
        <p class="text-gray-400 mb-4">
        In no event shall CredVeda be liable for any indirect, incidental, special, consequential, or punitive damages, including without limitation, loss of profits, data, use, goodwill, or other intangible losses, resulting from (i) your access to or use of or inability to access or use the platform; (ii) any conduct or content of any third party on the platform.
        </p>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">7. Governing Law</h3>
        <p class="text-gray-400 mb-4">
        These Terms shall be governed and construed in accordance with the laws of India, without regard to its conflict of law provisions.
        </p>
        <h3 class="text-2xl font-bold text-blue-accent mb-3">8. Changes to Terms</h3>
        <p class="text-gray-400 mb-4">
        We reserve the right, at our sole discretion, to modify or replace these Terms at any time. We will provide at least 30 days' notice prior to any new terms taking effect.
        </p>
        <p class="text-gray-400 mb-4">
        For any questions regarding these Terms of Service, please contact us.
        </p>
    </div>
</div>
{% endblock %}
        """)

    # Error Page (for data loading issues)
    with open(os.path.join(template_dir, 'error.html'), 'w', encoding='utf-8') as f:
        f.write("""
{% extends "base.html" %}
{% block title %}Error - CredVeda{% endblock %}
{% block content %}
<div class="container mx-auto py-8 text-center">
    <h1 class="text-3xl font-bold mb-4 text-red-accent">An Error Occurred!</h1>
    <p class="text-lg text-gray-400 mb-6">{{ message }}</p>
    <p class="text-gray-400">Please ensure your data ingestion pipeline has run successfully and try again.</p>
    <a href="{{ url_for('home') }}" class="btn-primary inline-block mt-8">Go to Home</a>
</div>
{% endblock %}
        """)

    # Create base.html last as it uses other templates
    with open(os.path.join(template_dir, 'base.html'), 'w', encoding='utf-8') as f:
        f.write(BASE_TEMPLATE)

    print("Templates directory and files created successfully.")

if __name__ == '__main__':
    # Initialize user database and load ML components once when the app starts
    init_user_db()
    load_ml_components()
    load_all_historical_features() # Load data into global df_raw

    setup_templates() # Create HTML template files

    print("\n--- Starting CredVeda AI-Powered Credit Intelligence Dashboard (Flask) ---")
    print(f"To login, you can sign up for a new account or use the default (if created):")
    print(f"  Username: admin (if you manually add it to DB)")
    print(f"  Password: password (if you manually add it to DB)")
    print("Open your browser and go to: http://127.0.0.1:5000/login")
    print("-----------------------------------------------------------------------")
    app.run(debug=True) # debug=True allows auto-reloading on code changes
