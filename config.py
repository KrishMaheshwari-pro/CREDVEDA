# config.py
"""
Central configuration file for the Credit Intelligence Platform.
"""

# --- API KEYS ---
NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE" # Placeholder for GitHub
API_NINJAS_KEY = "YOUR_API_NINJAS_KEY_HERE" # Placeholder for GitHub


# --- DATA SOURCES ---
TICKERS = [
    # --- US Market ---

    # Mega-Cap Tech & Growth
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "NFLX",
    "ADBE", "CRM", "ORCL", "SAP", "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU",

    # Financials & Payments
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "PYPL", "AXP",
    "BLK", "BRK-B", "SPGI", "SCHW", "COF", "USB", "PNC",

    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "LLY", "ABBV", "TMO", "MDT", "DHR",
    "GILD", "AMGN", "BMY", "ABT", "CVS", "CI", "ANTM", "ISRG", "SYK",

    # Consumer Discretionary & Staples
    "WMT", "COST", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "DIS",
    "KO", "PEP", "PG", "PM", "MO", "CL", "KMB", "MDLZ",

    # Industrials & Energy
    "CAT", "BA", "DE", "HON", "GE", "UNP", "UPS", "FDX", "LMT", "RTX",
    "XOM", "CVX", "SLB", "COP", "EOG",

    # Communications & Other
    "VZ", "T", "CMCSA", "IBM", "ACN", "NEE", "DUK", "SO",

    # More S&P 500 Components
    "", "AOS", "AES", "AFL", "A", "APD", "AKAM", "ALB", "ARE", "ALGN",
    "ALLE", "LNT", "ALL", "AMCR", "AEE", "AEP", "AIG", "AMT", "AWK", "AMP",
    "AME", "APH", "ADI", "ANSS", "AON", "APA", "AMAT", "APTV",
    "ADM", "ANET", "AJG", "AIZ", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY",
    "BKR", "BALL", "BDX", "WRB", "BR", "BSX", "BBY", "BIO", "TECH",
    "BIIB", "BK", "BAX", "BBWI", "BWA", "BXP", "BMY",
    "CHRW", "CDMMMNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX",
    "CCL", "CARR", "CTLT", "CBOE", "CBRE", "CDW", "CE", "CNC",
    "CNP", "CDAY", "CF", "CRL", "CHTR", "CMG", "CB",
    "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME",
    "CMS", "CTSH", "CL", "CMCSA", "CMA", "CAG", "COP", "ED",
    "STZ", "COO", "CPRT", "GLW", "CTVA", "CTRA", "CCI", "CSX",
    "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY",
    "DVN", "DXCM", "FANG", "DLR", "DFS", "DIS", "DG", "DLTR", "D",
    "DPZ", "DOV", "DOW", "DTE", "DUK", "DD", "DXC", "EMN", "ETN",
    "EBAY", "ECL", "EIX", "EW", "EA", "EL", "LLY", "EMR", "ENPH",
    "ETR", "EOG", "EPAM", "EFX", "EQIX", "EQR", "ESS", "ELV", "ETSY",
    "RE", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV",
    "FDS", "FAST", "FRT", "FDX", "FITB", "FSLR", "FE", "FIS", "FISV",
    "FLT", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX",
    "GRMN", "IT", "GE", "GNRC", "GD", "GIS", "GM", "GPC", "GILD",
    "GL", "GPN", "GS", "HAL", "HIG", "HAS", "HCA", "PEAK", "HSIC",
    "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST",
    "HWM", "HPQ", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW",
    "ILMN", "INCY",
    
    # --- Indian Market (NSE) --- (KEEP COMMENTED)
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "RELIANCE.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ADANIGREEN.NS",
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    "ULTRACEMCO.NS", "GRASIM.NS", "LT.NS", "AMBUJACEM.NS",
    "ASIANPAINT.NS", "PIDILITIND.NS", "BERGEPAINT.NS",
    "TITAN.NS", "DMART.NS", "HAVELLS.NS",
    "BHARTIARTL.NS", "INDIGO.NS", "ZEEL.NS",
]


MACRO_INDICATOR_FRED = "DGS10"


# --- DATA INGESTION & FEATURE ENGINEERING ---
HISTORY_PERIOD = "2y"
HISTORY_INTERVAL = "1d"
ROLL_WINDOW_VOL = 10
SMA_SHORT = 10
SMA_LONG = 50
RSI_PERIOD = 14
MACD_SHORT = 12
MACD_LONG = 26

# --- UNSTRUCTURED DATA ANALYSIS ---
EVENT_KEYWORDS = {
    "Negative_Debt": ["debt restructuring", "covenant breach", "default risk"],
    "Negative_Demand": ["declining demand", "weak sales", "reduced guidance", "headwinds"],
    "Negative_Legal": ["lawsuit", "investigation", "regulatory action", "fine"],
    "Positive_Growth": ["record revenue", "strong growth", "expansion", "new partnership", "beating expectations"],
    "Positive_Innovation": ["breakthrough", "new product", "successful launch", "innovation"]
}


# --- MODEL TRAINING ---
PREDICTION_HORIZON_DAYS = 30
FEATURE_COLS = [
    "Return", "Volatility", "TrendRatio", "RSI", "MACD", "Liquidity",
    "PE", "PB", "MarketCap", "Beta", "ProfitMargin",
    "MacroIndicator",
    "NewsSentiment", "TranscriptSentiment",
    "Negative_Debt", "Negative_Demand", "Negative_Legal",
    "Positive_Growth", "Positive_Innovation"
]

# --- DATABASE ---
DB_NAME = "credit_intelligence.db"
