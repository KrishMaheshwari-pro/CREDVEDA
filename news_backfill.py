"""
Backfill REAL news sentiment into historical_features.NewsSentiment using NewsAPI
(free tier ~100 req/day). Targeted UPDATE only — preserves prices/fundamentals/
macro/transcripts. Run:  python news_backfill.py
"""
import sys, time, sqlite3
sys.stdout.reconfigure(encoding="utf-8")
import requests, numpy as np, config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Curated, recognizable universe (kept under the NewsAPI free daily limit).
CURATED = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","NFLX","ADBE","CRM","ORCL",
    "INTC","AMD","QCOM","AVGO","TXN","CSCO","IBM","ACN",
    "JPM","BAC","WFC","GS","MS","C","V","MA","AXP","BLK","SCHW","COF","USB","SPGI",
    "JNJ","UNH","PFE","MRK","LLY","ABBV","TMO","MDT","AMGN","BMY","ABT","CVS",
    "WMT","COST","HD","NKE","MCD","SBUX","TGT","LOW","DIS","KO","PEP","PG","MDLZ",
    "CAT","BA","GE","HON","UPS","FDX","LMT","RTX","XOM","CVX","COP",
    "VZ","T","CMCSA","NEE","DUK",
    "TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","LT.NS",
]

conn = sqlite3.connect(config.DB_NAME)
db_tickers = {r[0] for r in conn.execute("SELECT DISTINCT Ticker FROM historical_features")}
targets = [t for t in CURATED if t in db_tickers]
print(f"{len(targets)} of {len(CURATED)} curated tickers found in DB. Fetching news...\n")

analyzer = SentimentIntensityAnalyzer()
updated = 0
for t in targets:
    q = t.split(".")[0]
    try:
        r = requests.get("https://newsapi.org/v2/everything",
                         params={"q": q, "language": "en", "sortBy": "publishedAt",
                                 "pageSize": 20, "apiKey": config.NEWS_API_KEY}, timeout=20)
        j = r.json()
        if j.get("status") != "ok":
            print(f"{t:12} SKIP ({j.get('code','')} {j.get('message','')[:50]})")
            if j.get("code") == "rateLimited":
                break
            continue
        arts = j.get("articles", [])
        scores = [analyzer.polarity_scores(a["title"])["compound"] for a in arts if a.get("title")]
        val = round(float(np.mean(scores)), 4) if scores else 0.0
        conn.execute("UPDATE historical_features SET NewsSentiment=? WHERE Ticker=?", (val, t))
        conn.commit()
        updated += 1
        print(f"{t:12} {val:+.3f}  ({len(scores)} headlines)")
        time.sleep(0.2)
    except Exception as e:
        print(f"{t:12} ERR {e}")

conn.close()
print(f"\nDone. Updated NewsSentiment for {updated} tickers with real news.")
