import os
import time
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config
import requests
import re
from bs4 import BeautifulSoup # Import BeautifulSoup for web scraping

# --- Global constants for safe handling ---
FUNDAMENTAL_KEYS = ["PE", "PB", "MarketCap", "Beta", "ProfitMargin"]

# --- Technical Indicator functions ---
def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, short_period: int, long_period: int) -> pd.Series:
    exp1 = series.ewm(span=short_period, adjust=False).mean()
    exp2 = series.ewm(span=long_period, adjust=False).mean()
    return exp1 - exp2

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(config.ROLL_WINDOW_VOL).std()
    df["SMA_Short"] = df["Close"].rolling(config.SMA_SHORT).mean()
    df["SMA_Long"] = df["Close"].rolling(config.SMA_LONG).mean()
    df["TrendRatio"] = df["SMA_Short"] / df["SMA_Long"]
    df["RSI"] = compute_rsi(df["Close"], config.RSI_PERIOD)
    df["MACD"] = compute_macd(df["Close"], config.MACD_SHORT, config.MACD_LONG)
    df["Liquidity"] = df["Volume"]
    return df

# --- Data Fetching Functions ---
def fetch_macro_data():
    """
    Fetches macroeconomic data from FRED.
    Includes a retry mechanism to handle temporary network issues.
    """
    print("Fetching macroeconomic data...")
    for attempt in range(3):  # Try up to 3 times
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=5)
            macro_data = web.DataReader(config.MACRO_INDICATOR_FRED, 'fred', start=start_date, end=end_date)
            macro_data.rename(columns={config.MACRO_INDICATOR_FRED: "MacroIndicator"}, inplace=True)
            print("Macroeconomic data fetched successfully.")
            return macro_data.ffill()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: Could not fetch macro data: {e}")
            if attempt < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
    print("Failed to fetch macro data after multiple attempts.")
    return pd.DataFrame()

def fetch_news_sentiment(ticker: str):
    if not config.NEWS_API_KEY or config.NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        print("[INFO] News API key not configured. Skipping news sentiment analysis.")
        return 0.0
    try:
        newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)
        # Removed 'timeout' argument as it's not supported by all NewsApiClient versions
        headlines = newsapi.get_everything(
            q=ticker.split('.')[0], language='en',
            sort_by='publishedAt', page_size=20
        )
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(a['title'])['compound'] for a in headlines['articles'] if a['title']]
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"[WARN] Could not fetch news sentiment for {ticker}: {e}")
        return 0.0

def fetch_fundamentals(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        return {
            "PE": info.get("trailingPE"),
            "PB": info.get("priceToBook"),
            "MarketCap": info.get("marketCap"),
            "Beta": info.get("beta"),
            "ProfitMargin": info.get("profitMargins")
        }
    except Exception as e:
        print(f"[WARN] Could not fetch fundamentals for {ticker}: {e}")
        return {k: np.nan for k in FUNDAMENTAL_KEYS}

# --- Transcript Analysis ---
def analyze_text_for_events(text: str, analyzer: SentimentIntensityAnalyzer):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentiment_scores = [analyzer.polarity_scores(sentence)['compound'] for sentence in sentences]
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
    event_flags = {key: 0 for key in config.EVENT_KEYWORDS.keys()}
    trigger_snippet = ""
    text_lower = text.lower()
    for event_type, keywords in config.EVENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                event_flags[event_type] = 1
                try:
                    snippet_sentence = next(s for s in sentences if keyword in s.lower())
                    trigger_snippet = f"[{event_type.replace('_', ' ')}]: ...{snippet_sentence.strip()}..."
                except StopIteration:
                    pass
                if trigger_snippet:
                    break
        if trigger_snippet:
            break
    return avg_sentiment, event_flags, trigger_snippet

def fetch_transcript_and_analyze(ticker: str):
    """
    Fetches and analyzes earnings call transcripts for US companies using API-Ninjas.
    """
    print(f"Fetching transcript for {ticker} using API-Ninjas...")
    base_ticker = ticker.split('.')[0]
    if not hasattr(config, 'API_NINJAS_KEY') or not config.API_NINJAS_KEY or config.API_NINJAS_KEY == "YOUR_API_NINJAS_KEY_HERE":
        print("[INFO] API Ninjas key not configured. Skipping transcript analysis for US companies.")
        return 0.0, {key: 0 for key in config.EVENT_KEYWORDS.keys()}, ""
    try:
        api_url = f'https://api.api-ninjas.com/v1/earningstranscript?ticker={base_ticker}'
        response = requests.get(api_url, headers={'X-Api-Key': config.API_NINJAS_KEY}, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data or 'transcript' not in data:
            print(f"[INFO] No transcript found for {ticker} via API-Ninjas.")
            return 0.0, {key: 0 for key in config.EVENT_KEYWORDS.keys()}, ""
        transcript_text = data.get('transcript', '')
        analyzer = SentimentIntensityAnalyzer()
        sentiment, events, snippet = analyze_text_for_events(transcript_text, analyzer)
        print(f"Transcript analysis for {ticker} complete. Sentiment: {sentiment:.2f}")
        if snippet:
            print(f"  -> Detected Event: {snippet[:80]}...")
        return sentiment, events, snippet
    except Exception as e:
        print(f"[WARN] Could not fetch/analyze transcript for {ticker} via API-Ninjas: {e}")
        return 0.0, {key: 0 for key in config.EVENT_KEYWORDS.keys()}, ""

def fetch_india_transcript_and_analyze(ticker: str):
    """
    Attempts to fetch and analyze earnings call transcripts for Indian companies (e.g., .NS tickers)
    from AlphaStreet using a simplified web scraping approach.
    If scraping fails, it will use a mock transcript.
    """
    print(f"Attempting to fetch Indian transcript for {ticker} from AlphaStreet via scraping...")
    base_ticker = ticker.split('.')[0] 

    transcript_text = ""
    try:
        alphastreet_url = f"https://alphastreet.com/stocks/{base_ticker}/earnings-call-transcripts"
        
        time.sleep(2) # Be polite and avoid getting blocked
        
        response = requests.get(alphastreet_url, timeout=15)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- SIMPLIFIED SCRAPING ATTEMPT ---
        # Try to find a common article body or main content area
        # Look for divs/sections that typically hold main text content
        content_div = soup.find('div', class_=lambda c: c and ('content' in c or 'body' in c or 'article' in c))
        
        if content_div:
            # Get all paragraphs within that content div
            paragraphs = content_div.find_all('p')
            transcript_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Basic check if the extracted text seems like a transcript (e.g., is it long enough?)
            if len(transcript_text) > 200: # Arbitrary length check
                print(f"[INFO] Successfully (simplified) scraped transcript for {ticker} from AlphaStreet.")
            else:
                print(f"[INFO] Scraped text for {ticker} was too short or generic. Using mock transcript.")
                transcript_text = "" # Fallback to mock if text is too short
        else:
            print(f"[INFO] Could not find a general content div for {ticker}. Using mock transcript.")
            transcript_text = "" # Fallback to mock if no content div found

    except requests.exceptions.RequestException as e:
        print(f"[WARN] Could not fetch/scrape transcript from AlphaStreet for {ticker}: {e}. Using mock transcript.")
        transcript_text = ""
    except Exception as e:
        print(f"[WARN] An unexpected error occurred during AlphaStreet scraping for {ticker}: {e}. Using mock transcript.")
        transcript_text = ""

    # --- FALLBACK TO MOCK TRANSCRIPT IF SCRAPING FAILED OR NO CONTENT ---
    if not transcript_text:
        transcript_text = f"This is a mock transcript for {ticker}. Company discussed general market conditions, financial outlook, and strategic initiatives. This is a placeholder as actual scraping failed or yielded no content."
        print(f"[INFO] Using MOCK transcript for {ticker}.")

    analyzer = SentimentIntensityAnalyzer()
    sentiment, events, snippet = analyze_text_for_events(transcript_text, analyzer)
    
    # Print sentiment only if it's not from an empty string (which would be 0.0)
    if transcript_text and "mock" not in transcript_text.lower(): # Don't print for mock
        print(f"Transcript analysis for {ticker} complete. Sentiment: {sentiment:.2f}")
        if snippet:
            print(f"  -> Detected Event: {snippet[:80]}...")
    elif "mock" in transcript_text.lower():
        print(f"Transcript analysis for {ticker} (MOCK) complete. Sentiment: {sentiment:.2f}")
    else:
        print(f"No transcript text available for {ticker} to analyze.")
        
    return sentiment, events, snippet


# --- Main pipeline ---
def process_all_tickers():
    all_features = []
    macro_data = fetch_macro_data()
    for ticker in config.TICKERS:
        print(f"\n--- Processing {ticker} ---")
        hist = yf.Ticker(ticker).history(period=config.HISTORY_PERIOD, interval=config.HISTORY_INTERVAL)
        if hist.empty:
            print(f"[WARN] No historical data found for {ticker}. Skipping.")
            continue
        df = add_technicals(hist)
        fundamentals = fetch_fundamentals(ticker)
        for key in FUNDAMENTAL_KEYS:
            df[key] = fundamentals.get(key, np.nan)
        df["NewsSentiment"] = fetch_news_sentiment(ticker)
        
        # --- CONDITIONAL CALL FOR TRANSCRIPTS ---
        if ticker.endswith(".NS"):
            transcript_sentiment, event_flags, _ = fetch_india_transcript_and_analyze(ticker)
        else:
            transcript_sentiment, event_flags, _ = fetch_transcript_and_analyze(ticker) # Uses API-Ninjas
        
        df["TranscriptSentiment"] = transcript_sentiment
        for event, flag in event_flags.items():
            df[event] = flag
        df["Ticker"] = ticker
        all_features.append(df)
        time.sleep(1) # Be mindful of API/scraping rate limits

    if not all_features:
        print("No data fetched for any ticker.")
        return

    combined_df = pd.concat(all_features)
    combined_df.reset_index(inplace=True)

    # --- FIX: Date conversion safe ---
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], utc=True)
    combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')

    if not macro_data.empty:
        macro_data.reset_index(inplace=True)
        # FIX: handle DATE/Date naming
        if "DATE" in macro_data.columns:
            macro_data.rename(columns={"DATE": "Date"}, inplace=True)
        elif "date" in macro_data.columns:
            macro_data.rename(columns={"date": "Date"}, inplace=True)

        macro_data['Date'] = pd.to_datetime(macro_data['Date'], utc=True)
        macro_data['Date'] = macro_data['Date'].dt.strftime('%Y-%m-%d')
        combined_df = pd.merge(combined_df, macro_data, on="Date", how="left")

    ffill_cols = FUNDAMENTAL_KEYS + ["NewsSentiment", "TranscriptSentiment", "MacroIndicator"] + list(config.EVENT_KEYWORDS.keys())
    combined_df[ffill_cols] = combined_df.groupby('Ticker')[ffill_cols].ffill()

    final_df = combined_df.dropna(subset=['Close', 'Return', 'RSI']).copy()
    final_df.set_index(['Date', 'Ticker'], inplace=True)

    try:
        conn = sqlite3.connect(config.DB_NAME)
        print(f"\nAttempting to read existing data from 'historical_features'...")
        try:
            # Read existing data, ensuring Date is parsed correctly and then formatted for consistent merging
            existing_df = pd.read_sql_query("SELECT * FROM historical_features", conn, index_col=['Date', 'Ticker'], parse_dates=['Date'])
            print(f"Successfully read {len(existing_df)} existing rows.")
            
            # Convert existing_df index 'Date' level to string for consistent merging with new data
            # This handles cases where dates might be stored as strings in DB or have different timezones
            existing_df.index = existing_df.index.set_levels(
                existing_df.index.levels[0].strftime('%Y-%m-%d'), level='Date'
            )

            # Combine old and new data
            # Use reset_index() to make 'Date' and 'Ticker' regular columns for concat and drop_duplicates
            combined_and_new_df = pd.concat([existing_df.reset_index(), final_df.reset_index()], ignore_index=True)
            
            # Drop duplicates based on 'Date' and 'Ticker', keeping the LAST (newest) entry
            # This effectively updates existing entries with new data and adds new ones
            deduplicated_df = combined_and_new_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
            
            # Set index back for saving
            deduplicated_df.set_index(['Date', 'Ticker'], inplace=True)
            
            print(f"Combined and deduplicated data has {len(deduplicated_df)} rows.")
            df_to_save = deduplicated_df
        except pd.io.sql.DatabaseError:
            print("Table 'historical_features' does not exist or is empty. Saving new data directly.")
            df_to_save = final_df
        
        print(f"\nSaving {len(df_to_save)} rows to database (replacing table with combined data)...")
        df_to_save.to_sql('historical_features', conn, if_exists='replace', index=True)
        print("Data ingestion complete.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    process_all_tickers()
