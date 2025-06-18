import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
import io
import os
from bs4 import BeautifulSoup
import json
import warnings

# ---------------- CONFIG ------------------
CONFIG = {
        'SMALLCAP_CSV': 'nifty_smallcap_250.csv',
        'SME_CSV': 'sme_watchlist.csv',
        'CACHE_DIR': 'cache',
        'PRICE_CHANGE_LOOKBACK_DAYS': 3,
        'PRICE_THRESHOLD_PERCENT': 12,
        'VOLUME_LOOKBACK_DAYS': 30,
        'VOLUME_SPIKE_MULTIPLIER': 2.0,
        'CACHE_EXPIRY_MINUTES': 120,
}

warnings.filterwarnings("ignore")
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

# ------------- LOAD SYMBOLS ---------------
def load_stock_lists():
    try:
        df_smallcap = pd.read_csv(CONFIG['SMALLCAP_CSV'])
        if 'symbol' not in df_smallcap.columns and 'Symbol' in df_smallcap.columns:
            df_smallcap['symbol'] = df_smallcap['Symbol'].astype(str) + ".NS"
        smallcaps = df_smallcap['symbol'].dropna().unique().tolist()
    except Exception as e:
        print(f"Error loading SmallCap CSV: {e}")
        smallcaps = []

    try:
        df_sme = pd.read_csv(CONFIG['SME_CSV'])
        smes = df_sme['symbol'].dropna().unique().tolist()
    except Exception as e:
        print(f"Error loading SME CSV: {e}")
        smes = []

    print(f"Loaded {len(smallcaps)} Smallcap symbol and {len(smes)} SME symbols")
    return smallcaps, smes

# -------- CSV AUTO-FILL MODULE ------------
def check_smallcap_250_csv():
    try:
        if os.path.exists(CONFIG['SMALLCAP_CSV']):
            print("Found the file...")
            return
        else:
            raise Exception("Please download the file manually....")
    except Exception as e:
        print("Failed to read the file..." + {e})

def update_sme_watchlist_csv():
    try:
        # Simple example list of popular SME tickers - you can replace this
        sme_symbols = ["VISHNU.NS", "JFL.NS", "AGARWALFT.NS"]
        df = pd.DataFrame({'symbol': sme_symbols})
        df.to_csv(CONFIG['SME_CSV'], index=False)
        print("✅ SME watchlist updated.")
    except Exception as e:
        print(f"Failed to update SME watchlist: {e}")

# ------------- PRICE/VOLUME CHECK ---------
def fetch_bulk_data(symbols):
    try:
        data = yf.download(tickers=symbols, period="40d", group_by="ticker", progress=True)
        return data
    except Exception as e:
        print(f"Error fetching bulk data: {e}")
        return {}

def fetch_price_volume_data(symbol, data):
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}.json")
    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mod_time < timedelta(minutes=CONFIG['CACHE_EXPIRY_MINUTES']):
            with open(cache_file, 'r') as f:
                return json.load(f)

    try:
        symbol_data = data[symbol] if isinstance(data, dict) else data
        if symbol_data.empty or len(symbol_data) < CONFIG['PRICE_CHANGE_LOOKBACK_DAYS']:
            print(f"No price data for {symbol} (possibly delisted or invalid)")
            return None

        recent = symbol_data.tail(CONFIG['PRICE_CHANGE_LOOKBACK_DAYS'])
        old_data = symbol_data.iloc[:-CONFIG['PRICE_CHANGE_LOOKBACK_DAYS']]

        if old_data.empty or 'Volume' not in old_data.columns:
            print(f"Not enough volume data for {symbol}")
            return None

        old_avg_vol_series = old_data['Volume'].tail(CONFIG['VOLUME_LOOKBACK_DAYS'])
        if old_avg_vol_series.empty:
            print(f"No historical volume for {symbol}")
            return None

        old_avg_vol = old_avg_vol_series.mean()
        if pd.isna(old_avg_vol) or old_avg_vol == 0:
            print(f"Invalid old volume for {symbol}")
            return None

        if recent['Close'].empty or len(recent['Close']) < 2:
            print(f"Not enough closing price data for {symbol}")
            return None

        price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
        volume_spike = recent['Volume'].mean() / old_avg_vol

        result = {
                'symbol': symbol,
                'price_change': round(price_change * 100, 2),
                'volume_spike': round(volume_spike, 2),
                'latest_close': round(recent['Close'].iloc[-1], 2)
        }

        with open(cache_file, 'w') as f:
            json.dump(result, f)
        return result
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# ---------- FILTER AND ALERT -------------
def detect_opportunities(symbols, category):
    results = []
    data = fetch_bulk_data(symbols)
    for sym in symbols:
        symbol_data = data[sym] if sym in data else data
        result = fetch_price_volume_data(sym, symbol_data)
        if result:
            if result['price_change'] < 0:
                print(f"{sym} rejected due to negative price change: {data['price_change']}%")
                continue
            if result['price_change'] >= CONFIG['PRICE_THRESHOLD_PERCENT'] and result['volume_spike'] >= CONFIG['VOLUME_SPIKE_MULTIPLIER']:
                result['category'] = category
                results.append(data)
    return results

# ------------ MAIN ---------------------
def main():

    print("Updating CSVs.....")
    check_smallcap_250_csv()
    update_sme_watchlist_csv()

    print("Detecting opportunities...")
    smallcaps, smes = load_stock_lists()
    all_opportunities =  []
    all_opportunities += detect_opportunities(smallcaps, 'Smallcap')
    all_opportunities += detect_opportunities(smes, 'SME/Microcap')

    print(f"Total opportunities: {len(all_opportunities)}")

    if all_opportunities:
        print("\n Detected Trade Opportunities: \n")
        for op in all_opportunities:
            print(f"\U0001F4C8 {op['symbol']} | {op['category']} | Price: Rs{op['latest_close']} | +{op['price_change']}% | Vol: {op['volume_spike']}x")
        else:
            print("No trade-worthy opportunity found today")

if __name__ == "__main__":
    main()

