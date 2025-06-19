import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import warnings
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ---------------- CONFIG ------------------
CONFIG = {
        'SMALLCAP_CSV': 'nifty_smallcap_250.csv',
        'SME_CSV': 'sme_watchlist.csv',
        'CACHE_DIR': 'cache',
        'PRICE_CHANGE_LOOKBACK_DAYS': 3,
        'PRICE_THRESHOLD_PERCENT': 1,
        'VOLUME_LOOKBACK_DAYS': 30,
        'VOLUME_SPIKE_MULTIPLIER': 1.5,
        'CACHE_EXPIRY_MINUTES': 120,
        'DEBUG_MODE': True
}

# Suppress warnings from yfinance or pandas
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
        sme_symbols = []
        df = pd.DataFrame({'symbol': sme_symbols})
        df.to_csv(CONFIG['SME_CSV'], index=False)
        print("✅ SME watchlist updated.")
    except Exception as e:
        print(f"Failed to update SME watchlist: {e}")

# ------------- PRICE/VOLUME CHECK ---------
def fetch_bulk_data(symbols):
    cache_file = os.path.join(CONFIG['CACHE_DIR'], 'bulk_data.json')
    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mod_time < timedelta(minutes=CONFIG['CACHE_EXPIRY_MINUTES']):
            try:
                cached_data = pd.read_json(cache_file, typ='series', convert_dates=True)
                return { ticker: pd.DataFrame(data) for ticker, data in cached_data.items() }
            except Exception as e:
                print(f"Failed to read cached bulk data: {e}")
    try:
        data = yf.download(symbols, period="40d", group_by="ticker", progress=True)
        if not isinstance(data, pd.DataFrame) or data.empty:
            print(f"Error: No data returned from yfinance")
            return {}

        data_dict = {}
        if isinstance(data.columns, pd.MultiIndex):
            for symbol in symbols:
                if symbol in data.columns.get_level_values(0):
                    data_dict[symbol] = data[symbol].copy()
        else:
            for symbol in symbols:
                data_dict[symbol] = data.copy()
        
        # Save to cache
        try:
            save_data = { k: v.to_dict() for k, v in data_dict.items() }
            pd.Series(save_data).to_json(cache_file)
        except Exception as e:
            print(f"Failed to cache bulk data: {e}")

        return data_dict
    except Exception as e:
        print(f"Error fetching bulk data: {e}")
        return {}

# ---------- FILTER AND ALERT -------------
def detect_opportunities(symbols, category):
    results = []
    data = fetch_bulk_data(symbols)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"detected_signals_{category}_{now_str}.csv"
    rows = []

    for symbol in symbols:
        if symbol not in data:
            continue
        df = data[symbol]
        if df.empty or len(df) < CONFIG['PRICE_CHANGE_LOOKBACK_DAYS']:
            continue

        df = df.dropna()
        recent = df.tail(CONFIG['PRICE_CHANGE_LOOKBACK_DAYS'])
        old_data = df.iloc[:-CONFIG['PRICE_CHANGE_LOOKBACK_DAYS']]
        if old_data.empty:
            continue

        old_avg_vol = old_data['Volume'].tail(CONFIG['VOLUME_LOOKBACK_DAYS']).mean()
        if old_avg_vol == 0 or pd.isna(old_avg_vol):
            continue

        price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0] * 100
        volume_spike = recent['Volume'].mean() / old_avg_vol

        # Technical indicator
        try:
            rsi = RSIIndicator(close=df['Close']).rsi().iloc[-1]
            macd = MACD(close=df['Close'])
            macd_cross = macd.macd_diff().iloc[-1] > 0
        except:
            rsi = None
            macd_cross = None

        if CONFIG['DEBUG_MODE']:
            print(f"{symbol}: {price_change:.2f}% | Vol: {volume_spike:.2f}x | RSI: {rsi} | MACD Cross: {macd_cross}")

        if price_change < 0:
            continue

        if price_change >= CONFIG['PRICE_THRESHOLD_PERCENT'] and volume_spike >= CONFIG['VOLUME_SPIKE_MULTIPLIER']:
            result = {
                    'symbol': symbol,
                    'category': category,
                    'latest_close': round(df['Close'].iloc[-1], 2),
                    'price_change': round(price_change, 2),
                    'volume_spike': round(volume_spike, 2),
                    'RSI': round(rsi, 2) if rsi else None,
                    'MACD_cross': macd_cross,
                    'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            rows.append(result)

    if rows:
        pd.DataFrame(rows).to_csv(csv_filename, index=False)
        print(f"Saved {len(rows)} signals to {csv_filename}")

    return results

# ------------ MAIN ---------------------
def main():

    check_smallcap_250_csv()
    update_sme_watchlist_csv()

    smallcaps, smes = load_stock_lists()
    all_opportunities =  []
    all_opportunities += detect_opportunities(smallcaps, 'Smallcap')
    all_opportunities += detect_opportunities(smes, 'SME/Microcap')
    
    if all_opportunities:
        print("\n Detected Trade Opportunities: \n")
        for op in all_opportunities:
            print(f"\U0001F4C8 {op['symbol']} | {op['category']} | Price: Rs{op['latest_close']} | +{op['price_change']}% | Vol: {op['volume_spike']}x")
    else:
        print("No trade-worthy opportunity found today")

if __name__ == "__main__":
    main()

