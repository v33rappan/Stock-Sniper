import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import warnings
from ta.momentum import RSIIndicator
from ta.trend import MACD
from strategies import STRATEGIES
import joblib
from optimisation_utils import apply_optimised_threshold

# ---------------- CONFIG ------------------
CONFIG = {
        'SMALLCAP_CSV': 'data/nifty_smallcap_250.csv',
        'SME_CSV': 'data/sme_watchlist.csv',
        'CACHE_DIR': 'cache',
        'PRICE_CHANGE_LOOKBACK_DAYS': 3,
        'VOLUME_LOOKBACK_DAYS': 30,
        'CACHE_EXPIRY_MINUTES': 120,
        'DEBUG_MODE': True,
        'HOLD_DAYS': 10,
        'BACKTEST_RESULTS_FILE': 'backtest_results.csv',
        'TOP_N': 10,
        'STOP_LOSS_PCT': 5,
        'TAKE_PROFIT_PCT': 8,
        'TRADE_LOG_FILE': 'Output/trade.csv',
        'ENABLE_ML_FILTERING': True,
        'MIN_WIN_PROBABILITY': 0.6,
        'MODEL_PATH': 'Output/logistic_model.pkl'
}

# Suppress warnings from yfinance or pandas
warnings.filterwarnings("ignore")
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

OUTPUT_DIR = 'Output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
def detect_opportunities(symbols, category, strategy):
    results = []
    data = fetch_bulk_data(symbols)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(OUTPUT_DIR, f"detected_signals_{category}_{strategy}_{now_str}.csv")
    rows = []
    backtest_rows = []
    trade_log = []
    
    strategy_config = STRATEGIES[strategy]
    threshold_price = strategy_config['PRICE_THRESHOLD_PERCENT']
    threshold_volume = strategy_config['VOLUME_SPIKE_MULTIPLIER']
    weights = strategy_config['SCORE_WEIGHTS']

    # Load ML model if enabled
    model = None
    if CONFIG['ENABLE_ML_FILTERING'] and os.path.exists(CONFIG['MODEL_PATH']):
        try:
            model = joblib.load(CONFIG['MODEL_PATH'])
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            model = None

    for symbol in symbols:
        if symbol not in data:
            continue
        df = data[symbol]
        if df.empty or len(df) < CONFIG['PRICE_CHANGE_LOOKBACK_DAYS'] + CONFIG['HOLD_DAYS']:
            continue

        df = df.dropna()
        recent = df.tail(CONFIG['PRICE_CHANGE_LOOKBACK_DAYS'] + CONFIG['HOLD_DAYS'])
        signal_data = recent.head(CONFIG['PRICE_CHANGE_LOOKBACK_DAYS'])
        hold_data = recent.tail(CONFIG['HOLD_DAYS'])
        old_data = df.iloc[:-(CONFIG['PRICE_CHANGE_LOOKBACK_DAYS'] + CONFIG['HOLD_DAYS'])]

        if old_data.empty:
            continue

        old_avg_vol = old_data['Volume'].tail(CONFIG['VOLUME_LOOKBACK_DAYS']).mean()
        if old_avg_vol == 0 or pd.isna(old_avg_vol):
            continue

        price_change = (signal_data['Close'].iloc[-1] - signal_data['Close'].iloc[0]) / signal_data['Close'].iloc[0] * 100
        volume_spike = signal_data['Volume'].mean() / old_avg_vol

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

        if price_change >= threshold_price and volume_spike >= threshold_volume:
            entry_price = signal_data['Close'].iloc[-1]
            stop_loss_price = entry_price * ( 1 - CONFIG['STOP_LOSS_PCT'] / 100 )
            take_profit_price = entry_price * ( 1 + CONFIG['TAKE_PROFIT_PCT'] / 100 )

            exit_price = None
            exit_reason = "TIME"
            
            if CONFIG['DEBUG_MODE']:
                print(f"\n🎯 {symbol}: Entry=₹{entry_price:.2f} | SL=₹{stop_loss_price:.2f} | TP=₹{take_profit_price:.2f}")

            for i, (_, row) in enumerate(hold_data.iterrows()):
                high = row['High']
                low = row['Low']
                if CONFIG['DEBUG_MODE']:
                    print(f"Day {i}: High=₹{high:.2f}, Low=₹{low:.2f}")
                if low <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = "STOP"
                    if CONFIG['DEBUG_MODE']:
                        print(f"✅ STOP triggered on Day {i}")
                    break
                elif high >= take_profit_price:
                    exit_price = take_profit_price
                    exit_reason = "TARGET"
                    if CONFIG['DEBUG_MODE']:
                        print(f"✅ TARGET hit on Day {i}")
                    break

            if exit_price is None and not hold_data.empty:
                exit_price = hold_data['Close'].iloc[-1]

            pnl_percent = ((exit_price - entry_price) / entry_price * 100) if exit_price else None
            r_multiple = (pnl_percent / CONFIG['STOP_LOSS_PCT']) if pnl_percent is not None else None

            rsi_score = (100 - abs(50 - rsi)) if rsi else 0
            
            score = (
                    price_change * weights['price'] +
                    volume_spike * weights['volume'] +
                    (weights['macd'] if macd_cross else 0) +
                    rsi_score * weights['rsi']
            )

            result = {
                    'symbol': symbol,
                    'category': category,
                    'strategy': strategy,
                    'latest_close': round(df['Close'].iloc[-1], 2),
                    'price_change': round(price_change, 2),
                    'volume_spike': round(volume_spike, 2),
                    'RSI': round(rsi, 2) if rsi else None,
                    'MACD_cross': macd_cross,
                    'score': round(score, 2),
                    'timestamp': datetime.now().isoformat(),
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2) if exit_price else None,
                    'return_%': round(pnl_percent, 2) if pnl_percent is not None else None,
                    'r_multiple': round(r_multiple, 2) if r_multiple is not None else None,
                    'exit_reason': exit_reason,
                    'result': "WIN" if pnl_percent and pnl_percent > 0 else "LOSS" if pnl_percent and pnl_percent < 0 else "NEUTRAL"
            }

            # Predict WIN probability
            if model:
                try:
                    x_row = pd.DataFrame([{
                        'price_change': price_change,
                        'volume_spike': volume_spike,
                        'RSI': rsi,
                        'MACD_cross': int(macd_cross),
                        'score': score
                    }])
                    win_prob = model.predict_proba(x_row)[0][1]
                    result['predicted_win_prob'] = round(win_prob, 4)
                    if CONFIG['ENABLE_ML_FILTERING'] and win_prob < CONFIG['MIN_WIN_PROBABILITY']:
                        continue # Skip low confidence trades
                except Exception as e:
                    print(f"Prediction failed for {symbol}: {e}")
                    result['predicted_win_prob'] = None
            else:
                result['predicted_win_prob'] = None


            results.append(result)
            rows.append(result)
            backtest_rows.append(result)
            trade_log.append(result)

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(by='score', ascending=False, inplace=True)
        top_df = df.head(CONFIG['TOP_N'])
        top_df.to_csv(csv_filename, index=False)
        print(f"Saved top {len(top_df)} signals to {csv_filename}")

    if backtest_rows:
        bt_file = os.path.join(OUTPUT_DIR, CONFIG['BACKTEST_RESULTS_FILE'])
        pd.DataFrame(backtest_rows).to_csv(bt_file, mode='a', header=not os.path.exists(bt_file), index=False)
        print(f"Backtest results saved to {bt_file}")
    
    if trade_log:
        trade_file = CONFIG['TRADE_LOG_FILE']
        pd.DataFrame(trade_log).to_csv(trade_file, mode='a', header=not os.path.exists(trade_file), index=False)
        print(f"Trade log updated to {trade_file}")

    return results

# ------------ MAIN ---------------------
def main():
    
    from symbol_utils import get_smallcap_symbols, get_sme_symbols

    check_smallcap_250_csv()
    update_sme_watchlist_csv()

    smallcaps, smes = load_stock_lists()
    all_opportunities =  []
    
    # Apply threshold optimisation
    apply_optimised_threshold(STRATEGIES)

    for strategy in STRATEGIES:
        print(f"\n Running strategy: {strategy}\n")
        all_opportunities += detect_opportunities(smallcaps, 'Smallcap', strategy)
    all_opportunities += detect_opportunities(smes, 'SME/Microcap', strategy)
    
    if all_opportunities:
        print("\n Detected Trade Opportunities: \n")
        for op in all_opportunities:
            print(f"\U0001F4C8 {op['symbol']} | {op['category']} | Price: Rs{op['latest_close']} | +{op['price_change']}% | Vol: {op['volume_spike']}x | {op['score']} | {op['entry_price']} | {op['exit_price']} | {op['result']} | {op['exit_reason']} | {op['predicted_win_prob']}")
    else:
        print("No trade-worthy opportunity found today")

if __name__ == "__main__":
    main()

