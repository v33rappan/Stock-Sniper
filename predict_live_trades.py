import pandas as pd
import glob
import os
import joblib
from datetime import datetime

MODEL_FILE = 'Output/logistic_model.pkl'
OUTPUT_FILE = 'Output/live_trade_predictions.csv'
SIGNALS_GLOB = 'Output/detected_signals_*.csv'

FEATURES = ['price_change', 'volume_spike', 'RSI', 'MACD_cross', 'score']

def get_latest_signal_file():
    files = glob.glob(SIGNALS_GLOB)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def predict_live_trades():
    latest_file = get_latest_signal_file()
    if not latest_file:
        print("No detected signals CSV found")
        return

    if not os.path.exists(MODEL_FILE):
        print(f"Trained model not found at {MODEL_FILE}")
        return

    print(f"Using signal file: {latest_file}")
    df = pd.read_csv(latest_file)

    # Preprocess
    df = df.dropna(subset=FEATURES)
    df['MACD_cross'] = df['MACD_cross'].astype(int)

    model = joblib.load(MODEL_FILE)
    X = df[FEATURES]
    probs = model.predict_proba(X)[:, 1] # Probability of WIN

    df['predicted_win_prob'] = probs
    df.sort_values(by='predicted_win_prob', ascending=False, inplace=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")
    print(df[['symbol', 'predicted_win_prob']].head())

if __name__ == '__main__':
    predict_live_trades()
