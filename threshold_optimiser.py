import pandas as pd
import numpy as np
import json
import os
from itertools import product
from optimisation_utils import convert_numpy

TRADES_FILE = "Output/trade.csv"
OUTPUT_FILE = "Output/optimised_strategies.json"

def load_trades():
    if not os.path.exists(TRADES_FILE):
        print("trade.csv not found. Run backtest first")
        return pd.DataFrame()
    return pd.read_csv(TRADES_FILE)

def optimised_thresholds(strategy_name, price_range, volume_range, metric="win_rate"):
    df = load_trades()
    if df.empty:
        return {}

    df = df[df['strategy'] == strategy_name]
    df = df.dropna(subset=['return_%'])

    best_score = -np.inf
    best_params = {}

    for price_th, vol_th in product(price_range, volume_range):
        filtered = df[(df['price_change'] >= price_th) & (df['volume_spike'] >= vol_th)]
        if len(filtered) < 10:
            continue

        wins = filtered[filtered['return_%'] > 0]
        win_rate = len(wins) / len(filtered)

        if metric == 'win_rate':
            score = win_rate
        elif metric == 'avg_return':
            score = filtered['return_%'].mean()
        else:
            continue

        if score > best_score:
            best_score = score
            best_params = {
                    "PRICE_THRESHOLD_PERCENT": price_th,
                    "VOLUME_SPIKE_MULTIPLIER": vol_th,
                    "samples": len(filtered),
                    "score": round(score, 4),
                    "metric": metric
            }
    
    return best_params

def run_optimisation():
    strategies = ["aggressive"] # TODO: Extend to other strategies
    price_range = np.arange(2, 20, 1) # 2% to 20%
    volume_range = np.arange(1, 10, 0.5) # 1x to 10x

    results = {}

    for strategy in strategies:
        print(f"Optimising thresholds for {strategy} strategy...")
        best = optimised_thresholds(strategy, price_range, volume_range)
        if best:
            results[strategy] = best
            print(f"Best for {strategy}: {best}")
        else:
            print(f"No sufficient data to optimise {strategy}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=convert_numpy)
    print(f" Saved optimised strategies to {OUTPUT_FILE}")

if __name__ == '__main__':
    run_optimisation()
