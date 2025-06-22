import argparse
import os
import pandas as pd
from sniper_stock_detector import main as run_detection
from symbol_utils import check_smallcap_250_csv, update_sme_watchlist_csv
from evaluate_ml_model import evaluate_model
from evaluate_feature_correlation import run as run_feature_corr
from evaluate_performance import run as run_performance
from evaluate_score_buckets import run as run_score_buckets
from evaluate_strategies import run as run_strategy_comparison
from predict_live_trades import run as run_predictor
from threshold_optimiser import run_optimisation

CONFIG = {
        'TRADE_FILE': 'Output/trade.csv',
        'TOP_N': 10,
        'MODEL': 'logistic'
}

def update_list():
    print("Updating SME and Smallcap lists...")
    check_smallcap_250_csv()
    update_sme_watchlist_csv()

def summarise_top_trades(top_n):
    print("\n Final Suggestions: \n")
    trades_file = CONFIG['TRADE_FILE']
    if not os.path.exists(trades_file):
        print(f"No trades file found at {trades_file}")

    df = pd.read_csv(trades_file, on_bad_lines='skip')
    df = df.dropna(subset=['predicted_win_prob'])
    df = df.sort_values(by='predicted_win_prob', ascending=False).head(top_n)

    if df.empty:
        print("No high confidence trades found")
        return

    for _, row in df.iterrows():
        print(f" {row['symbol']} | {row['category']} | Rs{row['latest_close']} | "f"+{row['price_change']}% | Vol: {row['volume_spike']}x | "f"Prob: {round(row['predicted_win_prob'] * 100, 2)}%")

def run_analysis():
    print(" Running post-detection analysis...")
    run_feature_corr()
    run_score_buckets()
    run_performance()
    run_strategy_comparision()

def run_all(args):
    update_list()

    if args.retrain:
        print(f"Retraining ML model: {CONFIG['MODEL']}")
        evaluate_model(CONFIG['MODEL'])

    run_detection()
    run_analysis()
    summarise_top_trades(args.top)

def main():
    parser = argparse.ArgumentParser(description='Stock Detection Pipeline Runner')
    parser.add_argument("--run", action='store_true', help='Run full pipeline')
    parser.add_argument("--retain-thresholds", action='store_true', help='Retrain strategy thresholds before detection')
    parser.add_argument("--analyse", action='store_true', help='Run evaluation/analysis scripts')
    parser.add_argument("--predict-only", action='store_true', help='Only run live trade predictor')
    parser.add_argument("--retrain", action='store_true', help='Retrain ML model')
    parser.add_argument("--top", type=int, default=CONFIG['TOP_N'], help='Number of top trades to show')

    args = parser.parse_args()

    if args.run:
        run_all(args)
    elif args.retain_thresholds:
        run_optimisation()
    elif args.analyse:
        run_analysis()
    elif args.predict_only:
        run_predictor()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
