import pandas as pd
import os
import argparse

OUTPUT_DIR = 'Output'
TRADES_FILE = os.path.join(OUTPUT_DIR, 'trade.csv')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'strategy_performance_summary.csv')

def list_available_columns(df):
    print("\n📊 Available columns for grouping:")
    for col in df.columns:
        print(f" - {col}")
    print()

def summarize(df, group_col):
    summaries = []

    if group_col not in df.columns:
        print(f"❌ Column '{group_col}' not found in trades file.")
        return

    groups = df[group_col].unique()
    for value in groups:
        group_df = df[df[group_col] == value]
        total = len(group_df)
        wins = group_df[group_df['return_%'] > 0]
        losses = group_df[group_df['return_%'] < 0]
        win_rate = len(wins) / total * 100 if total else 0
        avg_return = group_df['return_%'].mean()
        avg_r = group_df['r_multiple'].mean()
        profit_factor = wins['return_%'].sum() / abs(losses['return_%'].sum()) if not losses.empty else float('inf')

        summaries.append({
            group_col: value,
            'Total Trades': total,
            'Win Rate (%)': round(win_rate, 2),
            'Avg Return (%)': round(avg_return, 2),
            'Avg R-multiple': round(avg_r, 2),
            'Profit Factor': round(profit_factor, 2),
        })

    df_out = pd.DataFrame(summaries)
    df_out.sort_values(by='Profit Factor', ascending=False, inplace=True)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Summary grouped by '{group_col}' saved to {OUTPUT_FILE}\n")
    print(df_out.to_string(index=False))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='strategy', help="Column to group by (e.g. strategy, symbol)")
    parser.add_argument('--list-metrics', action='store_true', help="List all columns available for grouping")
    args = parser.parse_args()

    if not os.path.exists(TRADES_FILE):
        print(f"❌ Trade file not found: {TRADES_FILE}")
        return

    df = pd.read_csv(TRADES_FILE)
    df = df.dropna(subset=['return_%'])

    if args.list_metrics:
        list_available_columns(df)
        return

    summarize(df, args.group)

if __name__ == "__main__":
    run()

