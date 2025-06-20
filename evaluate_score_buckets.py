import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse

TRADES_FILE = 'Output/trade.csv'
OUTPUT_CSV = 'Output/score_bucket_summary.csv'
OUTPUT_PLOT = 'Output/score_vs_winrate.png'

DEFAULT_METRICS = ['Win Rate (%)', 'Avg Return (%)', 'Avg R-multiple']

def get_score_bucket(score, bucket_size):
    return f"{(score // bucket_size) * bucket_size:.0f}-{((score // bucket_size) + 1) * bucket_size:.0f}"

def evaluate_score_buckets(bucket_size, metrics):
    if not os.path.exists(TRADES_FILE):
        print(f"Missing trade file: {TRADES_FILE}")
        return

    df = pd.read_csv(TRADES_FILE)
    df = df.dropna(subset=['score', 'return_%'])

    df['score_bucket'] = df['score'].apply(lambda s: get_score_bucket(s, bucket_size))
    grouped = df.groupby('score_bucket')

    summary = []
    for bucket, group in grouped:
        wins = group[group['return_%'] > 0]
        total = len(group)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        avg_return = group['return_%'].mean()
        avg_r = group['r_multiple'].mean()

        summary.append({
            'score_bucket': bucket,
            'Total trades': total,
            'Win Rate (%)': round(win_rate, 2),
            'Avg Return (%)': round(avg_return, 2),
            'Avg R-Multiple': round(avg_r, 2)
        })

    summary_df = pd.DataFrame(summary)
    summary_df.sort_values(by='score_bucket', inplace=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Score bucket summary saved to {OUTPUT_CSV}")
    print(summary_df.to_string(index=False))
    
    # Validate metric names
    available_metrics = summary_df.columns.tolist()
    valid_metrics = [m for m in metrics if m in available_metrics]

    if not valid_metrics:
        print(f"Invalid metric '{metric}'. Available: {list(summary_df.columns)}")
        return

    # Plotting
    fig, axs = plt.subplots(len(valid_metrics), 1, figsize=(10, 4 * len(valid_metrics)))
    if len(valid_metrics) == 1:
        axs = [axs]
    
    for ax, metric in zip(axs, valid_metrics):
        x = range(len(summary_df['score_bucket']))
        ax.bar(x, summary_df[metric], width=0.6)
        ax.set_title(f"{metric} by Score Bucket")
        ax.set_xlabel("Score Bucket")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['score_bucket'], rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate score buckets from trades.")
    parser.add_argument('--bucket-size', type=int, default=20, help='Bucket size for score grouping ( default: 20 )')
    parser.add_argument('--metrics', nargs='*', default=DEFAULT_METRICS, help='List of metrics to plot')

    args = parser.parse_args()
    evaluate_score_buckets(args.bucket_size, args.metrics)
