import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

TRADES_FILE = 'Output/trade.csv'
OUTPUT_CSV = 'Output/feature_correlation_matrix.csv'
OUTPUT_IMG = 'Output/feature_correlation_matrix.png'

def evaluate_feature_correlation():
    if not os.path.exists(TRADES_FILE):
        print(f"Missing trade file: {TRADES_FILE}")
        return

    df = pd.read_csv(TRADES_FILE)

    df = df.dropna(subset=[
        'price_change', 'volume_spike', 'RSI', 'MACD_cross',
        'score', 'return_%', 'r_multiple', 'result'
    ])

    # Convert MACD boolean to integer
    df['MACD_cross'] = df['MACD_cross'].astype(int)

    # Convert result to 1 (WIN), 0, (LOSS/NEUTRAL)
    df['result_bin'] = df['result'].apply(lambda r: 1 if r == 'WIN' else 0)

    features = ['price_change', 'volume_spike', 'RSI', 'MACD_cross', 'score']
    targets = ['return_%', 'r_multiple', 'result_bin']

    corr_df = df[features + targets].corr().round(3)

    corr_df.to_csv(OUTPUT_CSV)
    print(f"Correlation matrix saved to {OUTPUT_CSV}")

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Heatmap saved to {OUTPUT_IMG}")

def run():
    evaluate_feature_correlation()

if __name__ == '__main__':
    evaluate_feature_correlation()
