import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = 'Output'
TRADES_FILE = 'Output/trade.csv'
SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'performance_summary.csv')
EQUITY_PLOT = os.path.join(OUTPUT_DIR, 'equity_curve.png')
INITIAL_CAPITAL = 100000

def load_trades(file):
    if not os.path.exists(file):
        print(f"Trade file not found: {file}")
        return None
    df = pd.read_csv(file)
    df = df.dropna(subset=['return_%'])
    df = df.sort_values(by='timestamp')
    df['result'] = df['return_%'].apply(lambda x: 'win' if x > 0 else 'loss')
    return df

def complete_summary(df):
    total_trades = len(df)
    wins = df[df['return_%'] > 0]
    losses = df[df['return_%'] < 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    avg_return = df['return_%'].mean()
    avg_r = df['r_multiple'].mean()

    profit_factor = wins['return_%'].sum() / abs(losses['return_%'].sum()) if not losses.empty else float('inf')

    targets = df[df['exit_reason'] == 'TARGET']
    stops = df[df['exit_reason'] == 'STOP']

    summary = {
            'Total Trades': total_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Avg Return (%)': round(avg_return, 2),
            'Avg R-Multiple': round(avg_r, 2),
            'Profit Factor': round(profit_factor, 2),
            'Target Hit %': round(len(targets)/total_trades*100, 2),
            'Stop Hit %': round(len(stops)/total_trades*100, 2),
    }

    return summary

def simulate_equity(df, start_capital=INITIAL_CAPITAL):
    capital = [start_capital]
    for pct in df['return_%']:
        next_capital = capital[-1] * (1 + pct / 100)
        capital.append(next_capital)
    df['equity'] = capital[1:]
    max_drawdown = max([max(capital[:i+1]) - capital[i] for i in range(1, len(capital))], default=0)
    return df, capital[1:], max_drawdown

def plot_equity(equity, output=EQUITY_PLOT):
    plt.figure(figsize=(10, 5))
    plt.plot(equity, label='Equity Curve')
    plt.title('Equity Curve')
    plt.xlabel('Trades')
    plt.ylabel('Capital (Rs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Equity curce saved to {output}")

def run():
    df = load_trades(TRADES_FILE)
    if df is None or df.empty:
        return

    summary = complete_summary(df)
    df, equity, drawdown = simulate_equity(df)

    summary['Max Drawdown (Rs)'] = round(drawdown, 2)
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)

    print(f"\n Summary saved to {SUMMARY_FILE}")
    for k, v in summary.items():
        print(f"{k}: {v}")

    plot_equity(equity)

if __name__ == '__main__':
    run()
