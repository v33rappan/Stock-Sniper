import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

TRADES_FILE = "Output/trade.csv"
INITIAL_CAPITAL = 100000
MAX_RISK_PER_TRADE = 0.02 # 2% of the capital
MAX_DRAWDOWN_ALERT = 0.2
SLIPPAGE_PERCENT_RANGE = (0, 0.005) # 0% to 0.5%$ slippage
RISK_ALERT_FILE = "Output/risk_alert.log"

def log_risk_alert(message):
    with open(RISK_ALERT_FILE, 'a') as f:
        f.write(message + "\n")
    print(message)

def compute_performance_metrics(equity_df):
    returns = equity_df['capital'].pct_change().dropna()
    cumulative_return = (equity_df['capital'].iloc[-1] / equity_df['capital'].iloc[0]) - 1
    avg_return = returns.mean()
    volatility = returns.std()
    sharpe_ratio = (avg_return / volatility ) * np.sqrt(252) if volatility != 0 else np.nan
    max_drawdown = equity_df['drawdown'].max()
    metrics = {
            'Cumulative Return': f"{cumulative_return*100:.2f}%",
            'Average Daily Return': f"{avg_return*100:.4f}%",
            'Volatility': f"{volatility*100:.4f}%",
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown': f"{max_drawdown*100:.2f}%"
    }

    print("\n Performance Metrics: ")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    pd.DataFrame([metrics]).to_csv("Output/performance_metrics.csv", index=False)
    print(" Performance metrics saved to Output/performance_metrics.csv")


def simulate_portfolio():
    if not os.path.exists(TRADES_FILE):
        print("{TRADES_FILE} not found")
        return

    df = pd.read_csv(TRADES_FILE)
    df = df.dropna(subset=['return_%', 'predicted_win_prob'])

    # Filter trades above a reasonable confidence threshold
    df = df[df['predicted_win_prob'] >= 0.6]
    if df.empty:
        print("No trades meet the minimum confidence threshold")
        return

    df = df.sort_values(by='predicted_win_prob', ascending=False)

    capital = INITIAL_CAPITAL
    peak_capital = capital
    equity_curve = []
    execution_plan = []

    for _, row in df.iterrows():
        # Dynamic risk scaling based on confidence
        confidence_scale = min(1.5, max(0.5, row['predicted_win_prob']))
        risk_amount = capital * MAX_RISK_PER_TRADE * confidence_scale
        trade_size = risk_amount / row['entry_price'] if row['entry_price'] else 0
        
        # Simulate slippage
        slippage_buy = 1 + random.uniform(*SLIPPAGE_PERCENT_RANGE)
        buy_price = row['entry_price'] * slippage_buy

        pnl = trade_size * buy_price * (row['return_%'] / 100)
        capital += pnl
        peak_capital = max(peak_capital, capital)
        drawdown = (peak_capital - capital) / peak_capital

        if drawdown >= MAX_DRAWDOWN_ALERT:
            alert_msg = f" ALERT: Drawdown {drawdown*100:.2f}% exceeded threshold at {row['timestamp']}"
            log_risk_alert(alert_msg)

        equity_curve.append({
            'timestamp': row['timestamp'],
            'capital': capital,
            'drawdown': drawdown,
            'symbol': row['symbol'],
            'strategy': row['strategy'],
            'return_%': row['return_%'],
            'position_size': trade_size,
            'predicted_win_prob': row['predicted_win_prob']
        })

        execution_plan.append({
            'timestamp': row['timestamp'],
            'symbol': row['symbol'],
            'action': 'BUY',
            'quantity': round(trade_size, 2),
            'price': round(buy_price, 2),
            'strategy': row['strategy'],
            'predicted_win_prob': row['predicted_win_prob']
        })

        if row['exit_price'] and trade_size > 0:
            slippage_sell = 1 - random.uniform(*SLIPPAGE_PERCENT_RANGE)
            sell_price = row['exit_price'] * slippage_sell

            execution_plan.append({
                'timestamp': row['timestamp'],
                'symbol': row['symbol'],
                'action': 'SELL',
                'quantity': round(trade_size, 2),
                'price': round(sell_price, 2),
                'strategy': row['strategy'],
                'predicted_win_prob': row['predicted_win_prob']
        })

    equity_df = pd.DataFrame(equity_curve)
    equity_df.to_csv("Output/portfolio_simulation.csv", index=False)
    print("Portfolio simulation saved to Output/portfolio_simulation.csv")
    
    exec_df = pd.DataFrame(execution_plan)
    exec_df.to_csv("Output/execution_plan.csv", index=False)
    print(" Execution plan saved to Output/execution_plan.csv")

    compute_performance_metrics(equity_df)

    plt.figure(figsize=(10, 5))
    plt.plot(equity_df['timestamp'], equity_df['capital'], marker='o')
    plt.xticks(rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Capital')
    plt.title('Portfoilio Equity Curve')
    plt.tight_layout()
    plt.savefig("Output/portfolio_equity_curve.png")
    print(" Equity curve plot saved to Output/portfolio_equity_curve.png")

if __name__ == '__main__':
    simulate_portfolio()
