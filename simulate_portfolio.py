import os
import pandas as pd

TRADES_FILE = "Output/trade.csv"
INITIAL_CAPITAL = 100000
MAX_RISK_PER_TRADE = 0.02 # 2% of the capital

def simulate_portfolio():
    if not os.path.exists(TRADES_FILE):
        print("{TRADES_FILE} not found")
        return

    df = pd.read_csv(TRADES_FILE)
    df = df.dropna(subset=['return_%', 'predicted_win_prob'])
    df = df.sort_values(by='predicted_win_prob', ascending=False)

    capital = INITIAL_CAPITAL
    equity_curve = []

    for _, row in df.iterrows():
        risk_amount = capital * MAX_RISK_PER_TRADE
        trade_size = risk_amount # for simplicity 1:1 risk-reward assumed
        pnl = trade_size * (row['return_%'] / 100)
        capital += pnl
        equity_curve.append(capital)

    summary = pd.DataFrame({
        'Trade': range(1, len(equity_curve) + 1),
        'Equity': equity_curve
    })

    summary.to_csv("Output/portfolio_simulation.csv", index=False)
    print("Portfolio simulation saved to Output/portfolio_simulation.csv")

if __name__ == '__main__':
    simulate_portfolio()
