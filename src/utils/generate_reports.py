"""
Automated Report Generator
Generates daily and weekly review reports from trade logs
"""

import pandas as pd
from datetime import datetime, timedelta
import os

# Configurable paths
TRADE_LOG_PATH = "logs/trades.csv"  # Assumes trade logs are stored as CSV
DAILY_REPORT_PATH = "reports/daily_review_{date}.md"
WEEKLY_REPORT_PATH = "reports/weekly_review_{date}.md"

os.makedirs("reports", exist_ok=True)

def load_trades():
    if not os.path.exists(TRADE_LOG_PATH):
        print(f"No trade log found at {TRADE_LOG_PATH}")
        return pd.DataFrame()
    return pd.read_csv(TRADE_LOG_PATH)

def generate_daily_report(date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    trades = load_trades()
    if trades.empty:
        print("No trades to report.")
        return
    trades_today = trades[trades['date'] == date]
    signals = trades_today['signal'].count() if 'signal' in trades_today else len(trades_today)
    trades_count = len(trades_today)
    pnl = trades_today['pnl'].sum() if 'pnl' in trades_today else 0
    drawdown = trades_today['drawdown'].min() if 'drawdown' in trades_today else 0
    errors = trades_today['error'].dropna().tolist() if 'error' in trades_today else []
    notes = ""
    # Markdown output
    md = f"""# Daily Review: {date}
| Date       | Signals | Trades | PnL   | Drawdown | Errors/Issues | Notes/Ideas |
|------------|---------|--------|-------|----------|--------------|-------------|
| {date} | {signals} | {trades_count} | {pnl:.2f} | {drawdown:.2f} | {', '.join(errors)} | {notes} |
"""
    with open(DAILY_REPORT_PATH.format(date=date), "w") as f:
        f.write(md)
    # CSV output
    csv_path = DAILY_REPORT_PATH.format(date=date).replace('.md', '.csv')
    df = pd.DataFrame([[date, signals, trades_count, pnl, drawdown, ', '.join(errors), notes]],
                      columns=["Date", "Signals", "Trades", "PnL", "Drawdown", "Errors/Issues", "Notes/Ideas"])
    df.to_csv(csv_path, index=False)
    print(f"Daily report generated: {DAILY_REPORT_PATH.format(date=date)} and {csv_path}")

def generate_weekly_report(end_date=None):
    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=6)
    trades = load_trades()
    if trades.empty:
        print("No trades to report.")
        return
    trades_week = trades[(trades['date'] >= start_date.strftime("%Y-%m-%d")) & (trades['date'] <= end_date.strftime("%Y-%m-%d"))]
    pnl = trades_week['pnl'].sum() if 'pnl' in trades_week else 0
    drawdown = trades_week['drawdown'].min() if 'drawdown' in trades_week else 0
    win_rate = (trades_week['pnl'] > 0).mean() if 'pnl' in trades_week else 0
    best_trade = trades_week.loc[trades_week['pnl'].idxmax()] if not trades_week.empty else None
    worst_trade = trades_week.loc[trades_week['pnl'].idxmin()] if not trades_week.empty else None
    top_issue = ""
    top_improvement = ""
    next_steps = ""
    # Markdown output
    md = f"""# End-of-Week Review: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
| Metric         | Value/Notes                        |
|----------------|------------------------------------|
| Total PnL      | {pnl:.2f}                          |
| Max Drawdown   | {drawdown:.2f}                     |
| Win Rate       | {win_rate:.2%}                     |
| Best/Worst Trade| {best_trade['pnl'] if best_trade is not None else ''} / {worst_trade['pnl'] if worst_trade is not None else ''} |
| Top Issue      | {top_issue}                        |
| Top Improvement| {top_improvement}                  |
| Next Steps     | {next_steps}                       |
"""
    with open(WEEKLY_REPORT_PATH.format(date=end_date.strftime("%Y-%m-%d")), "w") as f:
        f.write(md)
    # CSV output
    csv_path = WEEKLY_REPORT_PATH.format(date=end_date.strftime("%Y-%m-%d")).replace('.md', '.csv')
    df = pd.DataFrame([
        ["Total PnL", pnl],
        ["Max Drawdown", drawdown],
        ["Win Rate", f"{win_rate:.2%}"],
        ["Best/Worst Trade", f"{best_trade['pnl'] if best_trade is not None else ''} / {worst_trade['pnl'] if worst_trade is not None else ''}"],
        ["Top Issue", top_issue],
        ["Top Improvement", top_improvement],
        ["Next Steps", next_steps],
    ], columns=["Metric", "Value/Notes"])
    df.to_csv(csv_path, index=False)
    print(f"Weekly report generated: {WEEKLY_REPORT_PATH.format(date=end_date.strftime('%Y-%m-%d'))} and {csv_path}")

if __name__ == "__main__":
    generate_daily_report()
    generate_weekly_report() 