import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
from io import StringIO

def load_logs(file_path):
    with open(file_path, "r") as file:
        raw_log = file.read()
    activities_raw = raw_log.split("Activities log:")[1].split("Trade History:")[0].strip()
    trades_raw = raw_log.split("Trade History:")[1].strip()
    activities_df = pd.read_csv(StringIO(activities_raw), sep=";")
    trades_list = json.loads(trades_raw)
    trades_df = pd.DataFrame(trades_list)
    activities_df["timestamp"] = activities_df["timestamp"].astype(int)
    trades_df["timestamp"] = trades_df["timestamp"].astype(int)
    return activities_df, trades_df

def calculate_profit_and_positions(trades_df):
    submission_trades = trades_df[
        (trades_df["buyer"] == "SUBMISSION") | (trades_df["seller"] == "SUBMISSION")
    ].copy()

    def signed_qty(row):
        return row["quantity"] if row["seller"] == "SUBMISSION" else -row["quantity"]
    def signed_cashflow(row):
        return row["price"] * row["quantity"] if row["seller"] == "SUBMISSION" else -row["price"] * row["quantity"]

    submission_trades["signed_qty"] = submission_trades.apply(signed_qty, axis=1)
    submission_trades["signed_cash"] = submission_trades.apply(signed_cashflow, axis=1)

    profit_df = submission_trades.groupby(["timestamp", "symbol"])[["signed_cash"]].sum().unstack(fill_value=0)
    profit_cum = profit_df.cumsum()
    profit_cum.columns = profit_cum.columns.get_level_values(1)

    qty_df = submission_trades.groupby(["timestamp", "symbol"])[["signed_qty"]].sum().unstack(fill_value=0)
    qty_cum = qty_df.cumsum()
    qty_cum.columns = qty_cum.columns.get_level_values(1)

    return profit_cum, qty_cum

def calculate_statistics(profit_cum):
    stats = {}
    for symbol in profit_cum.columns:
        pnl = profit_cum[symbol].diff().dropna()
        mean_return = pnl.mean()
        std_return = pnl.std()
        sharpe = mean_return / std_return if std_return != 0 else np.nan
        var_95 = np.percentile(pnl, 5)
        final_value = profit_cum[symbol].iloc[-1]
        stats[symbol] = {
            "final": final_value,
            "sharpe": sharpe,
            "var_95": var_95
        }
    return stats

def plot_overall_cumulative_page(profit_cum, qty_cum, stats):
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Cumulative PnL", "Cumulative Position (% of limit)"),
        horizontal_spacing=0.15
    )

    total_pnl = profit_cum.sum(axis=1)
    fig.add_trace(go.Scatter(x=total_pnl.index, y=total_pnl, name="Total", line=dict(width=3)), row=1, col=1)

    for symbol in profit_cum.columns:
        pnl = profit_cum[symbol]
        label = f"{symbol} | PnL: {stats[symbol]['final']:.2f}, Sharpe: {stats[symbol]['sharpe']:.2f}, VaR: {stats[symbol]['var_95']:.2f}"
        fig.add_trace(go.Scatter(x=pnl.index, y=pnl, name=label, line=dict(dash="dot")), row=1, col=1)

    for symbol in qty_cum.columns:
        pos_pct = 100 * qty_cum[symbol] / 100  # assuming position limit = 100
        fig.add_trace(go.Scatter(x=pos_pct.index, y=pos_pct, name=symbol), row=1, col=2)

    fig.update_layout(title_text=f"Final Profit / Loss Dashboard", height=500)
    fig.show()

def plot_individual_dashboards(activities_df, profit_cum, qty_cum):
    for symbol in profit_cum.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f"{symbol} - PnL", f"{symbol} - Position (% of limit)"))

        fig.add_trace(go.Scatter(x=profit_cum.index, y=profit_cum[symbol], name="PnL"), row=1, col=1)
        pos_pct = 100 * qty_cum[symbol] / 100  # assuming position limit = 100
        fig.add_trace(go.Scatter(x=qty_cum.index, y=pos_pct, name="Position %"), row=2, col=1)

        fig.update_layout(title_text=f"{symbol} Dashboard", height=600)
        fig.show()

# --- MAIN ---
if __name__ == "__main__":
    file_path = r"C:\Users\timmv\Downloads\fb63196f-bf1f-45fd-924f-3c3fa471cb45_final.log"
    activities_df, trades_df = load_logs(file_path)
    profit_cum, qty_cum = calculate_profit_and_positions(trades_df)
    stats = calculate_statistics(profit_cum)

    # Page 1: Overall cumulative view
    plot_overall_cumulative_page(profit_cum, qty_cum, stats)

    # Pages per product
    plot_individual_dashboards(activities_df, profit_cum, qty_cum)
