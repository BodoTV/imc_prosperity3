import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from io import StringIO
import json

# --- Load logs ---
def load_logs(file_path):
    with open(file_path, "r") as file:
        raw_log = file.read()

    activities_raw = raw_log.split("Activities log:")[1].split("Trade History:")[0].strip()
    trades_raw = raw_log.split("Trade History:")[1].strip()

    activities_df = pd.read_csv(StringIO(activities_raw), sep=";")
    activities_df["timestamp"] = activities_df["timestamp"].astype(int)

    trades_list = json.loads(trades_raw)
    trades_df = pd.DataFrame(trades_list)
    trades_df["timestamp"] = trades_df["timestamp"].astype(int)
    
    return activities_df, trades_df

# --- Enrich activities ---
def calc_spread_and_vol(df):
    df["bid_vol"] = df[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].sum(axis=1)
    df["ask_vol"] = df[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].sum(axis=1)
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["mid"] = (df["ask_price_1"] + df["bid_price_1"]) / 2
    return df

# --- Profit Computation ---
def calc_profit(trades_df):
    submission_trades = trades_df[
        (trades_df["buyer"] == "SUBMISSION") | (trades_df["seller"] == "SUBMISSION")
    ].copy()

    def trade_sign(row):
        if row["buyer"] == "SUBMISSION":
            return -row["price"] * row["quantity"]
        elif row["seller"] == "SUBMISSION":
            return row["price"] * row["quantity"]
        return 0

    submission_trades["profit"] = submission_trades.apply(trade_sign, axis=1)
    profit_over_time = submission_trades.groupby(["timestamp", "symbol"])["profit"].sum().unstack().fillna(0)
    profit_over_time_cum = profit_over_time.cumsum()

    # Add total cumulative PnL
    profit_over_time_cum["Total"] = profit_over_time_cum.sum(axis=1)

    return profit_over_time_cum

# --- Rolling Stats ---
def calculate_rolling_stats(df, lookback_period=100):
    if 'mid' not in df.columns:
        raise KeyError("'mid' column is missing from DataFrame")
    
    df["rolling_mean"] = df["mid"].rolling(window=lookback_period).mean()
    df["rolling_std"] = df["mid"].rolling(window=lookback_period).std()
    df["upper_band"] = df["rolling_mean"] + 2 * df["rolling_std"]
    df["lower_band"] = df["rolling_mean"] - 2 * df["rolling_std"]
    return df

# --- Overall Performance Metrics ---
def calculate_performance_metrics(profit_series):
    total_profit = profit_series.sum()
    win_rate = (profit_series > 0).mean()
    average_profit = profit_series.mean()
    max_drawdown = (profit_series.cumsum() - profit_series.cumsum().cummax()).min()
    volatility = profit_series.std()
    
    return {
        "total_profit": total_profit,
        "win_rate": win_rate,
        "average_profit": average_profit,
        "max_drawdown": max_drawdown,
        "volatility": volatility
    }

# --- Per-product Dashboard ---
def plot_dashboard(activities_df, profit_over_time_cum, product, lookback_period=100):
    act_sub = activities_df[activities_df["product"] == product].copy()
    pnl = profit_over_time_cum[product] if product in profit_over_time_cum else pd.Series(index=act_sub["timestamp"], data=0)

    act_sub = act_sub.set_index("timestamp").sort_index()
    pnl = pnl.reindex(act_sub.index, method="ffill").fillna(0)

    act_sub = calculate_rolling_stats(act_sub, lookback_period)
    performance_metrics = calculate_performance_metrics(pnl)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{product} - PnL Over Time", f"{product} - Spread", f"{product} - Mid Price & Bollinger Bands", f"{product} - Rolling Std of Mid Price")
    )

    fig.add_trace(go.Scatter(x=act_sub.index, y=pnl, mode="lines", name="PnL", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=act_sub.index, y=act_sub["spread"], mode="lines", name="Spread", line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Scatter(x=act_sub.index, y=act_sub["mid"], name="Mid", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=act_sub.index, y=act_sub["rolling_mean"], name="Rolling Mean", line=dict(color="darkblue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=act_sub.index, y=act_sub["upper_band"], name="Upper Band", line=dict(color="lightblue"), opacity=0.4), row=3, col=1)
    fig.add_trace(go.Scatter(x=act_sub.index, y=act_sub["lower_band"], name="Lower Band", line=dict(color="lightblue"), opacity=0.4, fill='tonexty'), row=3, col=1)
    fig.add_trace(go.Scatter(x=act_sub.index, y=act_sub["rolling_std"], name="Rolling Std", line=dict(color="purple")), row=4, col=1)

    fig.update_layout(
        height=1000, width=1000, title_text=f"{product} - Market Making Metrics", showlegend=True,
        annotations=[dict(
            text=f"Total Profit: {performance_metrics['total_profit']:.2f} | "
                 f"Win Rate: {performance_metrics['win_rate']*100:.2f}% | "
                 f"Avg Profit: {performance_metrics['average_profit']:.2f} | "
                 f"Max Drawdown: {performance_metrics['max_drawdown']:.2f} | "
                 f"Volatility: {performance_metrics['volatility']:.2f}",
            x=0.5, 
            y=-0.15, 
            xref="paper", 
            yref="paper", 
            showarrow=False, 
            font=dict(size=12)
        )]
    )

    fig.show()

# --- Cumulative Performance Dashboard ---
def plot_overall_dashboard(activities_df, profit_over_time_cum):
    product_list = [col for col in profit_over_time_cum.columns if col != "Total"]
    rows = 2 + len(product_list)

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=(
            "Profit / Loss",
            "Positions (% of limit)",
            *[f"{prod} - Price" for prod in product_list],
            *[f"{prod} - Volume" for prod in product_list]
        ),
        shared_xaxes=True,
        vertical_spacing=0.03
    )

    # Profit / Loss
    for symbol in profit_over_time_cum.columns:
        fig.add_trace(go.Scatter(x=profit_over_time_cum.index, y=profit_over_time_cum[symbol], name=symbol, mode="lines"), row=1, col=1)

    # Positions (if 'position' column exists)
    for product in product_list:
        sub = activities_df[activities_df["product"] == product].copy()
        sub = sub.set_index("timestamp").sort_index()
        if "position" in sub.columns:
            fig.add_trace(go.Scatter(x=sub.index, y=sub["position"], name=f"{product} Position"), row=1, col=2)

    # Prices and Volumes
    for idx, product in enumerate(product_list):
        row = 2 + idx
        sub = activities_df[activities_df["product"] == product].copy()
        sub = sub.set_index("timestamp").sort_index()

        # Price
        fig.add_trace(go.Scatter(x=sub.index, y=sub["bid_price_1"], name=f"{product} Bid", line=dict(color="green")), row=row, col=1)
        fig.add_trace(go.Scatter(x=sub.index, y=sub["ask_price_1"], name=f"{product} Ask", line=dict(color="red")), row=row, col=1)

        # Volume
        fig.add_trace(go.Bar(x=sub.index, y=sub["bid_vol"], name=f"{product} Bid Vol", marker_color="lightgreen"), row=row, col=2)
        fig.add_trace(go.Bar(x=sub.index, y=sub["ask_vol"], name=f"{product} Ask Vol", marker_color="salmon"), row=row, col=2)

    fig.update_layout(
        height=350 * rows,
        width=1500,
        title_text="Cumulative Performance Dashboard",
        showlegend=True
    )

    fig.show()

# --- Main Analysis ---
activities_df, trades_df = load_logs(r"C:\Users\timmv\Downloads\fb63196f-bf1f-45fd-924f-3c3fa471cb45_final.log")

activities_df = activities_df.groupby("product").apply(calc_spread_and_vol).reset_index(drop=True)
profit_over_time_cum = calc_profit(trades_df)

for product in activities_df["product"].unique():
    plot_dashboard(activities_df, profit_over_time_cum, product, lookback_period=100)

plot_overall_dashboard(activities_df, profit_over_time_cum)
