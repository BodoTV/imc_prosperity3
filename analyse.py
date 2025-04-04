import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and flatten the JSON data
def load_and_flatten(json_path):
    with open(json_path) as f:
        data = json.load(f)

    flat_data = []
    for entry in data:
        row = {
            'total_pnl': entry['total_pnl'],
            'total_RAINFOREST_RESIN_pnl': entry['total_RAINFOREST_RESIN_pnl'],
            'total_KELP_pnl': entry['total_KELP_pnl'],
            'RAINFOREST_RESIN_history_size': entry['RAINFOREST_RESIN']['history_size'],
            'RAINFOREST_RESIN_soft_liquidation_tresh': entry['RAINFOREST_RESIN']['soft_liquidation_tresh'],
            'KELP_history_size': entry['KELP']['history_size'],
            'KELP_soft_liquidation_tresh': entry['KELP']['soft_liquidation_tresh'],
        }
        flat_data.append(row)

    return pd.DataFrame(flat_data)

# Find best-performing rows
def show_best_results(df):
    best_total = df['total_pnl'].max()
    best_rows = df[df['total_pnl'] == best_total]
    print("\n=== Best Results ===")
    print(best_rows)

# Heatmap of KELP param impact
def plot_kelps_heatmap(df):
    pivot = df.pivot_table(
        index='KELP_history_size',
        columns='KELP_soft_liquidation_tresh',
        values='total_KELP_pnl'
    )
    sns.heatmap(pivot, annot=False, fmt=".1f", cmap="viridis")
    plt.title("KELP PnL by History Size and Soft Liquidation Threshold")
    plt.tight_layout()
    plt.show()

# Grouped analysis
def average_per_param(df):
    print("\n=== Average KELP PnL by history_size ===")
    print(df.groupby('KELP_history_size')['total_KELP_pnl'].mean())

    print("\n=== Average KELP PnL by soft_liquidation_tresh ===")
    print(df.groupby('KELP_soft_liquidation_tresh')['total_KELP_pnl'].mean())

if __name__ == "__main__":
    json_path = "grid_search_tutorial_grid_search_tutorial.json"

    df = load_and_flatten(json_path)
    show_best_results(df)
    average_per_param(df)
    plot_kelps_heatmap(df)
