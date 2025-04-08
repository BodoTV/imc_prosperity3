import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class TradingDataAnalyzer:
    def __init__(self, orderbook_file_path: str, trades_file_path: str):
        """
        Initialize trading data: order book and trade history split per product.
        """
        self.orderbook_file_path = orderbook_file_path
        self.trades_file_path = trades_file_path

        # Load and prepare order book data
        self.full_orderbook = pd.read_csv(orderbook_file_path, sep=';', na_values=[''])
        self.kelp = self.full_orderbook[self.full_orderbook['product'] == 'KELP'].copy()
        self.rainforest_resin = self.full_orderbook[self.full_orderbook['product'] == 'RAINFOREST_RESIN'].copy()
        self.squid_ink = self.full_orderbook[self.full_orderbook['product'] == 'SQUID_INK'].copy()
        self.products = {
            'KELP': self.kelp,
            'RAINFOREST_RESIN': self.rainforest_resin,
            'SQUID_INK': self.squid_ink
        }

        # Load and prepare trade data
        self.full_trades = pd.read_csv(trades_file_path, sep=';', na_values=[''])

        # Keep only relevant columns
        trade_cols = ['timestamp', 'symbol', 'price', 'quantity']
        trades = self.full_trades[trade_cols].copy()

        # Split trade data per product
        self.kelp_trades = trades[trades['symbol'] == 'KELP'].copy()
        self.rainforest_resin_trades = trades[trades['symbol'] == 'RAINFOREST_RESIN'].copy()
        self.squid_ink_trades = trades[trades['symbol'] == 'SQUID_INK'].copy()

        self.trade_products = {
            'KELP': self.kelp_trades,
            'RAINFOREST_RESIN': self.rainforest_resin_trades,
            'SQUID_INK': self.squid_ink_trades
        }

    def plot_product_price_levels(self, 
                                  product_names, 
                                  bids, asks, 
                                  sampling_steps=None, 
                                  show_trades=True, 
                                  quantity_threshold=None,
                                  separate_subplots=False):
        """
        Plot bid/ask/mid price levels for a given product from order book data.
        
        Args:
            product_name (str): One of 'KELP', 'RAINFOREST_RESIN', 'SQUID_INK'.
            sampling_steps (list): List of 7 integers for subsampling each of the 7 price lines.
        """

        if isinstance(product_names, str):
            product_names = [product_names]

        if sampling_steps is None:
            sampling_steps = [1] * 7
        assert len(sampling_steps) == 7, "sampling_steps must be a list of 7 integers."

        n = len(product_names)
        fig, axes = plt.subplots(n if separate_subplots else 1, 1, figsize=(14, 5 * n), sharex=not separate_subplots)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for product_name in product_names:
            ax = axes[product_names.index(product_name)]

            if product_name not in self.products:
                print(f"Product '{product_name}' not found.")
                continue

            df = self.products[product_name]
            if df.empty:
                print(f"No order book data found for product: {product_name}")
                continue
        
            time = df['timestamp'].to_numpy()

        # Plot bid prices
            bid_colors = ['green', 'limegreen', 'lightgreen']
            for i in np.array(bids):
                col = f'bid_price_{i}'
                valid = df[col].notna()
                times = time[valid]
                values = df[col].to_numpy()[valid]
                step = sampling_steps[i - 1]
                ax.plot(times[::step], values[::step], label=f'Bid {i}', color=bid_colors[i - 1])

            # Plot ask prices
            ask_colors = ['darkred', 'red', 'lightcoral']
            for i in np.array(asks):
                col = f'ask_price_{i}'
                valid = df[col].notna()
                times = time[valid]
                values = df[col].to_numpy()[valid]
                step = sampling_steps[i + 2]
                ax.plot(times[::step], values[::step], label=f'Ask {i}', color=ask_colors[i - 1])

            # Plot mid price
            col = 'mid_price'
            valid = df[col].notna()
            times = time[valid]
            values = df[col].to_numpy()[valid]
            step = sampling_steps[6]
            ax.plot(times[::step], values[::step], label='Mid Price', color='black', linewidth=2)

            # Plot trades (optional)
            if show_trades:
                trade_df = self.trade_products.get(product_name)
                if trade_df is not None and not trade_df.empty:

                    if quantity_threshold is not None:
                        trade_df = trade_df[trade_df['quantity'] > quantity_threshold]

                    if not trade_df.empty and quantity_threshold is not None:
                        ax.scatter(
                            trade_df['timestamp'],
                            trade_df['price'],
                            color='blue',
                            alpha=0.6,
                            marker='o',
                            s=40,
                            edgecolor='k',
                            label=f'{product_name} - Trades (qty > {quantity_threshold})' if quantity_threshold else f'{product_name} - Trades'
                        )
                    

            if not df.empty:
                time = df['timestamp'].to_numpy()
                tick_positions = np.arange(min(time), max(time) + 1, 100000)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_positions, rotation=45)
                ax.grid(True)
                ax.set_title(f'Bid/Ask/Mid Prices for {product_name}')
                ax.legend()
                ax.set_ylabel('Price', fontsize=12)

        plt.xlabel('Timestamp',  fontsize=12)
        plt.title(f'Bid/Ask/Mid Prices for {product_name}')
        plt.tight_layout()
        plt.show()



# Set file paths for order book and trade data
prices_round_1_day_0 = '/Users/moritzrautenberg/Desktop/prosperity3/round-1-island-data-bottle/prices_round_1_day_-2.csv'  
trades_round_1_day_0 = '/Users/moritzrautenberg/Desktop/prosperity3/round-1-island-data-bottle/trades_round_1_day_-2.csv'  

# Create instance of the analyzer
analyzer = TradingDataAnalyzer(prices_round_1_day_0, trades_round_1_day_0)

# Plot order book price levels
## 'sampling_steps' in plot_product_price_levels defines how many data points to skip between adjacently plotted points
sampling_option1 = [1, 1, 1, 1, 1, 1, 1]
sampling_option2 = [10, 10, 10, 10, 10, 10, 10]

## 'bids'/'asks' in plot_product_price_levels defines which price levels to plot choosing from bid_price_1, ask_price_1, bid_price_2 etc.

analyzer.plot_product_price_levels(['SQUID_INK', 'KELP'], 
                                   bids=[1], 
                                   asks=[1], 
                                   sampling_steps=sampling_option2,
                                   show_trades=True,
                                   quantity_threshold=14,
                                   separate_subplots=True)


# Access order book and trade data directly
kelp_trades_df = analyzer.kelp_trades
kelp_orders_df = analyzer.kelp

