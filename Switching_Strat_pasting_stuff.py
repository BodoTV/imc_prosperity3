import numpy as np
from collections import deque
from abc import ABC, abstractmethod

class SwitchingStrategy(Strategy):
    def __init__(self, product: str, limit: int):
        super().__init__(product, limit)
        self.market_making = MarketMakingStrategy(product, limit)
        self.mean_reversion = MeanReversionStrategy(product, limit)
        self.trend_following = TrendFollowingStrategy(product, limit)
        self.current_strategy = self.market_making  # Start with Market Making
        self.price_history = deque(maxlen=50)  # Store recent prices

    def act(self, state: TradingState) -> None:
        # Update price history
        price_history = HelperFunctions.get_price_history(state, self.product)
        if price_history:
            self.price_history.extend(price_history[-1:])  # Append latest price

        if len(self.price_history) < 10:
            self.current_strategy = self.market_making  # Default to Market Making if not enough data
        else:
            # Compute moving average and standard deviation
            moving_avg = HelperFunctions.get_moving_average(self.price_history, 10, "SMA")
            std_dev = HelperFunctions.get_moving_standard_deviation(self.price_history, 10)

            # Compute trend strength using linear regression slope
            X = np.arange(len(self.price_history)).reshape(-1, 1)
            y = np.array(self.price_history)
            slope = HelperFunctions.lin_reg(X, y)[1]  # Get slope from regression

            # Define signal-based strategy switching
            if std_dev < 5:  
                self.current_strategy = self.market_making  # Low volatility → Market Making
            elif abs(self.price_history[-1] - moving_avg) > 2 * std_dev:
                self.current_strategy = self.mean_reversion  # Price too far from mean → Mean Reversion
            elif abs(slope) > 0.5:
                self.current_strategy = self.trend_following  # Strong trend detected → Trend Following

        # Run the selected strategy
        self.current_strategy.act(state)
        self.orders.extend(self.current_strategy.orders)




##############################################################################
class MeanReversionStrategy(Strategy):
    def __init__(self, product: str, limit: int):
        super().__init__(product, limit)
        self.lookback = 10  # Lookback for moving average

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.product, 0)
        order_depth = state.order_depths[self.product]
        current_price = HelperFunctions.get_price_history(state, self.product)[-1]
        prices = list(HelperFunctions.get_price_history(state, self.product))

        if len(prices) < self.lookback:
            return

        mean_price = HelperFunctions.get_moving_average(prices, self.lookback, "SMA")
        std_dev = HelperFunctions.get_moving_standard_deviation(prices, self.lookback)

        # Mean reversion signal: price far from mean
        if current_price > mean_price + std_dev and position > -self.limit:
            # Sell signal
            self.sell(current_price, min(self.limit + position, 10))
        elif current_price < mean_price - std_dev and position < self.limit:
            # Buy signal
            self.buy(current_price, min(self.limit - position, 10))

#############################################################################################################

class TrendFollowingStrategy(Strategy):
    def __init__(self, product: str, limit: int):
        super().__init__(product, limit)
        self.lookback = 10

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.product, 0)
        price_history = list(HelperFunctions.get_price_history(state, self.product))

        if len(price_history) < self.lookback:
            return

        X = np.arange(len(price_history)).reshape(-1, 1)
        y = np.array(price_history)
        slope = HelperFunctions.lin_reg(X, y)[1]  # Trend slope

        current_price = price_history[-1]

        # Follow the trend if it's strong enough
        if slope > 0.5 and position < self.limit:
            self.buy(current_price, min(self.limit - position, 10))
        elif slope < -0.5 and position > -self.limit:
            self.sell(current_price, min(self.limit + position, 10))


