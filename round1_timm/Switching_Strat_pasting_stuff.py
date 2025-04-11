import json
import numpy as np
from typing import Any
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque
from typing import Any, TypeAlias

#this sets JSON as the type alias for everything that can be a json
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
#this class is written by jmerle and needed for using the visualizer and backtester (just ignore)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class HelperFunctions: 
    price_history = {}
    @staticmethod
    def get_mid_price(state: TradingState, product: str) -> float | None:
        """Calculate the mid-price from the most popular buy and sell prices."""
        order_depths = state.order_depths[product]
        buy_orders = order_depths.buy_orders.items()
        sell_orders = order_depths.sell_orders.items()
        
        if not buy_orders or not sell_orders:
            return None  # Can't compute mid-price without both sides

        # Find price with max volume on each side
        most_popular_buy_price = max(buy_orders, key=lambda item: item[1])[0]
        most_popular_sell_price = min(sell_orders, key=lambda item: item[1])[0]
        
        return (most_popular_buy_price + most_popular_sell_price) / 2
    
    @staticmethod
    def get_price_history(state: TradingState, product: str) -> list[float]: #### is it really working correctly?
        mid_price = HelperFunctions.get_mid_price(state, product)
        
        if product not in HelperFunctions.price_history:
            HelperFunctions.price_history[product] = []

        if mid_price is not None:
            HelperFunctions.price_history[product].append(mid_price)
        
        return HelperFunctions.price_history[product]

    @staticmethod
    def get_moving_average(prices: list[float], lookback: int, moving_avg_type: str = 'SMA') -> float:
        """Get the moving average for the given prices with a specific moving average type"""
        if len(prices) < lookback:
            return None  # Not enough data
        
        if moving_avg_type == 'SMA':  # Simple Moving Average
            return np.mean(prices[-lookback:])
        elif moving_avg_type == 'EMA':  # Exponential Moving Average
            alpha = 2 / (lookback + 1)
            ema = prices[-lookback]
            for price in prices[-lookback+1:]:
                ema = alpha * price + (1 - alpha) * ema
            return ema
        elif moving_avg_type == 'WMA':  # Weighted Moving Average
            weights = np.arange(1, lookback + 1) #assigns integer weights, most recent one is most important.
            weighted_sum = np.dot(prices[-lookback:], weights)
            return weighted_sum / weights.sum()
        return np.mean(prices[-lookback:])  # Default to SMA if invalid type
    
    @staticmethod
    def get_moving_standard_deviation(prices: list[float], lookback: int) -> float:
        """Get the moving standard deviation for the given prices"""
        if len(prices) < lookback:
            return None  # Not enough data
        return np.std(prices[-lookback:])
    
    @staticmethod
    def lin_reg(X,y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X] # add ones for intercept
        theta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)
        return theta

class Strategy:
    def __init__(self, product: str, limit: int):
        self.product = product
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, price, -quantity))

    #this is for transferring data from one trader to the next
    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass
class LinearRegressionForecastingStrategy(Strategy):
    def __init__(self, product: str, limit: int, lookback: int = 10):
        super().__init__(product, limit)
        self.lookback = lookback

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.product, 0)
        price_history = list(HelperFunctions.get_price_history(state, self.product))
        
        if len(price_history) < self.lookback:
            return

        # Calculate the linear regression to forecast the next price
        X = np.arange(len(price_history)).reshape(-1, 1)
        y = np.array(price_history)
        intercept, slope = HelperFunctions.lin_reg(X, y)

        # Forecast the next price (next point in the trend)
        forecasted_price = intercept + slope * (len(price_history))

        current_price = price_history[-1]

        # Calculate how much we can buy/sell based on current position and limit
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Buy if the forecasted price is higher than the current price
        if forecasted_price > current_price and to_buy > 0:
            self.buy(current_price, to_buy)

        # Sell if the forecasted price is lower than the current price
        elif forecasted_price < current_price and to_sell > 0:
            self.sell(current_price, to_sell)

class RegimeSwitchingStrategy(Strategy):
    def __init__(
        self,
        product: str,
        limit: int,
        lookback: int = 10,
        vol_lookback: int = 10,
        low_vol_threshold: float = 1.0,
        high_vol_threshold: float = 3.0,
        mean_type: str = "SMA",
        std_threshold: float = 1.0,
    ):
        super().__init__(product, limit)
        self.product = product
        self.limit = limit
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.mean_type = mean_type
        self.std_threshold = std_threshold

        # Instantiate underlying strategies
        self.market_making = MarketMakingStrategy(product, limit)
        self.market_making.get_default_price = self.get_default_price
        self.mean_reversion = MeanReversionStrategy(
            product, limit, lookback=lookback,
            std_threshold=std_threshold,
            mean_type=mean_type
        )
        self.trend_following = TrendFollowingStrategy(
            product, limit, lookback=lookback,
            mean_type=mean_type
        )

        self.orders = []

    def act(self, state: TradingState) -> None:
        prices = HelperFunctions.get_price_history(state, self.product)

        if len(prices) < self.vol_lookback:
            # Not enough data — use market making as fallback
            self.market_making.act(state)
            self.orders.extend(self.market_making.run(state))
            return

        # Calculate volatility
        volatility = HelperFunctions.get_moving_standard_deviation(prices, self.vol_lookback)

        # Select strategy based on volatility
        if volatility is None:
            self.market_making.act(state)
        elif volatility < self.low_vol_threshold:
            self.mean_reversion.act(state)
        elif volatility > self.high_vol_threshold:
            self.trend_following.act(state)
        else:
            self.market_making.act(state)

    @abstractmethod
    def get_default_price(self, state: TradingState) -> int:
        raise NotImplementedError()

    # Save current history state
    def save(self) -> JSON:
        return list(self.history)

    # Load previously saved history
    def load(self, data: JSON) -> None:
        self.history = deque(data)

class TrendFollowingStrategy(RegimeSwitchingStrategy):
    def __init__(self, product: str, limit: int, lookback: int = 10, mean_type: str = "SMA"):
        super().__init__(product, limit)
        self.lookback = lookback
        self.mean_type = mean_type  # Mean type for trend calculation ("SMA", "EMA", "WMA", etc.)

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.product, 0)
        price_history = list(HelperFunctions.get_price_history(state, self.product))

        if len(price_history) < self.lookback:
            return

        # Calculate the mean price (SMA, EMA, WMA, etc.) over the price history
        mean_price = HelperFunctions.get_moving_average(price_history, self.lookback, self.mean_type)

        current_price = price_history[-1]

        # Calculate how much we can buy/sell based on current position and limit
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Buy if the mean price is above the current price (indicating a potential uptrend)
        if mean_price > current_price and to_buy > 0:
            self.buy(current_price, to_buy)

        # Sell if the mean price is below the current price (indicating a potential downtrend)
        elif mean_price < current_price and to_sell > 0:
            self.sell(current_price, to_sell)

    

class MeanReversionStrategy(RegimeSwitchingStrategy):
    def __init__(self, product: str, limit: int, lookback: int = 10, std_threshold: float = 1.0, mean_type: str = "SMA"):
        super().__init__(product, limit)
        self.lookback = lookback
        self.std_threshold = std_threshold
        self.mean_type = mean_type  # Choose mean type: "SMA", "EMA", "WMA", etc.

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.product, 0)
        prices = list(HelperFunctions.get_price_history(state, self.product))

        if len(prices) < self.lookback:
            return

        current_price = prices[-1]
        mean_price = HelperFunctions.get_moving_average(prices, self.lookback, self.mean_type)
        std_dev = HelperFunctions.get_moving_standard_deviation(prices, self.lookback)

        # Calculate how much we can buy/sell based on current position and limit
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Mean reversion signal: price too far from the mean
        if current_price > mean_price + self.std_threshold * std_dev and to_sell > 0:
            self.sell(current_price, to_sell)
        elif current_price < mean_price - self.std_threshold * std_dev and to_buy > 0:
            self.buy(current_price, to_buy)


class Strategies():
    def __init__(self, product: str, limit: int):
        super().__init__(product, limit)
        self.history = deque()
        self.history_size = 10

    def MarketMaking(self, state: TradingState) -> None:
        # Sort buy and sell orders by price
        # Buy: prioritize highest price (best bid), Sell: prioritize lowest price (best ask)
        buy_orders = sorted(state.order_depths[self.product].buy_orders.items(), reverse=True)
        sell_orders = sorted(state.order_depths[self.product].sell_orders.items())

        position = state.position.get(self.product, 0)

        # Calculate how much we can buy/sell based on current position and limit
        to_buy = self.limit - position
        to_sell = self.limit + position

        default_price = self.get_default_price(state)

        # Update position history for liquidation checks
        self.history.append(abs(position) == self.limit)
        if len(self.history) > self.history_size:
            self.history.popleft()

        # Determine liquidation triggers (only evaluate if history is full)
        soft_liquidate = (
            len(self.history) == self.history_size and
            sum(self.history) >= self.history_size / 2 and
            self.history[-1]
        )
        hard_liquidate = (
            len(self.history) == self.history_size and
            all(self.history)
        )

        # Adjust aggressiveness based on position exposure
        max_buy_price = default_price - 1 if position > 0.5 * self.limit else default_price
        min_sell_price = default_price + 1 if position < -0.5 * self.limit else default_price

        # Try to buy at favorable prices (below or equal to max_buy_price)
        for price, volume in sell_orders:
            if price <= max_buy_price and to_buy > 0:
                quantity = min(-volume, to_buy)
                self.buy(price, quantity)
                to_buy -= quantity

        # Hard liquidation for buying (extreme short position)
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(default_price, quantity)
            to_buy -= quantity

        # Soft liquidation for buying (moderate short position)
        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(default_price - 2, quantity)
            to_buy -= quantity

        # Normal buy behavior using the most popular buy price (by volume)
        if to_buy > 0 and buy_orders:
            most_popular_price = max(buy_orders, key=lambda item: item[1])[0]
            price = min(max_buy_price, most_popular_price + 1)
            self.buy(price, to_buy)

        # Try to sell at favorable prices (above or equal to min_sell_price)
        for price, volume in buy_orders:
            if price >= min_sell_price and to_sell > 0:
                quantity = min(volume, to_sell)
                self.sell(price, quantity)
                to_sell -= quantity

        # Hard liquidation for selling (extreme long position)
        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(default_price, quantity)
            to_sell -= quantity

        # Soft liquidation for selling (moderate long position)
        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(default_price + 2, quantity)
            to_sell -= quantity

        # Normal sell behavior using the most popular sell price (by volume)
        if to_sell > 0 and sell_orders:
            most_popular_price = min(sell_orders, key=lambda item: item[1])[0]
            price = max(min_sell_price, most_popular_price - 1)
            self.sell(price, to_sell)

    @abstractmethod
    def get_default_price(self, state: TradingState) -> int:
        raise NotImplementedError()

    # Save current history state
    def save(self) -> JSON:
        return list(self.history)

    # Load previously saved history
    def load(self, data: JSON) -> None:
        self.history = deque(data)



class RainForestResinStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> int:
        # return 10_000
        # Get price history using the helper class
        price_history = HelperFunctions.get_price_history(state, "RAINFOREST_RESIN")

        # Define a reasonable lookback period (e.g., 10 if enough data exists)
        lookback = min(10, len(price_history))  # Ensure we don't exceed available data

        # Compute the moving average
        moving_avg = HelperFunctions.get_moving_average(price_history, lookback, "SMA")

        # Return the moving average if available, otherwise default to 10,000
        return int(moving_avg) if moving_avg is not None else 10_000


class KelpStrategy(MarketMakingStrategy):
    #for kelp try a marketmaking strategy with a dynamic default price
    def get_default_price(self, state: TradingState) -> int:
        #calculate the average between the most popular buy and sell price
        order_depths = state.order_depths[self.product]
        sell_orders = order_depths.sell_orders.items()
        buy_orders = order_depths.buy_orders.items()

        most_popular_sell_price = min(sell_orders, key = lambda item : item[1])[0]
        most_popular_buy_price = max(buy_orders, key = lambda item : item[1])[0]
        
        #calculate average of those prices
        return (most_popular_buy_price + most_popular_sell_price)//2


class Trader:
    def __init__(self) -> None:
        #Define the limits and goods traded in the given round, this need to be changed for every submission
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP" : 50
        }

        #Define a strategy for every product, this is done by creating an instance of a specific strategy for every product
        self.strategies = { symbol : strategyClass(symbol, limits[symbol]) for symbol, strategyClass in {
            "RAINFOREST_RESIN" : RainForestResinStrategy,
            "KELP" : KelpStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        
        #saves what ever comes from the previous iteration into a dictionary (if empty its just a empty dictionary)
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        #creates empty dictionary
        new_trader_data = {}

        #iterate over every product
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                #this just makes sure the current iteration gets the data of the old iteration if important
                #not important in this case right now
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                result[symbol] = strategy.run(state)

            #write the data from the current iteration into the new_trader_data dictionary under the current product
            #in the case of the marketmaking strategy this is just the current 
            new_trader_data[symbol] = strategy.save()

        #convert the dictionary with the current trader_data back into a string with the format of a dictionary
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
