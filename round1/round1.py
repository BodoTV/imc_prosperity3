import json
from typing import Any
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque
from typing import Any, TypeAlias
import numpy as np

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
        self.orders.append(Order(self.product, int(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, int(price), -quantity))

    #this is for transferring data from one trader to the next
    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, 
                 product: str, limit: int, 
                 strategy_args):
        super().__init__(product, limit)

        self.history = deque()
        self.mid_price_history = deque(maxlen=10)

        self.EMA_alpha = strategy_args.get("EMA_alpha", 0.32)
        self.history_size = strategy_args.get("history_size", 10)
        self.soft_liquidate_thresh = strategy_args.get("soft_liquidation_tresh", 0.5)
        self.volatility_multiplier = strategy_args.get("volatility_multiplier", 1.0)
        self.EMA = None

    def get_popular_average(self, state : TradingState) -> int:
        #calculate the average between the most popular buy and sell price
        order_depths = state.order_depths[self.product]
        sell_orders = order_depths.sell_orders.items()
        buy_orders = order_depths.buy_orders.items()

        most_popular_sell_price = min(sell_orders, key = lambda item : item[1])[0]
        most_popular_buy_price = max(buy_orders, key = lambda item : item[1])[0]
        
        #calculate average of those prices
        return (most_popular_buy_price + most_popular_sell_price)//2

    #returns and updates the current EMA
    def get_EMA(self, state : TradingState) -> float:

        average_price = self.get_popular_average(state)

        alpha = self.EMA_alpha

        if self.EMA == None:
            self.EMA = average_price
        else:
            self.EMA = average_price * alpha + (1 - alpha) * self.EMA

        return self.EMA
    
    #estimates volatility as std from the last 20 mid prices
    def estimate_volatility(self):
        if len(self.mid_price_history)<5:
            return 0
        return np.std(self.mid_price_history)
    

    def act(self, state: TradingState) -> None:
        ##Logic
        #sort buy and sell orders on the market for later use
        #this priotizes the cheapest sell offer and the highest buy offers
        buy_orders = sorted(state.order_depths[self.product].buy_orders.items(), reverse = True)
        sell_orders = sorted(state.order_depths[self.product].sell_orders.items())

        #use the mid_price to calculate a volatility
        mid_price = self.get_popular_average(state)
        self.mid_price_history.append(mid_price)
        volatility = self.estimate_volatility()

        #how much we can buy/sell of this specific product
        position = state.position.get(self.product, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        default_price = self.get_default_price(state)

        #calculate a spread_corr depending on the volatility
        spread_corr = round(self.volatility_multiplier * volatility)

        #if we are completely short or long than add a true to the history
        self.history.append(abs(position) == self.limit)
        
        #if history is full remove first element
        if len(self.history) > self.history_size:
            self.history.popleft()

        #define if we want to hard or soft liquidate (default to false if the history isnt full)
        #soft: if more than half of the history is true and the last one is true
        soft_liquidate = len(self.history) == self.history_size and sum(self.history) >= self.history_size * self.soft_liquidate_thresh and self.history[-1]
        #hard: if all of the history is true
        hard_liquidate = len(self.history) == self.history_size and all(self.history)

        #now we want to define if we want to buy or sell more depending on how full our position is
        #we can regulate the prob. of buying and selling by increasing/decreasing the max_buy_price/min_sell_price

        #buy less likely if position to long
        max_buy_price = default_price - 1 - spread_corr if position > self.limit * 0.5 else default_price - spread_corr

        #sell less likely if position is to short
        min_sell_price = default_price + 1 + spread_corr if position < self.limit * -0.5 else default_price + spread_corr

        #buy as much as possible below the max_buy_price starting with cheaper offers first
        for price, volume in sell_orders:
            if price <= max_buy_price and to_buy > 0:
                #buy as much as possible, either restricted by limit or by order volume
                quantity = min(-volume, to_buy)
                self.buy(price, quantity)
                to_buy -= quantity
                
        #do the hard liquidation for buying --> we are very short
        if to_buy > 0 and hard_liquidate:
            #buy half of whats possible for the default price of that product
            quantity = to_buy // 2
            self.buy(default_price, quantity)
            to_buy -= quantity

        #do the soft liquidation for buying --> we are short but not that short
        if to_buy > 0 and soft_liquidate:
            #buy half of whats possible for the default price of that product - some value
            #this makes buying less likely
            quantity = to_buy //2
            self.buy(default_price - 2, quantity)
            to_buy -= quantity

        #do the normal buying without hard or soft liquidation
        #for this we need to know the most popular buy price
        if to_buy > 0:
            #this finds the buy price of the highest volume
            most_popular_price = max(buy_orders, key = lambda item: item[1])[0]
            #now we either buy at this price + 1 or the max_buy_price depending on whats smaller
            #--> we never buy over the max_buy_price
            price = min(max_buy_price, most_popular_price + 1)
            #now we buy as much as possible at this price
            self.buy(price, to_buy)

        #sell as much as possible above the min_sell_price starting with the highest offer first
        for price, volume in buy_orders:
            if price >= min_sell_price and to_sell > 0:
                #sell as much as possible, either restriced by limit or by order volume
                quantity = min(volume, to_sell)
                self.sell(price, quantity)
                to_sell -= quantity

        #now do the hard liquidation for selling --> we are completely long
        if to_sell > 0 and hard_liquidate:            
            #sell half of whats possible at default price
            quantity = to_sell // 2
            self.sell(default_price, quantity)
            to_sell -= quantity

        #now do the soft liquidation for selling --> we are very short
        if to_sell > 0 and soft_liquidate:
            #sell half of whats possible at default price + some value making it less likely
            quantity = to_sell // 2
            self.sell (default_price + 2, quantity)
            to_sell -= quantity

        #now do the normal selling at the most popular selling price
        if to_sell > 0:
            #find sell price with highest volume (this means the smallest volume)
            most_popular_price = min(sell_orders, key = lambda item: item[1])[0]
            #now we either sell at the most_popular_price or at the min_sell_price what ever is higher
            #-> we never sell under the min_sell_price
            price = max(min_sell_price, most_popular_price - 1)
            #now sell as much as possible for this price
            self.sell(price, to_sell)

    @abstractmethod
    def get_default_price(self, state: TradingState) -> int:
        raise NotImplementedError()
    
    #this saves the current history
    def save(self) -> JSON:
        return list(self.history)
    
    #this can load a history
    def load(self, data : JSON) -> None:
        self.history = deque(data)


class RainForestResinStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> int:
        return 10_000

class SquidInkStrategy(MarketMakingStrategy):
    def get_default_price(self, state):
        return self.get_popular_average(state)

class KelpStrategy(MarketMakingStrategy):
    #for kelp try a marketmaking strategy with a dynamic default price
    def get_default_price(self, state: TradingState) -> int:
        return self.get_EMA(state)

class Trader:
    def __init__(self, strategy_args = None) -> None:
        #Define the limits and goods traded in the given round, this need to be changed for every submission
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP" : 50,
            "SQUID_INK" : 50
        }

        self.strategy_args = strategy_args if strategy_args != None else {}
        #Define a strategy for every product, this is done by creating an instance of a specific strategy for every product
        self.strategies = { symbol : strategyClass(symbol, limits[symbol], self.strategy_args.get(symbol, {})) for symbol, strategyClass in {
            "RAINFOREST_RESIN" : RainForestResinStrategy,
            "KELP" : KelpStrategy,
            "SQUID_INK": SquidInkStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        
        #saves what ever comes from the previous iteration into a dictionary (if empty its just a empty dictionary)
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        #creates empty dictionary
        new_trader_data = {}

        #need to modify the input of parameters into a specific strategy

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