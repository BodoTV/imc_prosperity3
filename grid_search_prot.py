from prosperity3bt.runner import run_backtest
from prosperity3bt.file_reader import PackageResourcesReader
from prosperity3bt.models import TradeMatchingMode

#here you can import whatever trader class you want to test
from tutorial import Trader #this imports the trader class we have written with all the necessary dependencies

strategy_args = {
            "RAINFOREST_RESIN" : {"history_size": 10, "soft_liquidation_tresh" : 0.5},
            "KELP" : {"history_size": 3, "soft_liquidation_tresh" : 0.25}
        }

file_reader = PackageResourcesReader()

trader = Trader(strategy_args)
mode = TradeMatchingMode.all

results = run_backtest(trader, file_reader, 
             round_num= 0, 
             day_num = -2,
             print_output = False,
             no_names = True,
             show_progress_bar = True,
             trade_matching_mode = mode
             )

total_pnl = 0

for product in ["RAINFOREST_RESIN", "KELP"]:
    pnls = [row.columns[-1] for row in results.activity_logs if row.columns[2] == product]

    min_pnl = min(pnls)
    max_pnl = max(pnls)
    final_pnl = pnls[-1]

    total_pnl += final_pnl

    print("Product: ", product)

    print("min_pnl = ", min_pnl)
    print("max_pnl = ", max_pnl)
    print("final_pnl = ", final_pnl)

print("total_pnl = ", total_pnl)
