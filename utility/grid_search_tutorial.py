from prosperity3bt.runner import run_backtest
from prosperity3bt.file_reader import PackageResourcesReader
from prosperity3bt.models import TradeMatchingMode
import itertools 
import numpy as np
import json
from pathlib import Path

from tutorial import Trader

#here we need to input all the products in the current round
products = ["RAINFOREST_RESIN", 
            "KELP"] 

def run_gridsearch(param_grid, 
                   default_args, 
                   round_num = 0, 
                   day_num = -2, 
                   independent_products = True):
    
    out = []

    if independent_products:
        for product in products:
            product_param_grid = param_grid[product]

            product_param_names = product_param_grid.keys()

            param_combinations = list(itertools.product(*product_param_grid.values()))

            for combination in param_combinations:
                params = dict(zip(product_param_names, combination))
                
                strategy_args = default_args
                strategy_args[product] = params

                trader = Trader(strategy_args)

                out.append(run(trader, round_num, day_num))

    return out


def run(trader : Trader,
        round_num,
        day_num) -> dict[str, float]:
    
    #this is needed for data input
    file_reader = PackageResourcesReader()
    #this sets the matching mode to the default matching mode
    mode = TradeMatchingMode.all

    results = run_backtest(trader, file_reader, 
             round_num = round_num, 
             day_num = day_num,
             print_output = False,
             no_names = True,
             show_progress_bar = True,
             trade_matching_mode = mode
            )
    
    log = {
        "total_pnl" : 0
    }
    log.update({
        f"total_{product}_pnl" : 0 for product in products
    })
    log.update(trader.strategy_args)

    for product in products:
        pnls = [row.columns[-1] for row in results.activity_logs if row.columns[2] == product]

        final_pnl = pnls[-1]
        log["total_pnl"] += final_pnl
        log[f"total_{product}_pnl"] += final_pnl
    
    return log
    
#this defines how the grid search will be run [default parameter, +/- grid size, increment]
gridsearch_args = {
            "RAINFOREST_RESIN" : {"history_size": [10, 0, 1], 
                                  "soft_liquidation_tresh" : [0.5, 0, 0.05]},
            "KELP" : {"history_size": [40, 20, 1], 
                      "soft_liquidation_tresh" : [0.2, 0.20, 0.02]}
        }

default_args = {
            "RAINFOREST_RESIN" : {"history_size": 10, 
                                  "soft_liquidation_tresh" : 0.5},
            "KELP" : {"history_size": 10, 
                      "soft_liquidation_tresh" : 0.5}
        }

param_grid = {product : {argument : 
                         np.arange(gridsearch_args[product][argument][0] - gridsearch_args[product][argument][1], 
                                     gridsearch_args[product][argument][0] + gridsearch_args[product][argument][1], 
                                     gridsearch_args[product][argument][2]).tolist() 
                         for argument in gridsearch_args[product]} for product in products}

results = run_gridsearch(param_grid, default_args)

output_file = Path(__file__).parent / f"{Path(__file__).stem}_grid_search_tutorial.json"
with output_file.open("w+", encoding="utf-8") as file:
    file.write(json.dumps(results, separators=(",", ":")))





