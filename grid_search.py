from prosperity3bt.runner import run_backtest
from prosperity3bt.data import read_day_data
from prosperity3bt.file_reader import PackageResourcesReader

from tutorial import Trader #this imports the trader class we have written with all the necessary dependencies

strategy_args = {
            "RAINFOREST_RESIN" : {"history_size": 10, "soft_liquidation_tresh" : 0.5},
            "KELP" : {"history_size": 3, "soft_liquidation_tresh" : 0.25}
        }

file_reader = PackageResourcesReader()

trader = Trader(strategy_args)
data = read_day_data(file_reader, round_num = 0, day_num = -2, no_names = True)

run_backtest(trader, data, print_output= False, show_progress_bar= False, day_num=-2, round_num=0, no_names = True, trade_matching_mode= None)
