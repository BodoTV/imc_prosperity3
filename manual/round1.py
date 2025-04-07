import itertools

products = ["SNOWBALLS", "PIZZA", "NUGGETS", "SEA_SHELLS"]

product_index = {"SNOWBALLS" : 0, 
                 "PIZZA" : 1, 
                 "NUGGETS" : 2, 
                 "SEA_SHELLS" : 3}

conversion_table = [
     [1 , 1.45, 0.52, 0.72],
     [0.7, 1, 0.31, 0.48],
     [1.95, 3.1, 1, 1.49],
     [1.34, 1.98, 0.64, 1]
]
#conversion_table[from][to]

def convert(product_from : str, product_to : str, factor : float) -> float:
    return conversion_table[product_index[product_from]][product_index[product_to]] * factor

def run_strategy(trades):
    print("======================")

    factor = 1.0

    factor = convert("SEA_SHELLS", trades[0], factor)
    print("Current Trade : ", "SEA_SHELLS" " -> ", trades[0], " = ", factor)

    for i in range(len(trades) - 1):
        
        factor = convert(trades[i], trades[i+1], factor)
        print("Current Trade : ", trades[i], " -> ", trades[i+1], " = ", factor)

    factor = convert(trades[-1], "SEA_SHELLS", factor)
    print("Current Trade : ", trades[-1], " -> ", "SEA_SHELLS", " = ", factor)

    print("factor = ", factor)
    print("profit = ", factor * 2_000_000 - 2_000_000)

    return factor

possible_trades = list(itertools.product(products, repeat = 4))

current_max = 0
best_strategy = []

for trades in possible_trades:
    curr = run_strategy(trades)
    if curr > current_max:
        current_max = curr
        best_strategy = trades

print("BEST:")
print("Best strategy = ", run_strategy(best_strategy))



