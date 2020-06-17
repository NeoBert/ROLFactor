from data.stock import Stock
import datetime
from algorithm.olps.anticor import Anticor
from algorithm.olps.rmr import Rmr
from algorithm.olps.up import UP
from algorithm.olps.eg import EG
from algorithm.olps.pamr import Pamr
from algorithm.olps.corn import CORN
from algorithm.olps.olmar import Olmar
from algorithm.olps.ons import ONS
from eval import Eval
import json

names = []
jsons = {}
for stock_mode in ['last50', 'SZ500', 'SH500', 'SH50', 'ZZ100', 'random']:
    for start_year in range(2000, 2020):
        for algo in [Anticor, CORN, EG, Olmar, ONS, Pamr, Rmr, UP]:
            try:
                stock = Stock(start_date=datetime.date(start_year, 1, 1),
                              end_date=datetime.date(start_year + 1, 1, 1))
                frame = stock.generate_data_frame(mode=stock_mode)
            except:
                break
            algo = algo(n_stock=len(frame[0]))
            algo.compute_weight(frame)
            evaluate = Eval(relative_price=frame,
                            weight=algo.weights,
                            frequency='daily',
                            transaction_cost=0.000)
            jsons.setdefault(str(start_year), []).append(evaluate.cumulative_wealth)
            evaluate.print_info()
    json.dump(jsons, open("result/algo/{}.json".format(stock_mode), 'w'))
