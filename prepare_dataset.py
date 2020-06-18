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
from tqdm import tqdm

jsons = {}
for stock_mode in ['last50', 'SZ500', 'SH500', 'SH50', 'ZZ100', 'random']:
    jsons.clear()
    for start_year in tqdm(range(2012, 2020)):
        try:
            stock = Stock(start_date=datetime.date(start_year, 1, 1),
                          end_date=datetime.date(start_year + 1, 1, 1))
            frame = stock.generate_data_frame(mode=stock_mode)
        except:
            continue
        for algo in tqdm([CORN, EG, Olmar, ONS, Pamr, Rmr, UP]):
            algo = algo(n_stock=len(frame[0]))
            algo.compute_weight(frame)
            evaluate = Eval(relative_price=frame,
                            weight=algo.weights,
                            frequency='daily',
                            transaction_cost=0.000)
            jsons.setdefault(str(start_year), []).append(evaluate.cumulative_wealth)
    json.dump(jsons, open("result/algo/{}.json".format(stock_mode), 'w'))
