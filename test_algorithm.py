from data.stock import Stock
import datetime
from algorithm.olps.rmr import Rmr
from algorithm.olps.up import UP
from algorithm.olps.eg import EG
from algorithm.olps.pamr import Pamr
from algorithm.olps.olmar import Olmar
from algorithm.olps.ons import ONS
from algorithm.olps.bandit_exp3 import BanditExp3
from eval import Eval
import json

bandits = [[Rmr, 'SZ500'], [UP, 'SZ500'], [EG, 'SZ500'], [Pamr, 'SZ500'], [Olmar, 'SZ500'], [ONS, 'SZ500'],
           [Rmr, 'SH500'], [UP, 'SH500'], [EG, 'SH500'], [Pamr, 'SH500'], [Olmar, 'SH500'], [ONS, 'SH500'],
           [Rmr, 'SH50'], [UP, 'SH50'], [EG, 'SH50'], [Pamr, 'SH50'], [Olmar, 'SH50'], [ONS, 'SH50'],
           [Rmr, 'ZZ100'], [UP, 'ZZ100'], [EG, 'ZZ100'], [Pamr, 'ZZ100'], [Olmar, 'ZZ100'], [ONS, 'ZZ100']]
attribute = {}
bandit = BanditExp3(n_dataset=4, n_method=6)
for current_year in range(2012, 2020):
    for start_month in range(1, 13):
        end_month = start_month + 1
        diff_year = 0
        if end_month == 13:
            end_month = 1
            diff_year = 1
        stock = Stock(start_date=datetime.date(current_year, start_month, 1),
                      end_date=datetime.date(current_year + diff_year, end_month, 1))
        # GET INDEX FROM BANDIT
        current_index = bandit.choose()
        current_sr = []
        for index, [method, mode] in enumerate(bandits):
            print("{},{} | 当前正在进行{}/{}".format(current_year, start_month, index, len(bandits)))
            frame = stock.generate_data_frame(mode=mode, current_year=current_year)
            algo = method(n_stock=len(frame[0]))
            algo.compute_weight(frame)
            evaluate = Eval(relative_price=frame,
                            weight=algo.weights,
                            frequency='daily',
                            transaction_cost=0.000)
            # 获得每轮的Sharpe Ratio
            ratio = evaluate.sharp_ratio
            current_sr.append(ratio)
            # 获得每轮的累积财富
            cw = evaluate.cumulative_wealth
            attribute.setdefault(index, []).append(cw)
            if index == current_index:
                attribute.setdefault(len(bandits), []).append(cw)
        # 更新Bandit
        bandit.update(current_sr)

json.dump(attribute, open('all_algo_cw.json', 'w'))
