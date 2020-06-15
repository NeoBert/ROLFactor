from data.stock import Stock
import datetime
from algorithm.rmr import Rmr
from algorithm.up import UP
from eval import Eval

# 获取该时间段下的所有数据
stock = Stock(start_date=datetime.date(2018, 1, 1), end_date=datetime.date(2020, 1, 1))
frame = stock.generate_data_frame(mode="random")
algo = UP(n_stock=len(frame[0]))
algo.compute_weight(frame)

evaluate = Eval(relative_price=frame,
                weight=algo.weights,
                frequency='daily',
                transaction_cost=0)
evaluate.print_info()
