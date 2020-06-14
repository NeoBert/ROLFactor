from data.stock import Stock
from data.fund import Fund
import datetime
from algorithm.rmr import Rmr
from eval import Eval

# 获取该时间段下的所有数据
stock = Stock(start_date=datetime.date(2019, 1, 1), end_date=datetime.date(2020, 1, 1))
# fund = Fund(start_date=datetime.date(2019, 1, 1), end_date=datetime.date(2020, 1, 1))
# exit()
frame = stock.generate_data_frame(mode="random")

# algo = Pamr(n_stock=len(frame[0]))
# algo = Olmar(n_stock=len(frame[0]))
algo = Rmr(n_stock=len(frame[0]))
algo.compute_weight(frame)

evaluate = Eval(relative_price=frame,
                weight=algo.weights,
                frequency='daily',
                transaction_cost=0.003)
evaluate.print_info()
