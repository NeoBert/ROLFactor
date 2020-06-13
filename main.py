from stock import Stock
import datetime
from algorithm.pamr import Pamr
from eval import Eval

# 获取该时间段下的所有数据
stock = Stock(start_date=datetime.date(2019, 1, 1), end_date=datetime.date(2020, 1, 1))
stock.generate_data_frame()

# 选择算法
algo = Pamr(n_stock=stock.nstock)
algo.compute_weight(stock.all_stock_price)

evaluate = Eval(relative_price=stock.all_stock_price,
                weight=algo.weights,
                frequency='daily',
                transaction_cost=0.003)
evaluate.print_info()
