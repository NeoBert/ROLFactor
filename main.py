from data.stock import Stock
import datetime
from algorithm.olps.anticor import Anticor
from algorithm.olps.rmr import Rmr
from algorithm.olps.corn import CORN
from eval import Eval

# 获取该时间段下的所有数据

stock_mode = "SH50"
stock = Stock(start_date=datetime.date(2019, 1, 1), end_date=datetime.date(2020, 1, 1))
frame = stock.generate_data_frame(mode=stock_mode)

# 选择相应的算法
algo_name = "CORN"
algo = CORN(n_stock=len(frame[0]))

# 计算模型的权重
algo.compute_weight(frame)
algo.write_weight(algo_name + '-' + stock_mode)
evaluate = Eval(relative_price=frame,
                weight=algo.weights,
                frequency='daily',
                transaction_cost=0.000)
evaluate.print_info()
evaluate.write_info(algo_name + '-' + stock_mode)
evaluate.write_cumulative_wealth(algo_name + '-' + stock_mode)
evaluate.write_periodic_return(algo_name + '-' + stock_mode)
