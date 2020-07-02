from data.stock import Stock
from data.factor import Factor
import datetime
from algorithm.olps.anticor import Anticor
from algorithm.olps.rmr import Rmr
from algorithm.olps.corn import CORN
from algorithm.olps.pamr import Pamr
from algorithm.olps.olmar import Olmar
<<<<<<< HEAD
=======
from algorithm.olps.bcrp import Bcrp
from algorithm.olps.best import Best
>>>>>>> 72137f4... add BCRP, BEST.
from eval import Eval
import numpy as np
# 获取该时间段下的所有数据
stock_mode = "factor"
factor = Factor()
frame = factor.generate_data_frame()
# 选择相应的算法
algo_name = "Olmar"
algo = Olmar(n_stock=len(frame[0]))

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
