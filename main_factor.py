from data.stock import Stock
from data.factor import Factor
import datetime
from algorithm.olps.factor_rmr import FaRmr
from algorithm.olps.factor_pamr import FaPamr
from algorithm.olps.factor_olmar import FaOlmar
from algorithm.olps.factor_bcrp import FaBcrp
from algorithm.olps.factor_best import FaBest
from algorithm.olps.factor_ucb import FaUcb
from algorithm.olps.factor_exp3 import FaExp3
from algorithm.olps.factor_awu import FaAwu
from algorithm.olps.factor_mwu import FaMwu
from algorithm.olps.factor_ewu import FaEwu
from algorithm.olps.factor_wmu import FaWmu
from eval_factor import EvalFactor
import numpy as np
import pandas as pd
# 获取该时间段下的所有数据
stock_mode = "factor"
factor = Factor()
frame = factor.generate_data_frame()
# 选择相应的算法
algo_list = ['FaBest', 'FaBcrp', 'FaPamr', 'FaOlmar', 'FaRmr', 'FaUcb', 'FaExp3', 'FaAwu', 'FaMwu', 'FaEwu', 'FaWmu']
result = pd.DataFrame()
for al in algo_list:
    algo_class = eval(al)
    reward = []
    for dup in range(10):
        algo = algo_class(n_factor=len(frame[0]))
        # 计算模型的权重
        algo.compute_weight(frame)
        algo.write_weight(algo.name + '-' + stock_mode)
        evaluate = EvalFactor(abs_ic=frame,
                            weight=algo.weights,)
        reward.append(evaluate.reward)
    # result[algo.name] = reward
    result[algo.name] = [np.mean(reward)]
print(result)