from data.stock import Stock
from data.factor import Factor
import datetime
# from algorithm.olps.factor_rmr import FaRmr
# from algorithm.olps.factor_pamr import FaPamr
# from algorithm.olps.factor_olmar import FaOlmar
from algorithm.olps.factor_bcrp import FaBcrp
from algorithm.olps.factor_best import FaBest
from algorithm.olps.factor_ucb import FaUcb
from algorithm.olps.factor_exp3 import FaExp3
from algorithm.olps.factor_awu import FaAwu
from algorithm.olps.factor_mwu import FaMwu
from algorithm.olps.factor_ewu import FaEwu
from algorithm.olps.factor_wmu import FaWmu
from algorithm.olps.ic_rank import ICRank
from eval_factor import EvalFactor
from eval import Eval
import numpy as np
import pandas as pd
# ======================
# 获取数据
# ======================
# factor
factor = Factor()
fa_frame = factor.generate_data_frame()
n_choose = 5  # 选择因子数
fa_order = ['po', 'ne']  # 经验判断因子为正相关or负相关，'po': 正相关; 'ne': 负相关
                         # len = n_factor
# stock
stock = Stock()
st_frame = stock.generate_data_frame()
n_window = 22
n_top = 5  # 选择股票数

# ======================
# 选择相应的因子选择算法
# ======================
fa_algo_list = ['FaBest', 'FaBcrp', 'FaUcb', 'FaExp3', 'FaAwu', 'FaMwu', 'FaEwu', 'FaWmu']
fa_result = pd.DataFrame()

# ======================
# 迭代每一个因子选择算法
# ======================
for al in fa_algo_list:
    fa_algo_class = eval(al)
    reward = []
    # ------------------
    # 迭代重复实验
    # ------------------
    dup_max = 50
    for dup in range(dup_max):
        # ------------------
        # factor selection
        # ------------------
        fa_algo = fa_algo_class(n_factor=len(fa_frame[0]), 
                                n_choose=n_choose)
        # 计算模型的权重
        fa_algo.compute_weight(fa_frame)
        fa_algo.write_weight(fa_algo.name + '-factor')
        fa_evaluate = EvalFactor(abs_ic=fa_frame,
                                 weight=fa_algo.weights,)
        reward.append(fa_evaluate.reward)
        # ------------------
        # portfolio-IC rank
        # ------------------
        st_algo = ICRank(n_stock=len(st_frame[0]),
                         n_choose=n_choose,
                         n_window=n_window)
        st_algo.compute_weight(relative_price=st_frame,
                               ic=fa_frame,
                               fa_weight=fa_algo.weights,
                               fa_order=fa_order,
                               n_top=n_top)
        st_algo.write_weight(st_algo.name + '-stock')
        st_evaluate = Eval(relative_price=st_frame,
                           weight=st_algo.weights,
                           frequency='daily',
                           transaction_cost=0)
        st_evaluate.print_info()
        is_last = True if dup == dup_max - 1 else False
        st_evaluate.dup_write_info(fa_algo.name + '-stock', dup, is_last)  # dup_max+1 lines of data in total, last line is mean

    # fa_result[fa_algo.name] = reward
    fa_result[fa_algo.name] = [np.mean(reward)]

# ======================
# 打印结果
# ======================
print(fa_result)