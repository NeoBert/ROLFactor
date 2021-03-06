# IC Rank Portfolio

import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath('result/statistic/weight')


class ICRank(object):
    # Exponential gradient portfolio method
    def __init__(self, n_stock, n_choose, n_window):
        """
        Variable:   n_stock: number of stock
                    n_choose: number of chosen factor
                    n_window: number of training window
                    name: name of method
                    weights: store historical weights
                    eta: learning rate
        """
        self.n_stock = n_stock
        self.n_choose = n_choose
        self.n_window = n_window
        self.name = 'ic_rank'
        self.weights = []
        self.rank = np.zeros((self.n_choose, self.n_stock))

    def compute_weight(self, relative_price, ic, fa_weight, fa_order, n_top):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-array (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
                    ic: float-array (n_time, n_factor)
                    fa_weight: 0/1-list (n_time, n_factor)
                    fa_order: boolean-list - (n_factor) True for positive correlation, False for negative
                    n_top: int - number of selected stock
        """
        for t in range(len(relative_price)):
            if t == 0:
                per_rp = []
                per_ic = []
                per_fa_weight = []
            elif t < self.n_window + 1:
                per_rp = relative_price[t - 1]
                per_ic = []
                per_fa_weight = []
            else:
                per_rp = relative_price[t-self.n_window:t]
                per_ic = ic[t-self.n_window-1:t]
                per_fa_weight = fa_weight[t]

            weight = self.weight_update(t, per_rp, per_ic, per_fa_weight, fa_order, n_top, "ic_rank")
            self.weights.append(weight)

    def __linear_regression(self, x, y, z):
        """
        Input:  x: feature array (n)
                y: label array (n)
                z: feature for prediction
        Output: w: weigth array (2)
        """
        x = np.vstack((np.ones((1, x.shape[0])), x))
        inv = np.linalg.inv(x @ x.T)
        w = np.dot(inv @ x, y)
        return w @ np.array([1, z])

    def weight_update(self, t, per_rp, per_ic, per_fa_weight, fa_order, n_top, model="ic_rank"):
        """
        Function:   update weight in each period
        Input:      t: int
                    per_rp: periodic relative price - array
                    per_ic: periodic ic - array
                    per_fa_weight: periodic factor weight - list
                    fa_order: boolean-list - (n_factor) True for positive correlation, False for negative
                    n_top: int - number of selected stock
                    model: str - "ic_rank", "linear_reg"
        Output:     weight: float-list (n_stock)
        """
        # equal weight in the first round
        if t == 0:
            weight = [1 / self.n_stock] * self.n_stock
            
        elif t < self.n_window + 1:
            """
            [0:n_window-1] for training
            [n_window] for prediction
            """
            # UBAH
            weight_array = np.array(self.weights[-1])
            price_array = per_rp
            weight_array = (weight_array * price_array) / (weight_array @ price_array)
            weight = list(w for w in weight_array)
            # UCRP
            weight = [1 / self.n_stock] * self.n_stock

        # following rounds
        else:
            chosen_fa = np.argwhere(np.array(per_fa_weight)==1)[:, 0]
            assert(len(chosen_fa) == self.n_choose)
            for i in range(self.n_choose):
                pred_prices = []
                x = per_ic[:self.n_window-1, chosen_fa[i]]  # (n_window)
                for j in range(self.n_stock):
                    y = per_rp[:,j]  # (n_window)
                    pred_prices.append(self.__linear_regression(x, y, per_ic[-1, chosen_fa[i]]))
                if model == "ic_rank":
                    if fa_order[i] == True:
                        sorted_idx = np.argsort(pred_prices)[::-1]
                        for j in range(self.n_stock):
                            self.rank[i][sorted_idx[j]] = j# + 1
                    elif fa_order[i] == False:
                        sorted_idx = np.argsort(pred_prices)
                        for j in range(self.n_stock):
                            self.rank[i][sorted_idx[j]] = j# + 1
                elif model == "linear_reg":
                    self.rank[i] = pred_prices
            if model == "ic_rank":    
                chosen_st = np.argsort(self.rank.sum(axis=0))
                weight = [0] * self.n_stock
                for i in range(n_top):
                    weight[chosen_st[i]] = 1 / n_top
            elif model == "linear_reg":
                weight = self.__softmax(self.rank.sum(axis=0))
                weight = weight.tolist()

        return weight

    def __softmax(self, weight):
        """
        Input:  list/array (n_stock)
        Output: array (n_stock)
        """
        w = np.exp(weight)
        return w / w.sum()

    def write_weight(self, file_name):
        """
        Function:   write weights
        Input       file_name: name of file
        """
        file_name = 'weight-' + file_name + '.csv'
        if exists(parent_dir_writer) == False:
            os.mkdir(parent_dir_writer)
        path = abspath(join(parent_dir_writer, file_name))
        pd_data = pd.DataFrame(self.weights)
        pd_data.to_csv(path, index=False, sep=',')